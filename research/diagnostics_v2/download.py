"""CPU-only, resumable downloads pinned to upstream and mirror SHA-256 identities."""
import argparse
import hashlib
from http.client import IncompleteRead
import json
from pathlib import Path
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait

from research.diagnostics_v1.common import ROOT, digest, save, utc

MANIFEST = ROOT / 'experiments/gemma31b-weights-v6.json'
BASE = ROOT / 'runs/diagnostics-v6/preparation'


def _fetch_range_once(url, start, end, destination):
    if destination.exists() and destination.stat().st_size == end - start + 1:
        return
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}',
                                                  'User-Agent': 'cuda-diagnostic-research/1'})
    temporary = destination.with_suffix('.incomplete')
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
            raise RuntimeError('Range download is not the exact requested segment')
        count = 0
        with temporary.open('wb') as stream:
            while chunk := response.read(1 << 20):
                count += len(chunk)
                if count > end - start + 1:
                    raise RuntimeError('Range response exceeds requested length')
                stream.write(chunk)
        if count != end - start + 1:
            raise RuntimeError('Truncated range response')
    temporary.replace(destination)


def fetch_range(url, start, end, destination):
    """Bound network retries per segment, not across an entire 50 GB shard."""
    for attempt in range(5):
        try:
            _fetch_range_once(url, start, end, destination)
            return
        except (OSError, IncompleteRead) as error:
            print(json.dumps({'utc': utc(), 'segment': [start, end],
                              'network_attempt': attempt + 1,
                              'error': f'{type(error).__name__}: {error}'}), flush=True)
            if attempt == 4:
                raise
            time.sleep(min(15, 2 ** attempt))


def parallel_resume(url, part, total, progress, *, workers=4, segment_bytes=64 << 20):
    """Four bounded range files; only contiguous verified-length bytes are appended."""
    ranges = part.with_suffix(part.suffix + '.ranges'); ranges.mkdir(exist_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while offset < total:
            segments = [(begin, min(total - 1, begin + segment_bytes - 1))
                        for begin in range(offset, min(total, offset + workers * segment_bytes), segment_bytes)]
            paths = [ranges / f'{begin}-{end}.range' for begin, end in segments]
            futures = [pool.submit(fetch_range, url, begin, end, path)
                       for (begin, end), path in zip(segments, paths, strict=True)]
            pending = set(futures)
            while pending:
                _, pending = wait(pending, timeout=15)
                def size(path):
                    try:
                        return path.stat().st_size
                    except FileNotFoundError:
                        return 0
                in_flight = sum(max(size(path), size(path.with_suffix('.incomplete'))) for path in paths)
                progress(offset + in_flight)
            for future in futures:
                future.result()
            with part.open('ab') as stream:
                for (begin, end), path in zip(segments, paths, strict=True):
                    if begin != offset or path.stat().st_size != end - begin + 1:
                        raise ValueError('Noncontiguous range assembly')
                    with path.open('rb') as source:
                        shutil.copyfileobj(source, stream, length=1 << 20)
                    stream.flush()
                    import os
                    os.fsync(stream.fileno())
                    offset = end + 1
                    path.unlink()
            progress(offset)


def download(manifest=MANIFEST, destination=None):
    manifest = json.loads(Path(manifest).read_text())
    dest = Path(destination or ROOT / 'models/gemma-4-31B-it')
    dest.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    if manifest['repo'] != 'google/gemma-4-31B-it':
        raise ValueError('Unexpected model repository')
    missing = 0
    for item in manifest['files']:
        if Path(item['path']).name != item['path']:
            raise ValueError('Only flat model files permitted')
        final = dest / item['path']; partial = final.with_suffix(final.suffix + '.part')
        present = final.stat().st_size if final.exists() else partial.stat().st_size if partial.exists() else 0
        missing += max(0, item['bytes'] - present)
    if shutil.disk_usage(dest).free < missing + (12 << 30):
        raise RuntimeError('Download requires all missing bytes plus 12 GiB working headroom')
    completed = []
    for item in manifest['files']:
        final = dest / item['path']; part = final.with_suffix(final.suffix + '.part')
        if final.exists():
            if final.stat().st_size != item['bytes'] or digest(final) != item['sha256']:
                raise ValueError('Existing model file identity mismatch: ' + final.name)
        else:
            url = ('https://modelscope.cn/models/' + manifest['repo'] + '/resolve/'
                   + item['modelscope_revision'] + '/' + item['path'])
            for attempt in range(5):
                offset = part.stat().st_size if part.exists() else 0
                if offset > item['bytes']:
                    raise ValueError('Oversized partial file')
                if offset == item['bytes']:
                    break
                try:
                    if item['bytes'] > 256 << 20:
                        def progress(count):
                            save(BASE / 'download-progress.json', {'status': 'downloading', 'utc': utc(),
                                'file': item['path'], 'file_bytes': count, 'file_total': item['bytes'],
                                'completed_bytes': sum(x['bytes'] for x in completed),
                                'total_bytes': manifest['bytes'], 'attempt': attempt + 1, 'range_workers': 4})
                        parallel_resume(url, part, item['bytes'], progress)
                        break
                    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-',
                                                                  'User-Agent': 'cuda-diagnostic-research/1'})
                    with urllib.request.urlopen(request, timeout=90) as response:
                        if response.status == 206:
                            if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                                raise ValueError('Download range does not match resumed offset')
                        elif response.status == 200:
                            offset = 0
                        else:
                            raise ValueError('Unexpected download status')
                        last = 0.0
                        with part.open('ab' if offset else 'wb') as stream:
                            while True:
                                chunk = response.read(4 << 20)
                                if not chunk:
                                    break
                                stream.write(chunk); offset += len(chunk)
                                if offset > item['bytes']:
                                    raise ValueError('Server sent too many bytes')
                                if time.monotonic() - last > 15:
                                    save(BASE / 'download-progress.json', {'status': 'downloading', 'utc': utc(),
                                        'file': item['path'], 'file_bytes': offset, 'file_total': item['bytes'],
                                        'completed_bytes': sum(x['bytes'] for x in completed),
                                        'total_bytes': manifest['bytes'], 'attempt': attempt + 1})
                                    last = time.monotonic()
                    if offset != item['bytes']:
                        raise RuntimeError('Incomplete download response')
                    break
                except (OSError, TimeoutError, RuntimeError) as error:
                    save(BASE / 'download-last-error.json', {'utc': utc(), 'file': item['path'],
                                                           'error': str(error), 'attempt': attempt + 1})
                    if attempt == 4:
                        raise
                    time.sleep(min(15, 2 ** attempt))
            save(BASE / 'download-progress.json', {'status': 'verifying', 'utc': utc(), 'file': item['path']})
            if part.stat().st_size != item['bytes'] or digest(part) != item['sha256']:
                raise ValueError('Downloaded weight SHA-256 mismatch: ' + item['path'])
            part.replace(final)
        completed.append(item)
        print(json.dumps({'verified': item['path'], 'bytes': item['bytes'], 'utc': utc()}), flush=True)
    from tools.weights import inspect_weights
    receipt = {'status': 'CPU_weights_verified_GPU_load_pending', 'utc': utc(),
               'manifest_sha256': digest(MANIFEST), 'model': manifest['repo'],
               'upstream_revision': manifest['hf_revision'], 'files': completed,
               'structure': inspect_weights(dest), 'free_GiB': shutil.disk_usage(dest).free / 2**30}
    save(BASE / 'download-receipt.json', receipt)
    save(BASE / 'download-progress.json', {'status': 'complete', 'utc': utc(), 'bytes': manifest['bytes']})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        download()
    except Exception as error:
        save(BASE / 'download-progress.json', {'status': 'stopped', 'utc': utc(),
                                             'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
