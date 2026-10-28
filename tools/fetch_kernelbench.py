"""K1–K4: restore the committed upstream revision; never overwrite a changed checkout."""
import json
import subprocess
from env.tasks import DEFAULT_BENCH, LOCK, verified_catalog

def main():
    lock = json.loads(LOCK.read_text())
    if not DEFAULT_BENCH.exists():
        DEFAULT_BENCH.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(DEFAULT_BENCH)], check=True)
        subprocess.run(["git", "-C", str(DEFAULT_BENCH), "remote", "add", "origin", lock["url"]], check=True)
        subprocess.run(["git", "-C", str(DEFAULT_BENCH), "fetch", "--depth", "1", "origin", lock["revision"]], check=True)
        subprocess.run(["git", "-C", str(DEFAULT_BENCH), "checkout", "--detach", "FETCH_HEAD"], check=True)
    verified_catalog()
    print(f"Verified KernelBench {lock['revision']}")

if __name__ == "__main__":
    main()
