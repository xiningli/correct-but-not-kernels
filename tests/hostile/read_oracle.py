"""K3/C1: locked reference, timing harness and secret expected value stay private."""
import json
import sys
for path in json.load(open('targets.json'))['read']:
    try:
        with open(path) as f:
            f.read(1)
    except PermissionError:
        continue
    print('ESCAPE: oracle readable', flush=True)
    sys.exit(0)
print('ORACLE_READ_BLOCKED', flush=True)
sys.exit(78)
