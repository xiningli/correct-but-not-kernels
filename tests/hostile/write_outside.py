"""K3/C1: test both an ordinary outside file and a proc-fd path to it."""
import json
import sys
paths = json.load(open('targets.json'))['write']
for path in paths:
    try:
        with open(path, 'w') as f:
            f.write('ESCAPE')
    except PermissionError:
        continue
    print('ESCAPE: outside write', flush=True)
    sys.exit(0)
print('OUTSIDE_WRITE_BLOCKED', flush=True)
sys.exit(78)
