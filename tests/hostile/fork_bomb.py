"""K3/C1: actual recursive fork pressure, bounded by kernel RLIMIT_NPROC."""
import errno
import os
import sys
import time
original = os.getpid()
while True:
    try:
        if os.fork() == 0:
            os.setpgid(0, 0)  # exercise Ninja-compatible cross-group cleanup
    except OSError as error:
        if error.errno != errno.EAGAIN:
            raise
        if os.getpid() == original:
            print('PROCESS_LIMIT_REACHED', flush=True)
            sys.exit(79)
        time.sleep(60)
