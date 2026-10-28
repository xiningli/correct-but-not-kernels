"""K3/C1: exhaust this exclusive GPU, then exit; supervisor checks reclamation."""
import sys
import torch
buffers = []
try:
    while True:
        buffers.append(torch.empty(256 << 20, dtype=torch.uint8, device='cuda'))
except torch.OutOfMemoryError:
    print('CUDA_OOM_REACHED', len(buffers), flush=True)
    sys.exit(77)
