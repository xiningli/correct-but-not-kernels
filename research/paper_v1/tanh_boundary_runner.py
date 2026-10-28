"""Container-side runner for research.paper_v1.tanh_boundary_probe; sees only /input and /out."""
import importlib.util
import json
import torch

# Triton's @jit needs a real source file, so import rather than exec.
spec = importlib.util.spec_from_file_location('candidate', '/input/program.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
model = candidate.ModelNew().cuda()
cases = torch.load('/input/cases.pt')
outputs = []
with torch.no_grad():
    for x, labels in cases:
        outputs.append(model(x.cuda(), labels.cuda()).cpu())
torch.cuda.synchronize()
torch.save(outputs, '/out/outputs.pt')
json.dump({'device': torch.cuda.get_device_name(), 'torch': torch.__version__}, open('/out/runner.json', 'w'))
