# Kernel task contract (K1–K4; frozen before C4)

Optimize the supplied reference into a GPU implementation that is both correct
and faster than eager PyTorch. Target an NVIDIA RTX PRO 6000 Blackwell GPU.
Look for kernel fusion, fewer memory passes, and efficient parallel reductions.
Prefer a custom Triton or CUDA kernel when it can improve the reference; simply
renaming the reference class or repeating its operator sequence usually leaves
the optimization objective unsolved. PyTorch operations remain allowed where
they are useful. Preserve the full numerical contract while optimizing.

Return one fenced Python program defining `ModelNew(torch.nn.Module)`.
The constructor and forward signatures, shapes, dtypes and output structure must
match the supplied PyTorch reference. A file may contain CUDA C++ or Triton code.
Preserve outputs across subsequent invocations and do not modify caller inputs.
Do not access reference/harness files, precompute answers, alter clocks/streams to
evade timing, use the network, or communicate with other rollouts.
Speed is compared with eager PyTorch on the same GPU in the same session.
You may use PyTorch operations, inline CUDA C++ through torch.utils.cpp_extension,
or Triton. Do not wrap torch.compile, TorchScript, or an external compiler service.
Compute from the actual runtime input on every call: no memoization, fixed random
seeds, hardcoded outputs or lookup tables. Use the current CUDA stream; do not
create auxiliary streams or manipulate events/timers. Do not introspect the
harness, import inspect/sys/ctypes/socket/subprocess, open files, or monkeypatch
Python/PyTorch APIs. Compiler-internal file access is handled by the sandbox.

Preserve shape and dtype exactly; all outputs must be finite. Floating point
outputs use rtol=1e-3, atol=1e-4; integer and boolean outputs must match exactly.
Output tensors must own their storage and remain unchanged when caller inputs
are reused or modified. Eager reference and candidate are timed after compilation
and warmup using CUDA events, an L2 flush, all-stream synchronization, and the
median of repeated measurements. Inputs and model initialization are fresh and
controlled externally; never reset the random seed yourself. The MinGPT GELU
task accepts signed as well as positive floating point inputs of the given shape.
