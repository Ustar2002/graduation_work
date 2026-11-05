# Language: Python with NumPy

## Environment

- Python ${python_version}, NumPy ${numpy_version}.
- Use only the Python standard library and NumPy: do not import any other package, and do not install anything.
- Your reply is saved as a Python module. The harness imports the module and calls its top-level function `gemm` directly. Top-level code other than imports and definitions is not allowed.

## Function to implement

Define, at module top level, exactly:

```python
${signature}
```

Parameters:

- `A`: `numpy.ndarray`, dtype float32, shape (M, K), C-contiguous, 64-byte aligned, read-only.
- `B`: `numpy.ndarray`, dtype float32, shape (K, N), C-contiguous, 64-byte aligned, read-only.
- `C`: `numpy.ndarray`, dtype float32, shape (M, N), C-contiguous, 64-byte aligned; the output buffer that must be filled.
- `M`, `N`, `K`: Python `int`; the matrix dimensions, always equal to the array shapes above.
- Returns: `None` (results are written into `C`).

## Execution environment

- Hardware: ${cpu_summary}.
- The process is pinned to ${cores_desc}. Thread count: ${threads}. `OMP_NUM_THREADS=${threads}` is set, and the thread-count environment variables honoured by NumPy's bundled numerical backend are set to the same value.
- The inputs are already in memory as NumPy arrays when `gemm` is called; the measured time is the duration of the call itself.

## Output format

Reply with exactly one fenced code block tagged `python` containing the complete module. No other code block.

---

${task_spec}
