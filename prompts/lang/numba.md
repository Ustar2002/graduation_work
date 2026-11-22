# Language: Python with Numba

## Environment

- Python ${python_version}, NumPy ${numpy_version}, Numba ${numba_version}.
- Use only the Python standard library, NumPy and Numba: do not import any other package, and do not install anything.
- Your reply is saved as a Python module. The harness imports the module and calls its top-level function `${function}` directly. Top-level code other than imports and definitions is not allowed.
- `${function}` must be compiled by Numba in nopython mode with an explicit type signature: it carries the decorator `njit` (`numba.njit`), called with exactly the signature string shown below as its first argument. Keyword options of the decorator, after the signature string, are your choice, except that `cache=True` is not allowed.
- Every other function in the module that is decorated with `njit` or `jit` must also be given an explicit signature string as the first positional argument of its decorator, so that nothing is compiled during a call. A compiled function without one, or any use of `cache=True`, is rejected before the module is imported.
- Because of the explicit signature, Numba compiles `${function}` when the module is imported. The import, including this compilation, takes place before the first call and is not part of the measured time. A compilation error is an import error.
- SciPy is not installed. Numba's implementations of `np.dot`, `np.vdot`, the `@` operator and `np.linalg` depend on it, so they are not available inside compiled functions.

## Function to implement

Define, at module top level, exactly:

```python
${signature}
```

Parameters:

${param_docs}${args_note}

The harness passes every array as a writable, C-contiguous NumPy array of the stated dtype (float32 or int32), which is what the array types in the signature accept, and every size as a Python `int`. Arrays described as read-only must still not be modified; a modification is detected and counts as a failure.

## Execution environment

- Hardware: ${cpu_summary}.
- The process is pinned to ${cores_desc}. Thread count: ${threads}. The environment sets `NUMBA_NUM_THREADS=${threads}`, `NUMBA_THREADING_LAYER=${numba_threading_layer}`, `OMP_NUM_THREADS=${threads}`, `OMP_PROC_BIND=true`, `OMP_PLACES=cores` and `OMP_DYNAMIC=false`.
- The inputs are already in memory as NumPy arrays when `${function}` is called; the measured time is the duration of the call itself.

## Output format

Reply with exactly one fenced code block tagged `python` containing the complete module. No other code block.

${grammar_hint}---

${task_spec}
