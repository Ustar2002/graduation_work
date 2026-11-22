# Language: Python with Numba

## Environment

- Python ${python_version}, NumPy ${numpy_version}, Numba ${numba_version}.
- Use only the Python standard library, NumPy and Numba: do not import any other package, and do not install anything.
- Your reply is saved as a Python module. The harness imports the module and calls its top-level function `${function}` directly. Top-level code other than imports and definitions is not allowed.
- `${function}` must be compiled by Numba in nopython mode with an explicit type signature: it carries the decorator `njit` (`numba.njit`), called with exactly the signature string shown below as its first argument. Keyword options of the decorator, after the signature string, are your choice.
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
- The process is pinned to ${cores_desc}. Thread count: ${threads}. The environment sets `NUMBA_NUM_THREADS=${threads}` and `NUMBA_THREADING_LAYER=${numba_threading_layer}`.
- The inputs are already in memory as NumPy arrays when `${function}` is called; the measured time is the duration of the call itself.

## Output format

Reply with exactly one fenced code block tagged `python` containing the complete module. No other code block.

${grammar_hint}---

${task_spec}
