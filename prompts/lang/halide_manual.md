# Language: Halide (Python bindings, algorithm and schedule)

## Environment

- Python ${python_version}, NumPy ${numpy_version}, Halide ${halide_version} (the `halide` Python package, imported as `import halide as hl`).
- Use only the Python standard library, NumPy and Halide: do not import any other package, and do not install anything.
- Your reply is saved as a Python module that defines, at top level, a pipeline-building function named `${function}`. Top-level code other than imports and definitions is not allowed.
- `${function}` does not receive array data and has no output parameter. For each input shape the harness builds a new pipeline by calling `${function}` with an `hl.ImageParam` for every input array and the actual sizes of that shape as Python `int` values. `${function}` must return the `hl.Func` that defines the output array.
- Coordinate order: the harness wraps every array with `hl.Buffer(array)` (default `reverse_axes=True`), so the last array axis is the first Halide coordinate. For a row-major array of shape (D0, D1), the element at row i0 and column i1 is `X[i1, i0]`; a 1-D array is indexed as `x[i]`. ${halide_coord_example} The returned `hl.Func` is defined with the same rule.
- The harness compiles the pipeline for the host target with Halide's JIT compiler. Where the task specification below speaks of a call of the function, that call is one realization of the returned `hl.Func` over the full output shape, written directly into the output buffer allocated by the caller. Where the specification lists the output buffer among the parameters, the signature below applies instead.
- No autoscheduler is applied: the pipeline is compiled with exactly the schedule that `${function}` gives to the `hl.Func` objects it creates. Scheduling calls, if any, are written inside `${function}`, before it returns.

## Function to implement

Define, at module top level, exactly:

```python
${signature}
```

Parameters:

${param_docs}${args_note}

## Execution environment

- Hardware: ${cpu_summary}.
- The process is pinned to ${cores_desc}. Thread count: ${threads}. The environment sets `HL_NUM_THREADS=${threads}`.
- The inputs are already in memory when the pipeline is realized. Building the pipeline, scheduling it and compiling it happen before the measured realizations and are not measured; the measured time is the duration of one realization.

## Output format

Reply with exactly one fenced code block tagged `python` containing the complete module. No other code block.

${grammar_hint}---

${task_spec}
