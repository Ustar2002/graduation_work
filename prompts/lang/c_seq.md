# Language: C (sequential, single-threaded)

## Environment

- C11 with GNU extensions (`-std=gnu11`). Compiler: `${cc}` — `${cc_version}`.
- Your reply is saved as `kernel.c` and compiled into a shared library with exactly this command:

```
${build_command}
```

- Only the C standard library (libc) and the math library (libm) are linked; the headers shipped with the compiler and the C library are available. Any symbol that is not defined in your file, libc or libm is a build error. Compiler warnings do not fail the build; errors do.
- Single-threaded: the command has no `-fopenmp`, so OpenMP pragmas are not compiled and no OpenMP runtime is linked. No threads of any kind (no pthreads), no child processes (no fork, system or popen). The function runs on exactly one core.
- The harness loads the shared library and looks up the symbol `gemm` by name, so `gemm` must have external linkage (not `static`) and the C calling convention.

## Function to implement

Define exactly:

```c
${signature}
```

Element access: ${indexing} for 0 <= i < M, 0 <= j < N, 0 <= k < K.

- `A`: M*K float32 values, read-only. `B`: K*N float32 values, read-only. `C`: M*N float32 values, the output buffer that must be filled. All three start at 64-byte-aligned addresses.
- `M`, `N`, `K`: the matrix dimensions, always equal to the actual buffer sizes.
- Return value: none (results are written into `C`).

## Execution environment

- Hardware: ${cpu_summary}.
- The process is pinned to ${cores_desc}. Thread count: ${threads}.
- The inputs are already in memory when `gemm` is called; the measured time is the duration of the call itself.

## Output format

Reply with exactly one fenced code block tagged `c` containing the complete `kernel.c`. No other code block.

---

${task_spec}
