# Language: C with OpenMP

## Environment

- C11 with GNU extensions (`-std=gnu11`). Compiler: `${cc}` — `${cc_version}`.
- Your reply is saved as `kernel.c` and compiled into a shared library with exactly this command:

```
${build_command}
```

- Only the C standard library (libc), the math library (libm) and the OpenMP runtime (GNU libgomp, via `-fopenmp`) are linked; the headers shipped with the compiler and the C library are available, including `<omp.h>`. Any symbol that is not defined in your file, libc, libm or the OpenMP runtime is a build error. Compiler warnings do not fail the build; errors do.
- OpenMP is enabled and is the only permitted form of parallelism: `#pragma omp` directives and the OpenMP runtime functions may be used. No other threads (no pthreads) and no child processes (no fork, system or popen).
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
- The process is pinned to ${cores_desc}. Thread count: ${threads}. The environment sets `OMP_NUM_THREADS=${threads}`, `OMP_PROC_BIND=true`, `OMP_PLACES=cores` and `OMP_DYNAMIC=false`.
- The inputs are already in memory when `gemm` is called; the measured time is the duration of the call itself.

## Output format

Reply with exactly one fenced code block tagged `c` containing the complete `kernel.c`. No other code block.

---

${task_spec}
