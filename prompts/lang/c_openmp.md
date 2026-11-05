# Language: C with OpenMP

## Environment

- C11 with GNU extensions (`-std=gnu11`). Compiler: `${cc}` — `${cc_version}`.
- Your reply is saved as `kernel.c` and compiled into a shared library with exactly this command:

```
${build_command}
```

- Only the C standard library (libc), the math library (libm) and the OpenMP runtime (GNU libgomp, via `-fopenmp`) are linked; the headers shipped with the compiler and the C library are available, including `<omp.h>`. Any symbol that is not defined in your file, libc, libm or the OpenMP runtime is a build error. Compiler warnings do not fail the build; errors do.
- OpenMP is enabled and is the only permitted form of parallelism: `#pragma omp` directives and the OpenMP runtime functions may be used. No other threads (no pthreads) and no child processes (no fork, system or popen).
- The harness loads the shared library and looks up the symbol `${function}` by name, so `${function}` must have external linkage (not `static`) and the C calling convention.

## Function to implement

Define exactly:

```c
${signature}
```

${param_docs}

## Execution environment

- Hardware: ${cpu_summary}.
- The process is pinned to ${cores_desc}. Thread count: ${threads}. The environment sets `OMP_NUM_THREADS=${threads}`, `OMP_PROC_BIND=true`, `OMP_PLACES=cores` and `OMP_DYNAMIC=false`.
- The inputs are already in memory when `${function}` is called; the measured time is the duration of the call itself.

## Output format

Reply with exactly one fenced code block tagged `c` containing the complete `kernel.c`. No other code block.

---

${task_spec}
