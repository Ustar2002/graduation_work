# Language: Rust

## Environment

- Rust, edition 2021. Compiler: `${rustc_version}`.
- Your reply is saved as `src/lib.rs` of a library crate that is built as a C-compatible shared library (`crate-type = ["cdylib"]`). `Cargo.toml` and `Cargo.lock` are supplied by the harness and cannot be changed by your reply. The crate is built with exactly this command:

```
${cargo_command}
```

- Release profile: ${rust_profile}.
- The crates available to your code are the Rust standard library (`std`) and `rayon` ${rayon_version}. No other dependency is declared in `Cargo.toml`, so the use of any other crate is a build error. Compiler warnings do not fail the build; errors do.
- A panic aborts the process and is recorded as a crash.
- The harness loads the shared library, looks up the symbol `${function}` by name and calls it through the C ABI, so `${function}` must be exported unmangled with the C calling convention, as in the signature below.

## Function to implement

Define exactly:

```rust
${signature}
```

${param_docs}${args_note}

Every pointer passed by the harness is non-null and valid, for the whole duration of the call, for the number of elements stated above: for reads if it refers to an input, for reads and writes if it refers to the output. The parameter names must be kept exactly as given, even where they are not snake_case; the crate-level attribute `#![allow(non_snake_case)]` silences the compiler's naming warning.

## Execution environment

- Hardware: ${cpu_summary}.
- The process is pinned to ${cores_desc}. Thread count: ${threads}. The environment sets `RAYON_NUM_THREADS=${threads}`.
- The inputs are already in memory when `${function}` is called; the measured time is the duration of the call itself.

## Output format

Reply with exactly one fenced code block tagged `rust` containing the complete `src/lib.rs`. No other code block.

${grammar_hint}---

${task_spec}
