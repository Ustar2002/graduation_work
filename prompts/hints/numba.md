These notes describe Numba's nopython mode (`numba.njit`). They are the same for every task.

**Signature strings**

```python
import numpy as np
from numba import njit, prange

@njit("int64(float32[:, ::1], int32[::1], int64)")
def count_selected(values, flags, n):
    count = 0
    for i in range(n):
        if flags[i] != 0 and values[i, 0] > 0.0:
            count += 1
    return count
```

- A signature string has the form `"return_type(argument_type, ...)"`; `void` is the return type of a function that returns nothing.
- Scalar types: `int32`, `int64`, `float32`, `float64`, `boolean`. A Python `int` argument is accepted for `int64` and a Python `float` for `float64`.
- Array types: `float32[::1]` is a 1-D C-contiguous array and `float32[:, ::1]` a 2-D C-contiguous array (one `:` for each leading dimension and `::1` for the last); `float32[:]` and `float32[:, :]` accept any layout. Every scalar type has the same array forms (`int32[::1]`).
- A function with an explicit signature is compiled when its decorator is executed and accepts only arguments of those types: an array of another dtype or number of dimensions, a non-contiguous array or a read-only array does not match a C-contiguous array type.
- Keyword options follow the signature string: `@njit("void(float32[::1], int64)", parallel=True)`.

**Values and indexing**

- A Python `float` literal is float64 and a Python `int` literal is int64; loop indices, `len()` and `shape` entries are int64. `np.float32(v)` and `np.int32(v)` convert a value, and a value stored into an array element is converted to the dtype of the array.
- Array indexing is not bounds-checked: an out-of-range index reads or writes other memory without an error. Negative indices count from the end, as in NumPy.

**NumPy support**

- Only part of NumPy is implemented, and a supported function may accept fewer arguments than in NumPy (`np.sum(a, axis=0)` is accepted, the `keepdims=` argument is not). An unsupported function or argument is a compilation error.
- Shapes are tuples, not lists: `np.empty((n, m), dtype=np.float32)`. Without `dtype=` a new array is float64.
- A ufunc takes its output array as a positional argument (`np.add(a, b, out)`); the keyword form `out=` is not accepted. Ufunc methods such as `np.add.reduce` and `np.multiply.outer` are not available.
- Basic slices are views of the original array.

**prange**

```python
@njit("int64(float32[::1], float32[::1], int64)", parallel=True)
def negate_and_count(src, dst, n):
    count = 0
    for i in prange(n):
        dst[i] = -src[i]
        if src[i] > 0.0:
            count += 1
    return count
```

- `prange` runs the iterations of a loop in parallel only inside a function compiled with `parallel=True`; elsewhere it behaves like `range`. Its step must be 1. In a nest of `prange` loops only the outermost one is parallel.
- Reduction rule: a scalar that is updated in the loop body from its own previous value with `+=`, `-=`, `*=`, `/=` (or `x = x + ...`), or with `max()` or `min()`, is a reduction. Each thread accumulates its own copy and the copies are combined after the loop.
- A scalar that is assigned in the loop body in any other way is private to the loop: after the loop it does not hold a value assigned inside it. Writes to array elements are not checked: different iterations must write different elements.

**Not supported in nopython mode**

- Calling a Python function that is not compiled: a helper must be decorated with `@njit` too, with its own explicit signature string, and it must be defined above the function that calls it, because the caller is compiled as soon as its decorator is executed.
- Python classes and objects of arbitrary types, generator expressions (list comprehensions are accepted), and `except` clauses that name a specific exception class (`except Exception:` is accepted).
- A tuple whose items have different types cannot be iterated or indexed with a variable index.
