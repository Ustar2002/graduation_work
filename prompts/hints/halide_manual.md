The pipeline is written with the Halide Python API (`import halide as hl`). These notes are the same for every task.

**Vars, Funcs and pure definitions**

```python
x, y = hl.Var("x"), hl.Var("y")
f = hl.Func("f")
f[x, y] = hl.f32(x) + hl.f32(y) * 0.5
g = hl.Func("g")
g[x, y] = f[x, y] * 2 + f[x + 1, y]
```

- A `hl.Var` is a free integer coordinate of type int32; two Vars with the same name are the same Var. A `hl.Func` maps integer coordinates to a value. Its first definition, the pure definition, has only distinct `hl.Var`s on the left-hand side and defines the Func at every coordinate.
- The number of dimensions of a Func is the number of coordinates of its pure definition; its element type is the type of that definition's right-hand side. A Func with zero dimensions is indexed as `f[()]`.
- Funcs, `hl.ImageParam`s and `hl.Buffer`s are read with the same indexing syntax, with any integer expressions as coordinates: `f[x + 1, y]`, `img[hl.clamp(x, 0, 9), y]`.
- Python statements run once, while the pipeline is being built: a Python `for` loop can create a chain of Funcs, and Python `int` values can be used as constants. A Halide expression has no truth value, so Python `if`, `and`, `or`, `not` and chained comparisons such as `0 < x < n` cannot be applied to it; use `hl.select` and the operators `&`, `|`, `~`.
- The Python built-ins `abs`, `min` and `max` and the functions of `math` and NumPy do not accept Halide expressions; the `hl.` functions listed below do.

**Coordinate order**

- A buffer made from a NumPy array (`hl.Buffer(array)`) has its axes reversed: the last array axis is the first Halide coordinate. For an array `a` of shape (D0, D1), the element `a[i0, i1]` is `img[i1, i0]`; `img.dim(0)` is the axis of length D1 and `img.dim(1)` the axis of length D0. A Func that defines such an array is written `out[i1, i0] = ...`. For a 1-D array, `a[i]` is `img[i]`.
- `hl.Buffer(array)` refers to the memory of the NumPy array without copying it, and a pipeline that reads the buffer does not keep the array alive: once the array and the `hl.Buffer` object are garbage-collected (for example when the function that created them returns), the pipeline reads freed memory. `hl.Buffer(array).copy()` returns a buffer that owns its data.

**Types, casts and constants**

- Types: `hl.Float(32)`, `hl.Float(64)`, `hl.Int(32)`, `hl.Int(64)`, `hl.UInt(8)`, `hl.Bool()`.
- `hl.f32(e)`, `hl.f64(e)`, `hl.i32(e)`, `hl.i64(e)` convert an expression or a Python number to that type; `hl.cast(hl.Float(32), e)` is the general form. Converting a floating-point value to an integer type truncates toward zero.
- A Python `float` is a float32 constant and a Python `int` an int32 constant. In mixed arithmetic an integer operand is converted to the floating-point type of the other operand, the narrower of two floating-point operands is widened, and an integer constant takes the type of the other operand.
- Integer `/` rounds toward negative infinity and integer `%` is never negative. Comparisons give `hl.Bool()` expressions.
- Elementwise math: `hl.sqrt`, `hl.abs`, `hl.min`, `hl.max`, `hl.floor`, `hl.ceil`, `hl.round`, `hl.exp`, `hl.log`, `hl.pow`, `hl.sin`, `hl.cos`.

**Reduction domains and update definitions**

```python
n, m = 6, 4
r = hl.RDom([(0, n)])                     # r.x takes the values 0 .. n-1; each entry is (min, extent)
c = hl.Func("c")
c[y] = 0                                  # pure definition
c[y] += hl.select(f[r.x, y] > 2, 1, 0)    # update definition, applied for r.x = 0, 1, ..., n-1 in order
r2 = hl.RDom([(0, n), (1, m - 1)])        # r2.x = 0 .. n-1 (inner loop), r2.y = 1 .. m-1 (outer loop)
t = hl.Func("t")
t[x, y] = f[x, y]
t[r2.x, r2.y] = t[r2.x, r2.y - 1] * 0.5 + f[r2.x, r2.y]   # an update may read values written by earlier iterations
```

- An update definition follows the pure definition of the same Func and must have the same element type. It may use `=`, `+=`, `-=`, `*=` or `/=`. On its left-hand side a Var may appear only as itself, in the position it has in the pure definition; the other coordinates are integer constants, RDom components or integer expressions of RDom components.
- The components of an RDom are `r.x`, `r.y`, `r.z`, `r.w` (also `r[0]`, `r[1]`, ...), of type int32. An RDom is iterated in order, with its first component innermost.
- An update definition writes every coordinate that its left-hand side reaches, whatever region is requested: realizing the Func into a buffer that does not contain all of them is an error.
- A coordinate computed from data needs provable bounds where it indexes an input or the Func being updated, as in `table[hl.clamp(idx[x], 0, n - 1)]`; without them the pipeline does not compile.
- Inline reductions over the RDom components that appear in the argument: `hl.sum(e)`, `hl.product(e)`, `hl.maximum(e)`, `hl.minimum(e)`; `hl.argmax(e)[0]` and `hl.argmin(e)[0]` give the position.

**Select, clamp and boundaries**

- `hl.select(cond, a, b)` evaluates to `a` where `cond` holds and to `b` elsewhere; `hl.select(c1, a, c2, b, c)` is the multi-way form.
- `hl.clamp(e, lo, hi)` limits `e` to the range lo .. hi (inclusive).
- From the requested output region, Halide infers the region of every input that is read. Reading an input outside its bounds is an error at run time: `img[x - 1, y]` over the full output reads x = -1.
- `hl.BoundaryConditions.repeat_edge(img)`, `constant_exterior(img, value)`, `mirror_image(img)` and `mirror_interior(img)` return a Func that equals `img` inside its bounds and is defined everywhere. For a Func instead of an input the bounds are given explicitly: `hl.BoundaryConditions.repeat_edge(f, [(0, w), (0, h)])`.

**Scheduling**

```python
xo, xi = hl.Var("xo"), hl.Var("xi")
g.split(x, xo, xi, 8, hl.TailStrategy.GuardWithIf)   # loop x becomes an outer loop xo and an inner loop xi of 8 iterations
g.reorder(xi, xo, y)                                 # loop order, innermost first
g.vectorize(xi)                                      # the iterations of xi are evaluated as vector operations
g.parallel(y)                                        # the iterations of y are distributed over the thread pool
f.compute_root()                                     # f is computed and stored once, before its consumers
c.update(0).reorder(y, r.x)                          # update definition 0 of c is scheduled separately
```

- The pure definition of a Func is scheduled through the Func itself and update definition number k through `f.update(k)`; each definition has its own loop nest. By default the first coordinate is the innermost loop and, in an update definition, the RDom loops are inside the loops of the Vars.
- Loop transformations: `f.reorder(inner, ..., outer)`, `f.split(v, outer, inner, factor)`, `f.tile(x, y, xo, yo, xi, yi, fx, fy)`, `f.fuse(inner, outer, fused)`, `f.vectorize(v)` or `f.vectorize(v, width)`, `f.unroll(v)` or `f.unroll(v, factor)`, `f.parallel(v)`, `f.serial(v)`. Each call returns the Func or the update stage it was applied to, so calls can be chained. `vectorize(v)` and `unroll(v)` without a width or factor need a loop of constant extent, such as the inner loop of a split.
- `split`, `tile`, `vectorize(v, width)` and `unroll(v, factor)` take a tail strategy as an optional last argument; it applies when the factor does not divide the extent of the loop. `hl.TailStrategy.GuardWithIf` evaluates exactly the requested region. `hl.TailStrategy.ShiftInwards` moves the last iterations back inside the region, evaluating some points twice; it is not allowed in update definitions. `hl.TailStrategy.RoundUp` evaluates up to the next multiple of the factor, beyond the requested region. The default is `ShiftInwards` in a pure definition, `RoundUp` for a Var of an update definition and `GuardWithIf` for an RDom loop.
- The loop of an RDom component (`r.x`) can be split, reordered and unrolled like the loop of a Var. Making it parallel or vectorized is rejected when Halide cannot prove that its iterations are independent; `.atomic()` (accepted only for updates that Halide can prove associative) or `.allow_race_conditions()` on the same definition lifts the check.
- Placement of a Func that is used by another Func: by default a Func without update definitions is inlined into its consumers, and a Func with update definitions is computed inside the innermost loop of its consumer. Loop directives on an inlined Func are an error. `f.compute_root()` computes it once over the whole required region; `f.compute_at(g, v)` computes it inside the loop over `v` of `g`; `f.store_at(g, v)` and `f.store_root()` set where its storage is allocated; `f.compute_inline()` restores inlining.
- Further calls: `f.bound(v, min, extent)` declares the range of a coordinate; `f.reorder_storage(y, x)` changes the storage order of the dimensions; `f.in_(g)` returns a wrapper Func for the uses of `f` in `g`; `f.update(k).rfactor([(r.x, v)])` factors an associative reduction over `r.x` into an intermediate Func with the additional Var `v`, which it returns.
