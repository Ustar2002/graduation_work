# Task: complex fast Fourier transform (single precision)

## Definition

Implement a function named `fft` that computes the forward discrete Fourier transform of a sequence of n complex numbers x_0, ..., x_(n-1) and stores the n complex results X_0, ..., X_(n-1) in the output buffer `out`. For every k from 0 to n-1:

    X_k = sum over j from 0 to n-1 of x_j * exp(-2 * pi * i * j * k / n)

where i is the imaginary unit and exp(-2 * pi * i * j * k / n) = cos(2 * pi * j * k / n) - i * sin(2 * pi * j * k / n). The exponent carries a negative sign (forward transform). No scaling is applied: the result is not divided by n, sqrt(n) or anything else. Written out in real arithmetic, with x_j = a_j + i * b_j and theta = 2 * pi * j * k / n:

    Re(X_k) = sum over j of ( a_j * cos(theta) + b_j * sin(theta) )
    Im(X_k) = sum over j of ( b_j * cos(theta) - a_j * sin(theta) )

In particular, X_0 is the plain sum of all input elements.

n is always a power of two (n = 2^m with m >= 1), which allows the transform to be computed in O(n log n) operations.

## Data layout

- `x` holds the n complex input elements as 2 * n IEEE-754 single-precision (32-bit) floating-point numbers (float32), contiguous and interleaved: the real part of element j is at linear index 2*j and its imaginary part at linear index 2*j + 1 (real and imaginary part of element 0, then of element 1, and so on).
- `out` holds the n complex output elements as 2 * n contiguous float32 values in the same interleaved layout: the real part of X_k is at linear index 2*k and its imaginary part at linear index 2*k + 1.
- Do not read or write outside these ranges.
- The start addresses of `x` and `out` are 64-byte aligned. The two buffers do not overlap.
- n is a power of two, n >= 2, and always agrees with the actual sizes of the buffers (each holds 2 * n float32 values).

## Contract

- The function has exactly three parameters, in this order: x, out, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- out is allocated by the caller. Its initial contents are unspecified (arbitrary finite values) and must not be used: the function must overwrite every element of out, both the real and the imaginary part of every X_k, with the correct result. It must not add to, or otherwise combine with, whatever out contained before the call.
- The result must be written into out. The return value of the function is ignored: results are read only from the buffer out, so returning a new array instead of filling out counts as not writing out.
- `x` is read-only. The function must not modify it, not even temporarily (an in-place transform must therefore work on a copy or directly in out).
- Temporary memory (working buffers, twiddle-factor tables, index tables) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls (tables must be recomputed or rebuilt on every call).
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce results that agree within the tolerance below.
- The function must be correct for every power-of-two n >= 2, including very small n (2, 4, 8).
- The algorithm is a free choice: the order in which the terms are combined, the radix, the use and order of bit-reversal or other permutations, how the twiddle factors (the complex exponentials) are computed and stored, and how complex numbers are represented internally are all left to the implementation. The result must be the exact transform defined above, evaluated to within the tolerance below: approximations that drop terms are not permitted.
- Input values are finite float32 numbers. In the evaluation both the real and the imaginary parts are drawn uniformly from the interval [-1, 1), so the output components have a root mean square of about sqrt(n / 3) and magnitudes up to a few times that. No special handling of NaN, infinity or subnormal values is required, but the output must not contain NaN or infinity.

## Accuracy

The output is compared element by element (every real part and every imaginary part) against a reference computed from the same inputs in 64-bit (double-precision) arithmetic. The check passes if, for every element,

    |out[e] - ref[e]| <= atol + rtol * |ref[e]|,    with rtol = 1e-3 and atol = 1e-3 * RMS(ref),

where RMS(ref) is the root mean square over all real and imaginary parts of the reference output of the shape being checked (so atol scales with the size of the problem). Any NaN or infinity in `out` fails the check. Within this tolerance, the precision used for intermediate values and twiddle factors and the order of operations are free choices. A transform with the opposite sign in the exponent, or a scaled transform, does not satisfy this check.

## Evaluation

- Correctness shapes (n): 8, 1,024 and 65,536.
- Performance shape (n): 4,194,304 (2^22). The output for the performance shape must also pass the accuracy check.
- Memory: the process that calls the function is limited to 8 GiB of address space, inputs and outputs included. An allocation beyond that limit fails the call and counts as a failure.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with out pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of the input are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The input is already in memory when the function is called; the measured time is the duration of the call itself, including any table setup the call performs.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
