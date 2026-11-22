# Task: discrete Fourier transform of a complex sequence (forward, length a power of two)

## Definition

Implement a function named `fft` that computes the forward discrete Fourier transform of a sequence of n complex numbers z[0], z[1], ..., z[n-1], where n is a power of two. The result is the sequence of n complex numbers

    X[k] = sum over j from 0 to n-1 of z[j] * exp(-2*pi*i*j*k/n),    for 0 <= k < n

where i is the imaginary unit and exp(-i*t) = cos(t) - i*sin(t). The exponent is negative, and the sum is not normalised: it is not divided by n or by sqrt(n). Written with real numbers, for z[j] = a[j] + i*b[j] and t(j, k) = 2*pi*j*k/n:

    real part of X[k] = sum over j from 0 to n-1 of ( a[j]*cos(t(j, k)) + b[j]*sin(t(j, k)) )

    imaginary part of X[k] = sum over j from 0 to n-1 of ( b[j]*cos(t(j, k)) - a[j]*sin(t(j, k)) )

X[0] is therefore the sum of all input elements. The input z is given in the array `x`, and the output array `out` must contain X[0], X[1], ..., X[n-1] in this order.

## Data layout

- A complex number is stored as two consecutive IEEE-754 single-precision (32-bit) floating-point numbers (float32): its real part followed by its imaginary part. The input element z[j] is the pair x[2*j] (real part) and x[2*j+1] (imaginary part). The output element X[k] is the pair out[2*k] (real part) and out[2*k+1] (imaginary part).
- The real and imaginary parts are interleaved in this way in both arrays: an array is not split into a block of all real parts followed by a block of all imaginary parts.
- `x` and `out` each contain exactly 2*n float32 values and are contiguous. Do not read or write outside these ranges.
- The start addresses of `x` and `out` are 64-byte aligned. The two buffers do not overlap.
- n is the number of complex elements, not the number of float32 values. It is a power of two with n >= 2 (n = 2, 4, 8, 16, ...) and always agrees with the actual sizes of the buffers.

## Contract

- The function has exactly three parameters, in this order: x, out, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- out is allocated by the caller. Its initial contents are unspecified (arbitrary finite values) and must not be used: the function must overwrite every element of out with the correct result. It must not add to, or otherwise combine with, whatever out contained before the call.
- The result must be written into out. The return value of the function is ignored: results are read only from the buffer out, so returning a new array instead of filling out counts as not writing out.
- `x` is read-only. The function must not modify it, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce results that agree within the tolerance below.
- The function must be correct for every power of two n >= 2, including the smallest sizes n = 2 and n = 4, and for n = 2^m with m odd as well as with m even.
- Input values are finite float32 numbers. In the evaluation every real part and every imaginary part of the input is drawn uniformly from the interval [-1, 1) and then multiplied by 1/sqrt(n). With this scaling the output has the same typical magnitude for every n: the real and imaginary parts of the output elements have a standard deviation of about 0.58, and |X[k]| has a root mean square of about 0.82. No special handling of NaN, infinity or subnormal values is required, but the output must not contain NaN or infinity.

## Accuracy

The output is compared element by element against a reference computed from the same input with 64-bit (double-precision) arithmetic. The real part and the imaginary part of every output element are checked separately, as the 2*n float32 values of `out`. The check passes if, for every m with 0 <= m < 2*n,

    |out[m] - ref[m]| <= atol + rtol * |ref[m]|,    with rtol = 1e-3 and atol = 1e-3.

Any NaN or infinity in `out` fails the check. Within this tolerance, the order of the arithmetic operations and the precision used for intermediate values are free choices.

## Evaluation

- Correctness shapes (n): 8, 1,024 and 65,536.
- Performance shape (n): 4,194,304. The output for the performance shape must also pass the accuracy check.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with out pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
