# Homework 1: SIMD Programming

[**Report**](https://hackmd.io/@alanlin/pp-f22-hw1)

## Part 1: Vectorizing Code Using Fake SIMD Intrinsics

Utilize the mock SIMD library `PPintrin` provided by TAs to implement vector arithmetic operations as follows:
- [x] `clampExpVector`
    - calculate element-wise exponent, with clamping on a specific value.
- [ ] `arraySumVector`
    - sum up the vector in `O(N / VECTOR_WIDTH + log2(VECTOR_WIDTH))` span.

## Part 2: Vectorizing Code with Automatic Vectorization Optimizations

