#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float* values, float* output, int N) {
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH) {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

#define MAX_VALUE 9.999999f

void clampedExpVector(float* values, int* exponents, float* output, int N) {
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  __pp_vec_float bases, result;
  __pp_vec_float maxValue = _pp_vset_float(MAX_VALUE);
  __pp_vec_int exps, currentExps;
  __pp_vec_int ones = _pp_vset_int(1);
  __pp_mask maskIsValidPosition, maskNeedMultiply, maskNeedClamp;

  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    // create a mask of valid positions
    size_t numValidPositions = min(N - i, VECTOR_WIDTH);
    maskIsValidPosition = _pp_init_ones(numValidPositions);

    // load valid data
    _pp_vload_float(bases, values + i, maskIsValidPosition);
    _pp_vload_int(exps, exponents + i, maskIsValidPosition);
    _pp_vset_float(result, 1.0f, maskIsValidPosition);

    // set currentExp to 0
    _pp_vset_int(currentExps, 0, maskIsValidPosition);

    // needMultiply = isValidPosition && (currentExp < exp);
    maskNeedMultiply = _pp_init_ones(0);
    _pp_vlt_int(maskNeedMultiply, currentExps, exps, maskIsValidPosition);

    while (_pp_cntbits(maskNeedMultiply) > 0) {
      // result *= base;
      _pp_vmult_float(result, result, bases, maskNeedMultiply);

      // currentExp++;
      _pp_vadd_int(currentExps, currentExps, ones, maskNeedMultiply);

      // recalculate needMultiply
      maskNeedMultiply = _pp_init_ones(0);
      _pp_vlt_int(maskNeedMultiply, currentExps, exps, maskIsValidPosition);
    }

    // clamp at MAX_VALUE
    maskNeedClamp = _pp_init_ones(0);
    _pp_vgt_float(maskNeedClamp, result, maxValue, maskIsValidPosition);
    _pp_vset_float(result, MAX_VALUE, maskNeedClamp);

    // store back computed results
    _pp_vstore_float(output + i, result, maskIsValidPosition);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float* values, int N) {

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH) {
  }

  return 0.0;
}