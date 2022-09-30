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

// needMultiply = isValidPosition && (currentExp < exp);
inline void updateNeedMultiplyMask(
  __pp_mask& maskNeedMultiply,
  __pp_vec_int& currentExponents,
  __pp_vec_int& exponents,
  __pp_mask& maskIsValidPosition
) {
  maskNeedMultiply = _pp_init_ones(0);
  _pp_vlt_int(maskNeedMultiply, currentExponents, exponents, maskIsValidPosition);

}

// perform element-wise power
void powerVector(
  __pp_vec_float& result,
  __pp_vec_float& bases,
  __pp_vec_int& exponents,
  __pp_mask& maskIsValidPosition
) {
  static __pp_vec_int ones = _pp_vset_int(1);
  __pp_vec_int currentExponents;
  __pp_mask maskNeedMultiply;

  // initialize result to 1
  _pp_vset_float(result, 1.0f, maskIsValidPosition);

  // set currentExponent to 0
  _pp_vset_int(currentExponents, 0, maskIsValidPosition);

  updateNeedMultiplyMask(maskNeedMultiply, currentExponents, exponents, maskIsValidPosition);

  while (_pp_cntbits(maskNeedMultiply) > 0) {
    // result *= base;
    _pp_vmult_float(result, result, bases, maskNeedMultiply);

    // currentExp++;
    _pp_vadd_int(currentExponents, currentExponents, ones, maskNeedMultiply);

    updateNeedMultiplyMask(maskNeedMultiply, currentExponents, exponents, maskIsValidPosition);
  }
}

// clamp the element in result on maxValue
void clampVector(
  __pp_vec_float& result,
  float maxValue,
  __pp_mask& maskIsValidPosition
) {
  static __pp_vec_float vecMaxValue = _pp_vset_float(maxValue);
  __pp_mask maskNeedClamp = _pp_init_ones(0);

  _pp_vgt_float(maskNeedClamp, result, vecMaxValue, maskIsValidPosition);
  _pp_vset_float(result, maxValue, maskNeedClamp);
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
  __pp_vec_int exps;
  __pp_mask maskIsValidPosition;

  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    // create a mask of valid positions
    size_t numValidPositions = min(N - i, VECTOR_WIDTH);
    maskIsValidPosition = _pp_init_ones(numValidPositions);

    // load valid data
    _pp_vload_float(bases, values + i, maskIsValidPosition);
    _pp_vload_int(exps, exponents + i, maskIsValidPosition);

    powerVector(result, bases, exps, maskIsValidPosition);
    clampVector(result, MAX_VALUE, maskIsValidPosition);

    // store back computed results
    _pp_vstore_float(output + i, result, maskIsValidPosition);
  }
}

float sumUpNumbersInVector(__pp_vec_float& vec) {
  static __pp_mask maskAll = _pp_init_ones();
  size_t numValidElements = VECTOR_WIDTH;
  __pp_vec_float result, store;

  _pp_vmove_float(result, vec, maskAll);

  while (numValidElements > 1) {
    // store[i,i+1] = result[i] + result[i+1];
    _pp_hadd_float(store, result);

    // result[:numValidElements/2] = store[0:numValidElements:2];
    // result[numValidElements/2:] = store[1:numValidElements:2];
    _pp_interleave_float(result, store);

    // only the first half is valid now
    numValidElements /= 2;
  }

  return result.value[0];
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float* values, int N) {

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  static __pp_mask maskAll = _pp_init_ones();

  float sum = 0.0;
  __pp_vec_float vec;

  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    _pp_vload_float(vec, values + i, maskAll);
    sum += sumUpNumbersInVector(vec);
  }

  return sum;
}