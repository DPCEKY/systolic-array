//===========================================================================
// typedefs.h
//===========================================================================
// @brief: define bitwise variables & macros

#ifndef LOADATA_H
#define LOADATA_H

#include <ap_int.h>
#include <ap_fixed.h>

#define TOT_WIDTH_IN 32
#define INT_WIDTH_IN 8
#define TOT_WIDTH_OUT 64
#define INT_WIDTH_OUT 16

typedef ap_fixed<TOT_WIDTH_IN, INT_WIDTH_IN> INPUT_32;
typedef ap_fixed<TOT_WIDTH_OUT, INT_WIDTH_OUT> OUTPUT_64;

typedef ap_int<16> bit16;
typedef ap_int<32> bit32;

#endif
