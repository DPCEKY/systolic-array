//========================================================================
// Gemm header file
//========================================================================
// @brief: function prototype & macro definition

#ifndef SRC_GEMM_H_
#define SRC_GEMM_H_

#include <stdio.h>
#include <stdlib.h>

#include "typedefs.h"

// data size
#define SIZE_BATCH 16
#define MAX_A 3*3*1024*SIZE_BATCH
//#define MAX_A 1024*425
#define MAX_B 210*210*16
#define MAX_C 416*416*16
// finter size
#define SIZE_FILTER 3*3
#define SIZE_FILTER_EXTRA 1*1
#define MAX_FILTER_DEPTH 1024
// line buffer size
#define NUM_LINE_BUFFER 3
#define SIZE_LINE_BUFFER 15*1024
#define NUM_LINE_BUFFER_EXTRA 1
// window buffer size
#define NUM_WINDOW_BUFFER 3*3
#define SIZE_WINDOW_BUFFER 1024
#define NUM_WINDOW_BUFFER_EXTRA 1
// systolic kernel size
#define SystolicKernelSize 13 //greatest number the zc706 FPGA can hold: 13
// data access pattern
#pragma SDS data mem_attribute(A:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(B:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(C:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(A:SEQUENTIAL, B:SEQUENTIAL, C:SEQUENTIAL)
#pragma SDS data copy(A[0:size_filter*SIZE_BATCH], B[0:(width+2*pad)*(height+2*pad)*channels], C[0:size_channel*SIZE_BATCH])
// gemm with filter size 3x3
void gemm2(float A[MAX_A], float B[MAX_B], float C[MAX_C],int num_filter, int size_channel,int size_filter,\
		  int channels, int height, int width, int ksize, int pad);
// extra gemm with filter size 1x1
void gemm_extra2(float A[MAX_A], float B[MAX_B], float C[MAX_C],int size_channel,int size_filter,int ksize,
		        INPUT_32 weights[SIZE_BATCH][SIZE_FILTER][MAX_FILTER_DEPTH],OUTPUT_64 output[SIZE_BATCH]);

#endif /* SRC_GEMM_H_ */
