//========================================================================
// Blas header file
//========================================================================
// @brief: function prototype definition

#ifndef SRC_BLAS_H_
#define SRC_BLAS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <math.h>

// multiply some values in *X with ALPHA
void scal_cpu(int N, float ALPHA, float *X, int INCX);
// assign some values in *X with ALPHA
void fill_cpu(int N, float ALPHA, float *X, int INCX);
// mean calculation
void mean_cpu(float *x, int batch, int filters, int spatical, float *mean);
// variance calculation
void variance_cpu(float *x, float *mean, int batch, int filters, int spatical,float *variance);
// multiply some values in *X with ALPHA
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
// array copy
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
// normalization with mean and variance
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
// scale an array
void scale_cpu(int N, float ALPHA, float *X, int INCX);
// flatten layer
void flatten(float *x, int size, int layers, int batch, int forward);
// softmax layer
void softmax(float *input, int n, float temp, float *output);

#endif /* SRC_BLAS_H_ */
