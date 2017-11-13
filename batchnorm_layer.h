//========================================================================
// Batchnorm header file
//========================================================================
// @brief: function prototype definition

#ifndef SRC_BATCHNORM_LAYER_H_
#define SRC_BATCHNORM_LAYER_H_

#include <stdio.h>
#include <stdlib.h>

#include "blas.h"
#include "layer.h"
#include "image.h"
#include "network.h"
#include "convolutional_layer.h"

// batchnorm layer top function
void forward_batchnorm_layer(layer l, network_state state);
// scale calculation for backward propagation
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
// mean calculation
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
// variance calculation
void variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
// barchnorm with delta
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

#endif /* SRC_BATCHNORM_LAYER_H_ */
