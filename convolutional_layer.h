//========================================================================
// Convolutional layer header file
//========================================================================
// @brief: function prototype definition

#ifndef SRC_CONVOLUTIONAL_LAYER_H_
#define SRC_CONVOLUTIONAL_LAYER_H_

#include <stdio.h>
#include <stdlib.h>
#include "sds_lib.h"

#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"
#include "utilities.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include "inits.h"
#include "timer.h"

// redefine struct layer
typedef layer convolutional_layer;

// build and configure convolutional layer
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride,\
		int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
// calculate workspace size for memory allocation
size_t get_workspace_size(layer l);
// convolutional layer top function
void forward_convolutional_layer(const convolutional_layer layer, network_state state);
// calculate output height
int convolutional_out_height(convolutional_layer l);
// calculate output_weight
int convolutional_out_width(convolutional_layer l);
// add bias to output values
void add_bias(float *output, float *biases, int batch, int n, int size);
// scale bias
void scale_bias(float *output, float *scales, int batch, int n, int size);
// swap values
void swap_binary(convolutional_layer *l);

#endif /* SRC_CONVOLUTIONAL_LAYER_H_ */
