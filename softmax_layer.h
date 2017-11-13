//========================================================================
// Softmax layer header file
//========================================================================
// @brief: function prototype definition

#ifndef SRC_SOFTMAX_LAYER_H_
#define SRC_SOFTMAX_LAYER_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <float.h>

#include "blas.h"
#include "layer.h"
#include "network.h"
#include "inits.h"

// redefine layer
typedef layer softmax_layer;

// update softmax tree
void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output);
// make softmax layer
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
// softmx layer top function
void forward_softmax_layer(const softmax_layer l, network_state state);
// backward softmax function
void backward_softmax_layer(const softmax_layer l, network_state state);


#endif /* SRC_SOFTMAX_LAYER_H_ */
