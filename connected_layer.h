//========================================================================
// Connected layer header file
//========================================================================
// @brief: function prototype definition

#ifndef SRC_CONNECTED_LAYER_H_
#define SRC_CONNECTED_LAYER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "batchnorm_layer.h"
#include "utilities.h"
#include "blas.h"
#include "gemm.h"
#include "activations.h"
#include "layer.h"
#include "network.h"
#include "inits.h"

// redefine layer
typedef layer connected_layer;

connected_layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize);
void forward_connected_layer(connected_layer layer, network_state state);
void backward_connected_layer(connected_layer layer, network_state state);
void update_connected_layer(connected_layer l, int batch, float learning_rate, float momentum, float decay);
void denormalize_connected_layer(layer l);
void statistics_connected_layer(layer l);

#endif /* SRC_CONNECTED_LAYER_H_ */
