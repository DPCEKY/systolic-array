//========================================================================
// Maxpooling header file
//========================================================================
// @brief: function prototype definition

#ifndef SRC_MAXPOOL_LAYER_H_
#define SRC_MAXPOOL_LAYER_H_

#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "layer.h"
#include "network.h"
#include "inits.h"

// redefine layer
typedef layer maxpool_layer;

// make maxpooling layer
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
// maxpooling top function
void forward_maxpool_layer(const maxpool_layer l, network_state state);

#endif /* SRC_MAXPOOL_LAYER_H_ */
