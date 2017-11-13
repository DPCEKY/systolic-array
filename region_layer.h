//========================================================================
// Region layer header file
//========================================================================
// @brief: function prototype & activate type definition

#ifndef SRC_REGION_LAYER_H_
#define SRC_REGION_LAYER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "layer.h"
#include "network.h"
#include "box.h"
#include "utilities.h"
#include "blas.h"
#include "activations.h"
#include "region_layer.h"
#include "inits.h"

// make region layer
layer make_region_layer(int batch, int h, int w, int n, int classes, int coords);
// get bounding boxes
void get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map, float tree_thresh);
// get bounding box (single)
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h);
// region layer top function
void forward_region_layer(const layer l, network_state state);
// extra region classes
void delta_region_class(float *output, float *delta, int index, int class_s, int classes, tree *hier, float scale, float *avg_cat);
// extra region boxes
float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale);

#endif /* SRC_REGION_LAYER_H_ */
