//========================================================================
// Network header file
//========================================================================
// @brief: function prototype & type definition

#ifndef SRC_NETWORK_H_
#define SRC_NETWORK_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "image.h"
#include "data.h"
#include "utilities.h"
#include "blas.h"
#include "tree.h"

#include "layer.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "region_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "softmax_layer.h"

// learning rate policy
typedef enum
{
	CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

// struct network
typedef struct network
{
	float *workspace;       
	int n;
	int batch;
	int *seen;
	float epoch;
	int subdivisions;
	float momentum;
	float decay;
	layer *layers;	        // layers 
	int outputs;			// output sizes
	float *output; 			// output values 
	learning_rate_policy policy;
	//
	float learning_rate;
	float gamma;
	float scale;
	float power;
	int time_steps;
	int step;
	int max_batches;
	float *scales;
	int *steps;
	int num_steps;
	int burn_in;
	//
	int adam;
	float B1;
	float B2;
	float eps;
	//
	int inputs;
	int h;
	int w;
	int c;
	int max_crop;
	int min_crop;
	float angle;
	float aspect;
	float exposure;
	float saturation;
	float hue;
	//
	int gpu_index;
	tree *hierarchy;
} network;

// network state
typedef struct network_state
{
	float *truth;
	float *input;
	float *delta;
	float *workspace;
	int train;
	int index;
	network net;
} network_state;

// make a new && empty network with n layers
network make_network(int n);
// calculate the size of network
int get_network_output_size(network net);
// get network netowrk
float *get_network_output(network net);
// set batch mode
void set_batch_network(network *net, int b);
// top prediction function
float *network_predict(network net, float *input);
// go through all network layers ************************
void forward_network(network net, network_state state);

#endif /* SRC_NETWORK_H_ */
