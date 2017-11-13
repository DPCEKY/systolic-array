//========================================================================
// Parser header file
//========================================================================
// @brief: function prototype & type definition

#ifndef SRC_PARSER_H_
#define SRC_PARSER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "sds_lib.h"

#include "network.h"
#include "activations.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "list.h"
#include "maxpool_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "utilities.h"
#include "inits.h"

// parameters 
typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network net;
} size_params;

typedef struct section{
    char *type;
    list *options;
} section;

// parser differnet layers
maxpool_layer parse_maxpool(list *options, size_params params);
void transpose_matrix(float *a, int rows, int cols);
layer parse_region(list *options, size_params params);
convolutional_layer parse_convolutional(list *options, size_params params);
LAYER_TYPE string_to_layer_type(char * type);
learning_rate_policy get_policy(char *s);
void parse_net_options(list *options, network *net);
void free_section(section *s);
network parse_network_cfg(char *filename);
// read data from file
list *read_cfg(char *filename);
// load weights for different layers
void load_convolutional_weights(layer l, FILE *fp);
void load_batchnorm_weights(layer l, FILE *fp);
void load_connected_weights(layer l, FILE *fp, int transpose);
void load_weights_upto(network *net, char *filename, int cutoff);
// load weights top function
void load_weights(network *net, char *filename);

#endif /* SRC_PARSER_H_ */
