//========================================================================
// Data header file
//========================================================================
// @brief: function prototype & struct type defination

#ifndef SRC_DATA_H_
#define SRC_DATA_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "typedef_tree.h"
#include "tree.h"
#include "matrix.h"
#include "list.h"
#include "image.h"
#include "utilities.h"

// data struct
typedef struct data
{
	int w;
	int h;
	matrix X;
	matrix Y;
	int shallow;
	int *num_boxes;
	box **boxes;
} data;

// data type
typedef enum
{
	CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA,\
	IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA,\
	OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA
} data_type;

// loading struct
typedef struct load_args
{
	int threads;
	char **paths;
	char *path;
	int n;
	int m;
	char **labels;
	int h;
	int w;
	int out_w;
	int out_h;
	int nh;
	int nw;
	int num_boxes;
	int min;
	int max;
	int size;
	int classes;
	int background;
	int scale;
	float jitter;
	float angle;
	float aspect;
	float saturation;
	float exposure;
	float hue;
	data *d;
	image *im;
	image *resized;
	data_type type;
	tree *hierarchy;
} load_args;

// loading box labels
typedef struct box_label
{
	int id;
	float x;
	float y;
	float w;
	float h;
	float left;
	float right;
	float top;
	float bottom;
} box_label;

// get 80 labels(classes) and store them into 2D array
char **get_labels(char *filename);
// read each line from a file, return a list
list *get_paths(char *filename);


// static inline function
// the compiler simply copy codes when it is invoked
static inline float distance_from_edge (int x, int max)
{
	int dx = (max/2) - x;
	if (dx < 0)
	{
		dx = -dx;
	}
	dx = (max/2) + 1 -dx;
	dx *= 2;
	float dis = (float)dx/(float)max;
	if(dis > 1)
	{
		dis = 1;
	}
	return dis;
}

#endif /* SRC_DATA_H_ */
