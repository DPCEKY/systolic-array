//========================================================================
// Utilities header file
//========================================================================
// @brief: function prototype definition

#ifndef SRC_UTILITIES_H_
#define SRC_UTILITIES_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <float.h>
#include <limits.h>

#include "list.h"

#define SECRET_NUM -1234
#define TWO_PI 6.2831853071795864769252866

// read files
int *read_map(char *filename);
// report file open error
void file_error(char *s);
// remove space, tab, and enter in a string
void strip(char *s);
// get one line from file
char *fgetl(FILE *fp);
// report malloc error
void malloc_error();
// report specific error
void error(const char *s);
// free 2D array (array of pointer)
void free_ptrs(void **ptrs, int n);
// find the maximum value in an array, return its index
int max_index(float *a, int n);
// return a random number in the given range(min, max)
float rand_uniform(float min, float max);
// print function
void print_statistics(float *a, int n);
// mean value of array a
float mean_array(float *a, int n);
// sum of array a
float sum_array(float *a, int n);
// variance of array a
float variance_array(float *a, int n);
// mean squared error of array a
float mse_array(float *a, int n);
// difference of two squares
float mag_array(float *a, int n);

#endif /* SRC_UTILITIES_H_ */
