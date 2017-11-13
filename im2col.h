//========================================================================
// Im2col header file
//========================================================================
// @brief: function prototype & activate type definition

#ifndef SRC_IM2COL_H_
#define SRC_IM2COL_H_

#include <stdio.h>
#include <stdlib.h>

// image to column : filters%batch == 0
void im2col(float *data_im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col);
// image to column : filters%batch != 0
void im2col_extra(float *data_im,int channels, int height, int width, int ksize,  int stride, int pad, float* data_col);

#endif /* SRC_IM2COL_H_ */
