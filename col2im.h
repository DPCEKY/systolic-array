//========================================================================
// Col2im header file
//========================================================================
// @brief: function prototype definition

#ifndef SRC_COL2IM_H_
#define SRC_COL2IM_H_

#include <stdio.h>
#include <stdlib.h>

// column to image: filters%batch == 0
void col2img(float *c_col,float *c, int m, int n, int count, int batch);
// column to image: filters%batch != 0
void col2img_extra(float *c_col,float *c, int m, int n, int count, int batch);

#endif /* SRC_COL2IM_H_ */
