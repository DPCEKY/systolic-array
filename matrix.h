//========================================================================
// Matrix header file
//========================================================================
// @brief: struct type definition

#ifndef SRC_MATRIX_H_
#define SRC_MATRIX_H_

// row, column, and data
typedef struct matrix
{
    int row;
    int cols;
    float **vals;
} matrix;

#endif /* SRC_MATRIX_H_ */
