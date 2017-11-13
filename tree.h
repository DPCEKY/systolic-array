//========================================================================
// Tree header file
//========================================================================
// @brief: function prototype definition

#ifndef SRC_TREE_H_
#define SRC_TREE_H_

#include <stdio.h>
#include <stdlib.h>

#include "typedef_tree.h"
#include "utilities.h"
#include "data.h"

// update prediction tree
int hierarchy_top_prediction(float *predictions, tree *hier, float thresh);
// build tree hierarchy
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves);
// get probabilities
float get_hierarchy_probability(float *x, tree *hier, int c);
// read values
tree *read_tree(char *filaname);

#endif /* SRC_TREE_H_ */
