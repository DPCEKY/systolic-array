//========================================================================
// Typedef_tree header file
//========================================================================
// @brief: struct type definition

#ifndef SRC_TYPEDEF_TREE_H_
#define SRC_TYPEDEF_TREE_H_

// tree structure
typedef struct tree
{
	int *leaf;
	int n;
	int *parent;
	int *child;
	int *group;
	char **name;

	int groups;
	int *group_size;
	int *group_offset;
} tree;

#endif /* SRC_TYPEDEF_TREE_H_ */
