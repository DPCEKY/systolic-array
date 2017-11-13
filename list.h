//========================================================================
// List header file
//========================================================================
// @brief: function prototype & activate type definition

#ifndef SRC_LIST_H_
#define SRC_LIST_H_

#include <stdlib.h>
#include <stdio.h>

// linked list
typedef struct node
{
	void *val;
	struct node *next;
	struct node *prev;
} node;

typedef struct list
{
	int size;
	node *front;
	node *back;
} list;

// make a new list
list *make_list();
// insert a node to l->back
void list_insert(list *l, void *val);
// convert the list to a 2D array (array of pointer)
void **list_to_array(list *l);
// free the space allocated for the list
void free_list(list *l);
// free the node
void free_node(node *n);

#endif /* SRC_LIST_H_ */
