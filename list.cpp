//========================================================================
// List
//========================================================================
// @brief: linked list for reading parameters

#include "list.h"

// make an empty list, return a pointer
list *make_list()
{
    list *l  = (list *)malloc(sizeof(list));
    l->size  = 0;
    l->front = 0;
    l->back  = 0;

    return l;
}

// insert a new node into list *l with "value":*val
void list_insert(list *l, void *val)
{
    node *new_node = (node *)malloc(sizeof(node));
    new_node->val  = val;
    new_node->next = 0;
    // add new node to l->back
    if(!l->back)
    {   // empty list
        l->front = new_node;
        new_node->prev = 0;
    }
    else
    {
        l->back->next = new_node;
        new_node->prev = l->back;
    }
    l->back = new_node;
    l->size++; ////
}

// convert a list to 2D array (***array of pointer***)
void **list_to_array(list *l)
{
    void **res = (void **)calloc(l->size, sizeof(void *));
    int counter = 0;
    node *n = l->front; // first node in list l
    // convert the list
    while(n)
    {
        res[counter++] = n->val; //
        n = n->next;
    }
    return res;
}

// free memory allocated for the list
void free_list(list *l)
{
    free_node(l->front); // first node
    free(l);
}

// free node
void free_node(node *n)
{
    node *next;
    // free all nodes
    while(n)
    {
        next = n->next;
        free(n);
        n = next;
    }
}
