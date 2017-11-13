//========================================================================
// Option list header file
//========================================================================
// @brief: function prototype & type definition

#ifndef SRC_OPTION_LIST_H_
#define SRC_OPTION_LIST_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utilities.h"
#include "list.h"

// key+value+number
typedef struct
{
	char *key;
	char *val;
	int used;
} kvp;

// function prototype
// read cfg data, and build a list
list *read_data_cfg(char *filename);
// change "=" to "\n", and insert it into a list option
// val stores the address of string of value
int read_option(char *s, list *options);
// insert value(*val) into list option
void option_insert(list *l, char *key, char *val);
// find specific key in list l
char *option_find(list *l, char *key);
// find specific strings
char *option_find_str(list *l, char *key, char *def);
// find specific ints
int option_find_int_quiet(list *l, char *key, int def);
int option_find_int(list *l, char *key, int def);
// find specific flaots
float option_find_float(list *l, char *key, float def);
float option_find_float_quiet(list *l, char *key, float def);
// find unused items
void option_unused(list *l);

#endif /* SRC_OPTION_LIST_H_ */
