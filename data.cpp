//========================================================================
// Data 
//========================================================================
// @brief: loading data function

#include "data.h"

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// get 80 labels(classes) and store them into 2D array
char **get_labels(char *filename)
{
    // get label list
    list *plist = get_paths(filename);

    /*
    // verify plist
    node * pnode = plist->front;
    int counter = 0;
    printf("name_list size: %d;\n",plist->size);
    while(pnode->next)
    {
        pnode = pnode->next;
        printf("name_list NO. %d: %s; \n",counter, (char *)pnode->val);
        counter++;
    }
    */


    char **labels = (char **)list_to_array(plist); //???
    free_list(plist);
    return labels;
}

// read each line from a file, return a list
list *get_paths(char *filename)
{
    char *line;
    FILE *file = fopen(filename, "r");
    if(!file)
    {
        file_error(filename);
    }
    // make a new list
    list *lines = make_list();
    // store every line (classes) into the list
    while((line=fgetl(file)))
    {
        list_insert(lines, line);
    }
    fclose(file);
    return lines;
}

