//========================================================================
// Option list
//========================================================================
// @brief: read and compare parameters

#include "option_list.h"

// read cfg data
list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == 0)
    {
        file_error(filename);
    }
    char *line;
    list *options = make_list();
    int nu = 0;
    // get each line
    while((line=fgetl(file)) != 0)
    {
        nu++;
        //printf("nu:%d; line: %s;\n",nu,line);
        strip(line);
        switch(line[0])
        {
            case '\0':
            case '#':
            case ';':
            {
                free(line);
                break;
            }
            default:
            {
                if (!read_option(line, options))
                {
                    fprintf(stderr,"Config file error line %d, could parse: %s\n",nu,line);
                    free(line);
                }
                break;
            }
        }
    }
    fclose(file);
    return options;
}

// change "=" to "\n", and insert it into a list option
// val stores the address of string of value
int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val=0;
    // split the string s within "="
    for (i = 0; i < len; i++)
    {
        if (s[i] == '=')
        {
            s[i] = '\0';
            val  = s+i+1;
            break;
        }
    }
    //
    if(i == len-1)
    {   // no value for this key, insert failed: return 0
        return 0;
    }
    char *key = s;
    option_insert(options, key, val);
    // successfully insert key&value into option: return 1
    return 1;
}

// insert value(*val) into list option
void option_insert(list *l, char *key, char *val)
{
    kvp *p = (kvp *)malloc(sizeof(kvp));
    p->key  = key;
    p->val  = val;
    p->used = 0;
    list_insert(l, p);
}

// check specific strings (keys)
char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if (v)
    {   //
        return v;
    }
    if (def)
    {   // use default cfg
        fprintf(stderr, "%s: Using default '%s' \n", key, def);
    }
    return def;
}

//traverse the list l
char *option_find (list *l, char *key)
{
    node *n = l->front;
    // traverse the list from the first node l->front
    while(n)
    {
        kvp *p = (kvp *)n->val;
        if (strcmp(p->key, key) == 0)
        {
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    // no match key found, return 0
    return 0;
}

// ???
void option_unused (list *l)
{
    node *n = l->front;
    // traverse the list from the first node l->front
    while(n)
    {
        kvp *p = (kvp *)n->val;
        if(!p->used)
        {
            fprintf(stderr, "Unused field: '%s' = '%s'\n", p->key, p->val);
        }
        n = n->next;
    }
}

// find specific ints
int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if (v)
    {
        return atoi(v);
    }
    if (def)
    {
        fprintf(stderr, "%s: Using default '%d'\n", key, def);
    }
    return def;
}

// find specific ints
int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if (v)
    {
        return atoi(v);
    }
    return def;
}

// find specific floats
float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if (v)
    {
        return atof(v);
    }
    fprintf(stderr, "%s: Using default: '%lf'\n", key, def);
    return def;
}

// find specific floats
float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if (v)
    {
        return atof(v);
    }
    return def;
}
