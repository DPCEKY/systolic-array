//========================================================================
// Tree
//========================================================================
// @brief: update tree of probabilities

#include "tree.h"

// update prediction tree
int hierarchy_top_prediction(float *predictions, tree *hier, float thresh)
{
    float p = 1;
    int group = 0;
    while(1)
    {
        float max = 0;
        int max_i = 0;

        for(int i = 0; i < hier->group_size[group]; i++)
        {
            int index = i + hier->group_offset[group];
            float val = predictions[i + hier->group_offset[group]];
            if(val > max)
            {
                max_i = index;
                max = val;
            }
        }
        if(p*max > thresh)
        {
            p = p*max;
            group = hier->child[max_i];
            if(hier->child[max_i] < 0)
            {
                return max_i;
            }
        }
        else
        {
            return hier->parent[hier->group_offset[group]];
        }
    }
    return 0;
}

// build tree hierarchy
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves)
{
    for (int j = 0; j < n; j++)
    {
        int parent = hier->parent[j];
        if(parent >= 0)
        {
            predictions[j] *= predictions[parent];
        }
    }
    if (only_leaves)
    {
        for (int j = 0; j < n; j++)
        {
            if (!hier->leaf[j])
            {
                predictions[j] = 0;
            }
        }
    }
}

// get probabilities
float get_hierarchy_probability(float *x, tree *hier, int c)
{
    float p = 1;
    while (c >= 0)
    {
        p *= x[c];
        c = hier->parent[c];
    }
    return p;
}

// read values
tree *read_tree(char *filename)
{
    tree t = {0};
    FILE *file = fopen(filename, "r");
    char *line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    //
    while ((line = fgetl(file)) != 0)
    {
        char *id = (char *)calloc(256, sizeof(char));
        int parent = -1;
        // read dat afrom string line
        sscanf(line, "%s %d", id, &parent);
        t.parent = (int *)realloc(t.parent, (n+1)*sizeof(int));
        t.parent[n] = parent;

        t.child = (int *)realloc(t.child, (n+1)*sizeof(int));
        t.child[n] = -1;

        t.name = (char **)realloc(t.name, (n+1)*sizeof(char *));
        t.name[n] = id;

        if(parent != last_parent)
        {
            groups++;
            t.group_offset = (int *)realloc(t.group_offset, groups * sizeof(int));
            t.group_offset[groups - 1] = n - group_size;
            t.group_size = (int *)realloc(t.group_size, groups * sizeof(int));
            t.group_size[groups - 1] = group_size;
            group_size = 0;
            last_parent = parent;
        }
        t.group = (int *)realloc(t.group, (n+1)*sizeof(int));
        t.group[n] = groups;
        if (parent >= 0)
        {
            t.child[parent] = groups;
        }
        n++;
        group_size++;
    }
    groups++;
    //
    t.group_offset = (int *)realloc (t.group_offset, groups * sizeof(int));
    t.group_offset[groups - 1] = n - group_size;
    t.group_size = (int *)realloc(t.group_size, groups * sizeof(int));
    t.group_size[groups - 1] = group_size;
    t.n = n;
    t.groups = groups;
    t. leaf = (int *)calloc(n, sizeof(int));
    //
    for (int i = 0; i < n; i++)
    {
        t.leaf[i] = 1;
    }
    for (int i = 0; i < n; i++)
    {
        if (t.parent[i] >= 0)
        {
            t.leaf[t.parent[i]] = 0;
        }
    }
    //
    fclose(file);
    tree *tree_ptr = (tree *)calloc(1, sizeof(tree));
    *tree_ptr = t;
    return tree_ptr;
}
