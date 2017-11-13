//========================================================================
// Softmax layer
//========================================================================
// @brief: softmax layer

#include "softmax_layer.h"

// update softmax tree
void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output)
{
    //
    for(int b = 0; b < batch; ++b)
    {
        int count = 0;
        for(int i = 0; i < hierarchy->groups; i++)
        {
            int group_size = hierarchy->group_size[i];
            softmax(input+b*inputs + count, group_size, temp, output+b*inputs + count);
            count += group_size;
        }
    }
}

// make softmax layer
softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);

    softmax_layer l;
    init_layer(l);

    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = (float *)calloc(inputs*batch, sizeof(float));
    l.delta = (float *)calloc(inputs*batch, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;

    return l;
}

// softmx layer top function
void forward_softmax_layer(const softmax_layer l, network_state state)
{
    int inputs = l.inputs / l.groups;
    int batch = l.batch * l.groups;
    if(l.softmax_tree)
    {
        softmax_tree(state.input, batch, inputs, l.temperature, l.softmax_tree, l.output);
    }
    else
    {
        for(int b = 0; b < batch; b++)
        {
            softmax(state.input+b*inputs, inputs, l.temperature, l.output+b*inputs);
        }
    }
}

// backward softmax function
void backward_softmax_layer(const softmax_layer l, network_state state)
{
    for(int i = 0; i < l.inputs*l.batch; i++)
    {
        state.delta[i] += l.delta[i];
    }
}
