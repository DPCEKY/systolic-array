//========================================================================
// Network
//========================================================================
// @brief: network layer

#include "network.h"

// make a new && empty network with n layers
network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.layers = (layer *)calloc(net.n, sizeof(layer));
    net.seen = (int *)calloc(1, sizeof(int));  // what's net.seen? 1 integer

    return net;
}

// get output size from the layer with type COST
int get_network_output_size(network net)
{
    int i;
    for (i = net.n - 1; i > 0; i--)
    {
        if (net.layers[i].type != COST)
        {
            break;
        }
    }
    return net.layers[i].outputs;
}

// get output from the layer with type COST
float *get_network_output(network net)
{
    int i;
    for (i = net.n - 1; i > 0; i--)
    {
        if (net.layers[i].type != COST)
        {
            break;
        }
    }
    return net.layers[i].output;
}

// set batch size for each layer in the network
void set_batch_network(network *net, int b)
{
    net->batch = b;
    for (int i = 0; i < net->n; i++)
    {
        net->layers[i].batch = b;
    }
}

// top forward function, return final output ???
float *network_predict(network net, float *input)
{
    printf("network_predict.\n");
    network_state state;
    state.net   = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    //
    forward_network(net, state);
    float *out = get_network_output(net);

    return out;
}

// go through all network layers ************************
void forward_network(network net, network_state state)
{
    state.workspace = net.workspace;
    for (int i = 0; i < net.n; i++)
    {
        //printf("predicting: layer NO. %d.\n",i);
        //Timer timer("the whole layer");
        //timer.start();
        state.index = i;
        layer l = net.layers[i];
        // delta = 0
        if (l.delta)
        {
            //Timer timer2("scale_cpu");
            //timer2.start();
            //printf("l.delta\n");
            scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
            //printf("ch1\n");
            //timer2.stop();
        }
        //printf("ch2\n");
        l.forward(l, state);
        state.input = l.output;
        //timer.stop();
    }
}

