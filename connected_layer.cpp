//========================================================================
// Connected layer
//========================================================================
// @brief: connected layer

#include "connected_layer.h"

// make connected layer
connected_layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize)
{
    connected_layer l;
    init_layer(l);
    //
    l.type            = CONNECTED;
    l.inputs          = inputs;
    l.outputs         = outputs;
    l.batch           = batch;
    l.batch_normalize = batch_normalize;
    //
    l.h     = 1;
    l.w     = 1;
    l.c     = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    //
    l.output = (float *)calloc(batch*outputs, sizeof(float));
    l.delta  = (float *)calloc(batch*outputs, sizeof(float));
    //
    l.weight_updates = (float *)calloc(inputs*outputs, sizeof(float));
    l.bias_updates   = (float *)calloc(inputs*outputs, sizeof(float));
    //
    l.weights = (float *)calloc(outputs*inputs, sizeof(float));
    l.biases  = (float *)calloc(outputs, sizeof(float));
    // function pointers
    l.forward  = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update   = update_connected_layer;
    //
    float scale = sqrt(2.0/inputs);
    for (int i = 0; i < outputs*inputs; i++)
    {
        l.weights[i] = scale * rand_uniform(-1,1);
    }
    //
    for (int i = 0; i < outputs; i++)
    {
        l.biases[i] = 0;
    }
    //
    if (batch_normalize)
    {
        l.scales = (float *)calloc(outputs, sizeof(float));
        l.scale_updates = (float *)calloc(outputs, sizeof(float));
        //
        for (int i = 0; i < outputs; i++)
        {
            l.scales[i] = 1;
        }
        //
        l.mean           = (float *)calloc(outputs, sizeof(float));
        l.mean_delta     = (float *)calloc(outputs, sizeof(float));
        l.variance       = (float *)calloc(outputs, sizeof(float));
        l.variance_delta = (float *)calloc(outputs, sizeof(float));
        //
        l.rolling_mean     = (float *)calloc(outputs, sizeof(float));
        l.rolling_variance = (float *)calloc(outputs, sizeof(float));
        //
        l.x      = (float *)calloc(outputs, sizeof(float));
        l.x_norm = (float *)calloc(outputs, sizeof(float));
    }
    //
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

//
void forward_connected_layer(connected_layer l, network_state state)
{
    // empty the l.output array
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = state.input;
    float *b = l.weights;
    float *c = l.output;
    // 
    //gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    //
    if(l.batch_normalize)
    {
        if(state.train)
        {
            mean_cpu(l.output, l.batch, l.outputs, 1, l.mean);
            variance_cpu(l.output, l.mean, l.batch, l.outputs, 1, l.variance);
            //
            scal_cpu(l.outputs, .95, l.rolling_mean, 1);
            axpy_cpu(l.outputs, .05, l.mean, 1, l.rolling_mean, 1);
            scal_cpu(l.outputs, .95, l.rolling_variance, 1);
            axpy_cpu(l.outputs, .05, l.variance, 1, l.rolling_variance, 1);

            copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
            normalize_cpu(l.output, l.mean, l.variance, l.batch, l.outputs, 1);
            copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
        }
        else
        {
            normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.outputs, 1);
        }
        scale_bias(l.output, l.scales, l.batch, l.outputs, 1);
    }
}

//
void backward_connected_layer(connected_layer l, network_state state)
{
    //
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    //
    for (int i = 0; i < l.batch; i++)
    {
        axpy_cpu(l.outputs, 1, l.delta + i*l.outputs, 1, l.bias_updates, 1);
    }
    //
    if(l.batch_normalize)
    {
        backward_scale_cpu(l.x_norm, l.delta, l.batch, l.outputs, 1, l.scale_updates);

        scale_bias(l.delta, l.scales, l.batch, l.outputs, 1);

        mean_delta_cpu(l.delta, l.variance, l.batch, l.outputs, 1, l.mean_delta);
        variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.outputs, 1, l.variance_delta);
        normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.outputs, 1, l.delta);
    }
    //
    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = state.input;
    float *c = l.weight_updates;
    //
    m = l.batch;
    k = l.outputs;
    n = l.inputs;
    //
    a = l.delta;
    b = l.weights;
    c = state.delta;
    //
    if(c)
    {
        //gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }
}

//
void update_connected_layer(connected_layer l, int batch, float learning_rate, float momentum, float decay)
{
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if(l.batch_normalize)
    {
        axpy_cpu(l.outputs, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

//
void denormalize_connected_layer(layer l)
{
    //
    for (int i = 0; i < l.outputs; i++)
    {
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + 0.000001);
        for (int j = 0; j < l.inputs; i++)
        {
            l.weights[i*l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

//
void statistics_connected_layer(layer l)
{
    if(l.batch_normalize)
    {
        printf("Scales ");
        print_statistics(l.scales, l.outputs);   //???
        /*
        printf("Rolling Mean ");
        print_statistics(l.rolling_mean, l.outputs);
        printf("Rolling Variance ");
        print_statistics(l.rolling_variance, l.outputs);
        */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}
