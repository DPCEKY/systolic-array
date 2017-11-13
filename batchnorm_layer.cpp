//========================================================================
// Batchnorm
//========================================================================
// @brief: batchnorm layer

#include "batchnorm_layer.h"

// batchnorm layer top function
void forward_batchnorm_layer(layer l, network_state state)
{
    normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);

    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
}

// scale calculation for backward propagation
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    for(int f = 0; f < n; ++f)
    {
        float sum = 0;
        for(int b = 0; b < batch; ++b)
        {
            for(int i = 0; i < size; ++i)
            {
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

// mean calculation
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
	//
    for(int i = 0; i < filters; ++i)
    {
        mean_delta[i] = 0;
        for (int j = 0; j < batch; ++j)
        {
            for (int k = 0; k < spatial; ++k)
            {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}

// variance calculation
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    for(int i = 0; i < filters; ++i)
    {
        variance_delta[i] = 0;
        for(int j = 0; j < batch; ++j)
        {
            for(int k = 0; k < spatial; ++k)
            {
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}

// barchnorm with delta
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    for(int j = 0; j < batch; ++j)
    {
        for(int f = 0; f < filters; ++f)
        {
            for(int k = 0; k < spatial; ++k)
            {
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f]) + .00001f) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}
