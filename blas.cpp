//========================================================================
// Blas
//========================================================================
// @brief: helper function for barchnorm layer

#include "blas.h"

// multiply some values in *X with ALPHA
void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    for (int i = 0; i < N; i++)
    {
        X[i] = 0;
    }
}

// assign some values in *X with ALPHA
void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    for (int i = 0; i < N; i++)
    {
        X[i] = 0;
    }
}

// calculation about *mean
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1.0/(batch * spatial);
    //
    for (int i = 0; i < filters; i++)
    {
        mean[i] = 0;
        for (int j = 0; j < batch; j++)
        {
            for (int k = 0; k < spatial; k++)
            {
                int index = j*filters*spatial + i*spatial + k;
                mean[i]  += x[index];
            }
        }
        mean[i] *= scale;
    }
}

// calculation about *variance
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1.0/(batch * spatial - 1);
    //
    for (int i = 0; i < filters; i++)
    {
        variance[i] = 0;
        for (int j = 0; j < batch; j++)
        {
            for (int k = 0; k < spatial; k++)
            {
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

// multiply some values in *X with ALPHA
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    for (int i = 0; i < N; i++)
    {
        Y[i*INCY] += ALPHA*X[i*INCX];
    }
}

// array copy
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    for (int i = 0; i < N; i++)
    {
        Y[i*INCY] = X[i*INCX];
    }
}

// normalization with mean and variance
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    for (int j = 0; j < batch; j++)
    {
        for (int k = 0; k < filters; k++)
        {
            float p = sqrt(variance[k])+0.000001f;
            for (int i = 0; i < spatial; i++)
            {
                int index = j*filters*spatial + k*spatial + i;
                x[index] = (x[index] - mean[k])/p;
                //x[index] *= scales[k];
                //x[index] += bias[k];
            }
        }
    }
}

// scale an array
void scale_cpu(int N, float ALPHA, float *X, int INCX)
{
    for (int i = 0; i < N; i++)
    {
        X[i*INCX] *= ALPHA;
    }
}

// flatten layer
void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = (float *)calloc(size*layers*batch, sizeof(float));
    //
    for(int b = 0; b < batch; ++b)
    {
        for(int c = 0; c < layers; ++c)
        {
            for(int i = 0; i < size; ++i)
            {
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward)
                {
                    swap[i2] = x[i1];
                }
                else
                {
                    swap[i1] = x[i2];
                }
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

// softmax layer
void softmax(float *input, int n, float temp, float *output)
{
    float sum = 0;
    float largest = -FLT_MAX;
    for(int i = 0; i < n; i++)
    {
        if(input[i] > largest)
        {
            largest = input[i];
        }
    }
    for(int i = 0; i < n; i++)
    {
        float e = exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for(int i = 0; i < n; i++)
    {
        output[i] /= sum;
    }
}
