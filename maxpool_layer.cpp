//========================================================================
// Maxpooling
//========================================================================
// @brief: maxpooling layer

#include "maxpool_layer.h"

// make maxpooling layer
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l;
    init_layer(l);

    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + 2*padding)/stride;
    l.out_h = (h + 2*padding)/stride;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    //printf("\noutput_size: %d;\n",output_size);
    l.indexes = (int *)calloc(output_size, sizeof(int));
    l.output  = (float *)calloc(output_size, sizeof(float));
    l.delta   = (float *)calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    //l.backward = backward_maxpool_layer;
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

// maxpooling top function
void forward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int w_offset = -l.pad;
    int h_offset = -l.pad;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    //
    for (int b = 0; b < l.batch; b++)
    {
        for (int k = 0; k < c; k++)
        {
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (int n = 0; n < l.size; n++)
                    {
                        for (int m = 0; m < l.size; m++)
                        {
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h && cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? state.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                    //printf("l.output[%d]:%f;\n",out_index,l.output[out_index]);
                }
            }
        }
    }
/*
    //
    for (int x = 900; x < 1000; x++)
    {
        printf("state.input[%d]:%.12f; l.output[%d]:%.12f;\n",x,state.input[x],x,l.output[x]);
    }
*/
}

