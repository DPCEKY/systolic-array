//========================================================================
// Convolutional layer
//========================================================================
// @brief: convolutional layer

#include "convolutional_layer.h"
#include <time.h>

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride,\
        int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    convolutional_layer l;
    init_layer(l);
    l.type = CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    l.weights = (float *)calloc(c*n*size*size, sizeof(float));
    //l.weights = (float *)sds_alloc(c*n*size*size * sizeof(float));
    l.weight_updates = (float *)calloc(c*n*size*size, sizeof(float));

    l.biases = (float *)calloc(n, sizeof(float));
    l.bias_updates = (float *)calloc(n, sizeof(float));

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    for (int i = 0; i < c*n*size*size; i++)
    {
        l.weights[i] = scale*rand_uniform(-1, 1);
    }
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    l.output = (float *)calloc(l.batch*l.outputs, sizeof(float));
    //l.output = (float *)sds_alloc(l.batch*l.outputs * sizeof(float));
    l.delta  = (float *)calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_convolutional_layer;
    //l.backward = backward_convolutional_layer;
    //l.update = update_convolutional_layer;
   /*if(binary){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.cweights = calloc(c*n*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }*/

    if(batch_normalize)
    {
        l.scales = (float *)calloc(n, sizeof(float));
        l.scale_updates = (float *)calloc(n, sizeof(float));
        for(int i = 0; i < n; i++)
        {
            l.scales[i] = 1;
        }

        l.mean = (float *)calloc(n, sizeof(float));
        l.variance = (float *)calloc(n, sizeof(float));

        l.mean_delta = (float *)calloc(n, sizeof(float));
        l.variance_delta = (float *)calloc(n, sizeof(float));

        l.rolling_mean = (float *)calloc(n, sizeof(float));
        l.rolling_variance = (float *)calloc(n, sizeof(float));
        l.x = (float *)calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = (float *)calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam)
    {
        l.adam = 1;
        l.m = (float *)calloc(c*n*size*size, sizeof(float));
        l.v = (float *)calloc(c*n*size*size, sizeof(float));
    }

    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

// get the size of output image 
size_t get_workspace_size(layer l)
{
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}

// forward to convolutional layer
void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    //printf("ch3\n");
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    // init l.output = 0
    //Timer timer2("fill_cpu");
    //timer2.start();
    fill_cpu(l.outputs * l.batch, 0, l.output, 1);
    //timer2.stop();
    //printf("ch4\n");
    //
    //Timer timer9("part left");
    //timer9.start();
    int m = l.n;
    int k = l.size * l.size * l.c;
    int n = out_h * out_w ;
    int n2 = (out_h + 2) * (out_w + 2);
    //
    float *a = l.weights;
    float *b = state.workspace;
    float *c = l.output;

	//Storing iamge
    b = (float *) sds_alloc ( l.c * n2 * sizeof(float) );
    //Storing convolution results
    float *c_col = (float *) sds_alloc ((m*n+n*(SIZE_BATCH-m%SIZE_BATCH)) * sizeof(float));
    //Storing weights
    float a_buf[3*3*1024*16];
    
    int count;
    int batch;
	//Change input image format
    if(m%SIZE_BATCH == 0)
    {
        count = m/SIZE_BATCH;
        batch = SIZE_BATCH;
        im2col(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
    }
    else
    {
		//for last layer: no special format
        count = m/SIZE_BATCH+1;
        batch = SIZE_BATCH;
        im2col_extra(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
    }
    
    int aiCount = batch*k;
    struct timeval t0, t1;
    static double Duration = 0;
    double DurationTemp = Duration;

    for (int x = 0; x < count; x++)
    {
		//Give the layer being calculated
    	if( x == 0 ){
        	printf("Layer info: channel = %d, x = %d, m = %d, n = %d\n", l.c, x, m, n);
        	fflush(stdout);
    	}
		//Copying weights
    	if( m % SIZE_BATCH != 0 && x == count - 1 )
    		aiCount = ( m % SIZE_BATCH ) * k;
    	for( int ai = 0; ai < aiCount; ai++ ){
    		a_buf[ai] = a[ai + x*batch*k];
    	}
        
        gettimeofday(&t0, 0);
        gemm2( a_buf,b,c_col,m,n,k,l.c,l.h,l.w,l.size, l.pad);
        gettimeofday(&t1, 0);
        Duration += (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
		
        c_col += batch * n;
    }

    std::cout << "Duration for channel = " << l.c << " : " << ( Duration - DurationTemp ) / 1000 << " msec" << std::endl;
    if( m == 425 )
        std::cout << "Duration in all: " << Duration / 1000 << " msec" << std::endl;

	//Transfer the output data format back
    if(m%SIZE_BATCH ==0)
    {
        c_col -= m*n;
        col2img(c_col,c,m,n,count,SIZE_BATCH);
    }
    else
    {
        c_col -= count * n * SIZE_BATCH;
        col2img_extra(c_col,c,m,n,count,SIZE_BATCH);
    }
    sds_free(b);
    sds_free(c_col);

    if (l.batch_normalize)
    {
        forward_batchnorm_layer(l, state);
    }
    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    activate_array(l.output, m*n*l.batch, l.activation);

}

// calculate output height
int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size)/l.stride + 1;
}

// calculate output_weight
int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size)/l.stride + 1;
}

// add bias to output values
void add_bias(float *output, float *biases, int batch, int n, int size)
{
    //
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < size; j++)
            {
                output[(b*n+i)*size + j] += biases[i];
            }
        }
    }
}

// scale bias
void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    //
    for(int b = 0; b < batch; b++)
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < size; j++)
            {
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

// scale bias
void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;
}
