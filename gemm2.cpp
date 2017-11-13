//========================================================================
// Gemm header file
//========================================================================
// @brief: top hardware function - convolutional computation

#include "gemm.h"

/*
                                                            //first convolutional layer
    M: l.n - number of filters & number of output channels  e.g. 16         num_filter
    N: out_h * out_w - size of input&output channel         e.g. 416x416    SIZE_CHANNEL
    K: l.size * l.size * l.c - size of one input filter     e.g. 3x3x3      SIZE_FILTER
    A  : weights/filters
    lda: l.size * l.size * l.c - size of one input filter   e.g. 3x3x3
    B  : input images
    ldb: out_h * out_w - size of input&output channel       e.g. 416x416
    C  : output images
    ldc: out_h * out_w - size of input&output channel       e.g. 416x416
 */
void gemm2(float A[MAX_A], float B[MAX_B], float C[MAX_C],int num_filter, int size_channel,int size_filter,\
        int channels, int height, int width, int ksize, int pad)
{

    INPUT_32 weights[SIZE_BATCH][SIZE_FILTER][MAX_FILTER_DEPTH];
    //store image of 3 separate column  
	INPUT_32 line_buffer[NUM_LINE_BUFFER][SIZE_LINE_BUFFER];
    //store image required by filter while computing
	INPUT_32 window_buffer[NUM_WINDOW_BUFFER][SIZE_WINDOW_BUFFER];
	//read image from window_buffer to achieve parallel computing
    INPUT_32 ParallelWindow[SystolicKernelSize][NUM_WINDOW_BUFFER];
    //store output image other than last layer
    OUTPUT_64 output[SIZE_BATCH][SystolicKernelSize];
    //store image of last layer
    OUTPUT_64 output2[SIZE_BATCH];
	//counter
    int index_lb;


#pragma HLS array_partition variable=weights complete dim=1
#pragma HLS array_partition variable=weights complete dim=2
#pragma HLS array_partition variable=line_buffer complete dim=1
#pragma HLS array_partition variable=window_buffer complete dim=1
#pragma HLS array_partition variable=output complete
#pragma HLS array_partition variable=output2 complete
#pragma HLS array_partition variable=ParallelWindow complete dim=1

    //systolic data
    INPUT_32 inA[SIZE_BATCH][SystolicKernelSize];
    INPUT_32 inB[SIZE_BATCH][SystolicKernelSize];
#pragma HLS array_partition variable=inA complete dim=0
#pragma HLS array_partition variable=inB complete dim=0


    

    if (ksize == NUM_WINDOW_BUFFER_EXTRA)
    {
		//different computing core for last layer
        gemm_extra2(A,B,C,size_channel,size_filter,ksize,weights,output2);
    }
    else
    {
        // copy loop: store filters/weights in local BRAM

        Copy_weights:
        for (int i = 0; i < SIZE_BATCH; i++)
        {
            for (int k = 0; k < channels; k++)
            {
                for (int j = 0; j < ksize*ksize; j++)
                {
#pragma HLS PIPELINE II=1
                    int index_j = (j % ksize) * ksize + j / ksize;
                    weights[i][index_j][k] = A[i*ksize*ksize*channels+k*ksize*ksize+j];
                }
            }
        }

        Copy_image1://pads of first column
        for (int w = 0; w < (width+2*pad); w++)
        {
            for (int c = 0; c < channels; c++)
            {
#pragma HLS PIPELINE II=1
                int index = w * channels + c;
                line_buffer[2][index] = B[index];
            }
        }

        Copy_image2://pads for fisrt channel of second column
        for (int c = 0; c < channels; c++)
        {
#pragma HLS PIPELINE II=1
            line_buffer[1][c] = line_buffer[2][c];
            line_buffer[2][c] =  B[(width+2*pad)*channels+c];
        }

        //initialize counters
        int CountStep = 1;
        int step = 0;

        // start calculation
        Cal_h:
        for (int h = 0; h < (height+2*pad-1); h++)
        {
            Cal_w:
            for ( int w = 0; w < (width+2*pad); )
            {
                // last iteration - after last pixel of map - do nothing
                if ((h == height+2*pad-2) && (w == width+2*pad-1))
                {
                    break;
                }

                // second column, first & last row -- read only, prepare image data
                int flag = (h != 0) && (w != 0) && (w != width+2*pad-1);

                // init output array
                Init_output:
                if (flag)
                {
                    for (int i = 0; i < SIZE_BATCH; i++)
                    {
#pragma HLS unroll
                        for (int j = 0; j < SystolicKernelSize; j++)
                        {
#pragma HLS unroll
                            output[i][j] = 0;
                        }
                    }
                }
                // data_fetch & computation
                Cal_c:
                //all systolic array are busy then
                if( flag == 1 || ( h == 0 && w != 0 && w != width + 2 * pad - 1 ) )
                    CountStep = SystolicKernelSize;

                for (int c = 0; c < channels; c++)
                {
#pragma HLS DEPENDENCE variable=line_buffer inter false
#pragma HLS DEPENDENCE variable=index_lb inter false

                    for( step = 0; step < CountStep; step++ ){
                        //when it comes to last row of each col
                        if( w + 1 + step == width + 2 * pad && c == 0 )
                        {
							//width mod SystolicKernelSize == 0 or finished computing this column
                            if( step == 0 )
                                CountStep = 1;
                            //width mod SystolicKernelSize != 0 and finished computing this column
							else
                            {
                                CountStep = step;
                                break;
                            }
                        }

                        // update window buffer
                        ParallelWindow[step][0] = ( window_buffer[0][c] = window_buffer[3][c] );
                        ParallelWindow[step][1] = ( window_buffer[1][c] = window_buffer[4][c] );
                        ParallelWindow[step][2] = ( window_buffer[2][c] = window_buffer[5][c] );
                        ParallelWindow[step][3] = ( window_buffer[3][c] = window_buffer[6][c] );
                        ParallelWindow[step][4] = ( window_buffer[4][c] = window_buffer[7][c] );
                        ParallelWindow[step][5] = ( window_buffer[5][c] = window_buffer[8][c] );
                        // update line buffer
                        int fetch_w;
                        if( w == 0 || w + 1 + step == width + 2 * pad )
                            fetch_w = ( w + 1 ) % ( width + 2 * pad );
                        else
                            fetch_w = ( w + 1 - ( w - 1 ) % SystolicKernelSize ) % ( width + 2 * pad );
                        int fetch_h = h + 1 + ( w + 1 ) / ( width + 2 * pad );
                        //
                        index_lb = fetch_w * channels + c * CountStep + step;  //column
                        int index_input = fetch_h * (width+2*pad) * channels + fetch_w * channels + c * CountStep + step;
                        //read new image data, combine data read before and generate ParallelWindow required by filter
                        ParallelWindow[step][6] = ( window_buffer[6][c] = (line_buffer[0][index_lb] = line_buffer[1][index_lb]) );
                        ParallelWindow[step][7] = ( window_buffer[7][c] = (line_buffer[1][index_lb] = line_buffer[2][index_lb]) );
                        ParallelWindow[step][8] = ( window_buffer[8][c] = (line_buffer[2][index_lb] = B[index_input]) );


                    }



                    // multiplication 16 x SystolicKernelSize using systolic core
                    if (flag)
                    {
				        //init data buffer of systolic core
                        for( int j = 0; j < SIZE_BATCH; j++ ){
#pragma HLS pipeline
                            for( int i = 0; i < SystolicKernelSize; i++ ){
                                inA[j][i]= 0;
                                inB[j][i] = 0;
                            }
                        }

                        //Iteration cycles determined by both array
                        for( int r = 0; r < SIZE_BATCH + SIZE_FILTER + step - 2; r++ ){
#pragma HLS pipeline

                            for (int i = 0; i < SIZE_BATCH; i++)
                                for (int j = SystolicKernelSize - 1; j >= 1; j--)
                                    inA[i][j] = inA[i][j-1];

                            for (int i = SIZE_BATCH - 1; i >= 1; i--)
                                for (int j = 0; j < SystolicKernelSize; j++)
                                    inB[i][j] = inB[i-1][j];


                            for( int i = 0; i < SIZE_BATCH; i++ )
                                if( r >= i && r < i + SIZE_FILTER )
                                    inA[i][0] = weights[i][r-i][c];
                                else
                                    inA[i][0] = 0;

                            for (int j = 0; j < SystolicKernelSize; j++)
                                if( r >= j && r < j + SIZE_FILTER )
                                    inB[0][j] = ParallelWindow[j][r-j];
                                else
                                    inB[0][j] = 0;

                            //PE
                            for( int i = 0; i < SIZE_BATCH; i++ )
                                for( int j = 0; j < SystolicKernelSize; j++ )
                                    output[i][j] += inA[i][j] * inB[i][j];

                        }

                    }
                }
                // output results
                if (flag)
                {
                    for( int OutChannel = 0; OutChannel < step; OutChannel++ ){

                        int index_c = ( h - 1 ) * width + w - 1 + OutChannel;
                        Output:
                        for (int i = 0; i < SIZE_BATCH; i++)
                        {
#pragma HLS DEPENDENCE variable=output inter false
#pragma HLS PIPELINE II=1
                            // output final result

                            C[index_c*SIZE_BATCH+i] = output[i][OutChannel];

                        }
                    }
                }

                if( w + 1 + step == width + 2 * pad )
                    CountStep = 1;
                w += step;
            }
        }
    }

}

// extra gemm with filter size 1x1
void gemm_extra2(float A[MAX_A], float B[MAX_B], float C[MAX_C],int size_channel,int size_filter,int ksize,
        INPUT_32 weights[SIZE_BATCH][SIZE_FILTER][MAX_FILTER_DEPTH],OUTPUT_64 output[SIZE_BATCH])
{
    // copy loop: store weights/filters in local BRAM
    Copy_weights_E:
    for (int i = 0; i < SIZE_BATCH; i++)
    {
        for (int k = 0; k < size_filter; k++) //1x1x425
        {
            for (int j = 0; j < ksize*ksize; j++) //1x1
            {
#pragma HLS PIPELINE II=1
                weights[i][j][k] = A[i*ksize*ksize*size_filter+k*ksize*ksize+j];
            }
        }
    }
    // start calculation
    Cal_t_E:
    for (int i = 0; i < size_channel; i++)
    {
        // init output
        Init_E:
        for (int k = 0; k < SIZE_BATCH; k++)
        {
#pragma HLS unroll
            output[k] = 0;
        }
        // start calculation 1024 mul+add
        Cal_L1_E:
        for (int j = 0; j < size_filter; j++)
        {
#pragma HLS PIPELINE II=1
            INPUT_32 input = B[i*size_filter+j];
            for (int k = 0; k < SIZE_BATCH; k++)
            {
                Cal_L2_E:
                output[k] += input * weights[k][0][j];
            }
        }
        // output results
        Output_E:
        for (int j = 0; j < SIZE_BATCH; j++)
        {
#pragma HLS PIPELINE II=1
            C[i*SIZE_BATCH+j] = output[j];
        }
    }
}


