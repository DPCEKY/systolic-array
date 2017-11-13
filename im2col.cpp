//========================================================================
// Im2col 
//========================================================================
// @brief: pre-processing image data

#include "gemm.h"
#include "im2col.h"

// image to column : filters%batch == 0
void im2col(float *data_im,int channels, int height, int width, int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = height + 2*pad;
    int width_col = width + 2*pad;
    float temp;
    int step;
    for(c = 0; c < channels; c++)
    {
    	for(h = 0; h < height_col; h++)
    	{
    		for(w = 0; w < width_col; w++)
    		{

		        //for those width mod SystolicKernelSize != 0 and comes to last few points in each column
    			if( ( w - ( w - 2 ) % SystolicKernelSize + SystolicKernelSize ) > width_col )
    				step = ( width_col - 2 ) % SystolicKernelSize;
    			else
    				step = SystolicKernelSize;
                //pad
    			if((w == 0) || (h == 0) || (w == width_col-1) || (h == height_col-1))
    				temp = 0;
    			//read data
				else
    				temp = data_im[c*width*height+(h-1)*width+(w-1)];
                
				//first two channel are directly read into buffer, thus it is transferred directly
    			if( w == 0 || w == 1 )
    			{
    				data_col[h * width_col * channels + w * channels + c] = temp;
    			}
				//deal with data other than first two channels: pls refer to report
    			else
    			    data_col[h * width_col * channels + ( w - ( w - 2 ) % SystolicKernelSize ) * channels + step * c + ( w - 2 ) % SystolicKernelSize ] = temp;
    		}
    	}
    }
}

// image to column : filters%batch != 0
void im2col_extra(float *data_im,int channels, int height, int width, int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    float temp;
    for(w = 0; w < width; w++)
    {
        for(h = 0; h < height; h++)
        {
            for(c = 0; c < channels; c++)
            {
                int index_col = (w+h*width)*channels+c;
                int index_im = c*width*height+h*width+w;
                data_col[index_col] = data_im[index_im];
            }
        }
    }
}

