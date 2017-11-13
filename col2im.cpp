//========================================================================
// Col2im header file
//========================================================================
// @brief: post-processing image data 

// column to image: filters%batch == 0
void col2img(float *c_col,float *c, int m, int n, int count, int batch)
{
    for(int k = 0; k < count; k++)
    {
        for(int i = 0; i < batch; i++)
        {
            for(int j = 0; j < n; j++)
            {
                c[j+k*n*batch+i*n] = c_col[k*n*batch+i+j*batch];
            }
        }
    }
}

// column to image: filters%batch == 0
void col2img_extra(float *c_col,float *c, int m, int n, int count, int batch)
{
    for(int k = 0; k < count-1; k++)
    {
        for(int i = 0; i < batch; i++)
        {
            for(int j = 0; j < n; j++)
            {
                c[i*n+j+k*n*batch] = c_col[i+j*batch+k*n*batch];
            }
        }
    }

        for(int i = 0; i < m%batch; i++)
        {
            for(int j = 0; j < n; j++)
            {
                c[i*n+j+(m/batch)*n*batch] = c_col[i+j*batch+(m/batch)*n*batch];
            }
        }

}
