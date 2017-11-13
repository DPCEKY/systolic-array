//========================================================================
// Image
//========================================================================
// @brief: loading image data

#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_read.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int windows = 0;
float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

// return 8*128*image
// load labels 8(different size), 32~126 (different type)
// store information&value of labels: w,h,c,*data
image **load_alphabet()
{
    const int nsize = LABEL_SIZE;
    image **alphabets = (image **)calloc(nsize, sizeof(image *));
    //
    for (int j = 0; j < nsize; j++)
    {
        alphabets[j] = (image *)calloc(LABEL_TYPE, sizeof(image));
        for (int i = 32; i < LABEL_TYPE - 1; i++)
        {
            char buffer[256];
            sprintf(buffer, "data/labels/%d_%d.png", i, j);
            // buffer: filename of labels
            alphabets[j][i] = load_image_color(buffer, 0, 0);
        }
    }
    return alphabets;
}

// pass value?
image load_image_color(char *filename, int w, int h)
{
    return load_image(filename, w, h, 3);
}

// load image top function
image load_image(char *filename, int w, int h, int c) //003
{
    // load image value BUG
    image out = load_image_stb(filename, c);

    //printf("out.h: %d; out.w: %d; out.c: %d;\n",out.h,out.w,out.c);
    // when resize???
    if((h && w) && (h != out.h || w != out.w))
    {
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

// return im.data: w(width); h(height); z(depth,channel)
image load_image_stb(char *filename, int channels) // filename, 3
{
    int w, h, c;
    // standard image load function
    // stbi_load output: z(depth, channel); w(width); h(height)
    //printf("filename: %s; channels: %d;\n",filename,channels);
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    //printf("out.h: %d; out.w: %d; out.c: %d;\n",h,w,c);

    if(channels)
    {
        c = channels;
    }
    // make new image
    image im = make_image(w, h, c);
    for (int k = 0; k < c; k++)
    {
        for (int j = 0; j < h; j++)
        {
            for(int i = 0; i < w; i++)
            {
                int index_dst = i + w*j + w*h*k;
                int index_src = k + c*i + c*w*j;
                im.data[index_dst] = (float)data[index_src]/255.0;
            }
        }
    }
    free(data);
    return im;
}

// make_image top function
image make_image(int w, int h, int c)
{
    image out = make_empty_image(w, h, c);
    out.data  = (float *)calloc(h*w*c, sizeof(float));
    return out;
}

// make empty image (data pointer: 0)
image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h    = h;
    out.w    = w;
    out.c    = c;

    return out;
}

// resize the given image (w*h)
image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);
    image part    = make_image(w, im.h, im.c);

    float w_scale = (float)(im.w - 1)/(w - 1);
    float h_scale = (float)(im.h - 1)/(h - 1);
    // stage 1: resize image within given width (column)
    for (int k = 0; k < im.c; k++)
    {
        for (int r = 0; r < im.h; r++) // row
        {
            for(int c = 0; c < w; c++) // column
            {
                float val = 0;
                // last column || only one column
                if (c == w-1 || im.w == 1)
                {   // simply fetch the original final column
                    val = get_pixel(im, im.w - 1, r, k);
                }
                else
                {
                    float sx = c*w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    // weighted sum for other columns
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                // store val into image part
                set_pixel(part, c, r, k, val);
            }
        }
    }
    // stage 2: resize image within given height (row)
    for (int k = 0; k < im.c; k++)
    {
        for(int r = 0; r < h; r++)
        {
            float sy = r*h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            //
            for (int c = 0; c < w; c++)
            {
                float val = (1 - dy) * get_pixel(part, c, iy, k);
                // store val into image resized
                set_pixel(resized, c, r, k, val);
            }
            // the last row || only one row
            if (r == h-1 || im.h == 1)
            {
                continue;
            }
            //
            for (int c = 0; c < w; c++)
            {
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }
    free_image(part);
    return resized;
}

// pick up pixel in m.data: x - width, y - height, c - channel
float get_pixel(image m, int x, int y, int c)
{
    // x < m.w && y < m.h && c < m.c == 0: assert
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

// fetch extra pixels
float get_pixel_extend(image m, int x, int y, int c)
{
    if(x < 0)
    {
        x = 0;
    }
    if(x >= m.w)
    {
        x = m.w-1;
    }
    if(y < 0)
    {
        y = 0;
    }
    if(y >= m.h)
    {
        y = m.h-1;
    }
    if(c < 0 || c >= m.c)
    {
        return 0;
    }
    return get_pixel(m, x, y, c);
}

// check the validity of data && store data into image
void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c)
    {
        return;
    }
    // x < m.w && y < m.h && c < m.c == 0: assert
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

// add value to pixels
void add_pixel(image m, int x, int y, int c, float val)
{
    // x < m.w && y < m.h && c < m.c == 0: assert
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

// draw detecting results
void draw_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes)
{
    //printf("probs[176][2]:%.12f; probs[176][7]:%.12f;\n",probs[176][2],probs[176][7]);
    for (int i = 0; i < num; i++)
    {
        //printf("ch0;\n");
        int class_max = max_index(probs[i], classes); // max_index???
        float prob = probs[i][class_max];
/*
        if(probs[i][class_max] != 0)
        {
            printf("i:%d; class_max:%d; prob:%.12f; \n",i,class_max,prob);
        }
*/
        if (prob > thresh)
        {
            //printf("ch1;\n");
            int width = im.h * 0.012;
            /* ??????????????????????????????????????
            if(0)
            {
                width = pow(prob, 1.0/2.0)*10 + 1;
                alphabet = 0;
            }
             */
            printf("%s: %.0f%%\n", names[class_max], prob*100);
            int offset  = class_max * 123457 % classes;
            float red   = get_color(2, offset, classes);
            float green = get_color(1, offset, classes);
            float blue  = get_color(0, offset, classes);
            float rgb[3];
            //
            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b  = boxes[i];
            //
            int left  = (b.x-b.w/2.0)*im.w;
            int right = (b.x+b.w/2.0)*im.w;
            int top   = (b.y-b.h/2.0)*im.h;
            int bot   = (b.y+b.h/2.0)*im.h;
            //
            if(left < 0)       left  = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0)        top   = 0;
            if(bot > im.h-1)   bot   = im.h-1;
            //printf("ch2;\n");
            //
            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            //printf("ch3;\n");
            if(alphabet)
            {
                image label = get_label(alphabet, names[class_max], (im.h*0.03)/10);
                draw_label(im, top + width, left, label, rgb);
            }
        }
        //printf("ch4;\n");
    }
}

// get label
image get_label(image **characters, char *string, int size)
{
    if (size > 7)
    {
        size = 7;
    }
    image label = make_empty_image(0, 0, 0);
    //
    while(*string)
    {
        image l = characters[size][(int)*string];
        image n = tile_images(label, l, -size - 1 + (size+1)/2);
        free_image(label);
        label = n;
        string++;
    }
    image b = border_image(label, label.h*.25);
    free_image(label);

    return b;
}

// splite image
image tile_images(image a, image b, int dx)
{
    if(a.w == 0)
    {
        return copy_image(b);
    }
    image c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
    fill_cpu(c.w*c.h*c.c, 1, c.data, 1);
    embed_image(a, c, 0, 0);
    composite_image(b, c, a.w + dx, 0);

    return c;
}

// border/wrap up image
image border_image(image a, int border)
{
    image b = make_image(a.w + 2*border, a.h + 2*border, a.c);
    //

    for(int k = 0; k < b.c; ++k)
    {
       for(int y = 0; y < b.h; ++y)
       {
           for(int x = 0; x < b.w; ++x)
           {
               float val = get_pixel_extend(a, x - border, y - border, k);
               if(x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h)
               {
                   val = 1;
               }
               set_pixel(b, x, y, k, val);
            }
       }
    }

    return b;
}

// copy image
image copy_image(image p)
{
    image copy = p;
    copy.data = (float *)calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));

    return copy;
}

// embed image (image data transmission)
void embed_image(image source, image dest, int dx, int dy)
{
    for(int k = 0; k < source.c; k++)
    {

        for(int y = 0; y < source.h; y++)
        {
            for(int x = 0; x < source.w; x++)
            {
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

// merge images
void composite_image(image source, image dest, int dx, int dy)
{
    for (int k = 0; k < source.c; k++)
    {
        for (int y = 0; y < source.h; y++)
        {
            for (int x = 0; x < source.w; x++)
            {
                float val = get_pixel(source, x, y, k);
                float val2 = get_pixel_extend(dest, dx+x, dy+y, k);
                set_pixel(dest, dx+x, dy+y, k, val * val2);
            }
        }
    }
}

// get width of boxes
void draw_box_width(image a, int x1, int y1, int x2, int y2,int w, float r, float g, float b)
{
    for (int i = 0; i < w; i++)
    {
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}

// draw one box
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    // normalize_image(a)
    // ensure the boxed in the picture
    if(x1 < 0)    x1 = 0;
    if(x1 >= a.w) x1 = a.w-1;
    if(x2 < 0)    x2 = 0;
    if(x2 >= a.w) x2 = a.w-1;

    if(y1 < 0)    y1 = 0;
    if(y1 >= a.h) y1 = a.h-1;
    if(y2 < 0)    y2 = 0;
    if(y2 >= a.h) y2 = a.h-1;
    // draw boxes: rgb
    for (int i = x1; i <= x2; i++)
    {   // two horizontal lines
        a.data[i + y1*a.w + 0*a.w*a.h] = r;
        a.data[i + y2*a.w + 0*a.w*a.h] = r;
        a.data[i + y1*a.w + 1*a.w*a.h] = g;
        a.data[i + y2*a.w + 1*a.w*a.h] = g;
        a.data[i + y1*a.w + 2*a.w*a.h] = b;
        a.data[i + y2*a.w + 2*a.w*a.h] = b;
    }
    for (int i = y1; i <= y2; i++)
    {   // two vertical lines
        a.data[x1 + i*a.w + 0*a.w*a.h] = r;
        a.data[x2 + i*a.w + 0*a.w*a.h] = r;
        a.data[x1 + i*a.w + 1*a.w*a.h] = g;
        a.data[x2 + i*a.w + 1*a.w*a.h] = g;
        a.data[x1 + i*a.w + 2*a.w*a.h] = b;
        a.data[x2 + i*a.w + 2*a.w*a.h] = b;
    }
}

// draw labels
void draw_label(image a, int r, int c, image label, const float *rgb)
{
    int w = label.w;
    int h = label.h;
    if(r - h >= 0)
    {
        r = r - h;
    }
    // replace corresponding pixels for labels
    for (int j = 0; j < h && j + r < a.h; j++)
    {
        for (int i = 0; i < w && i + c < a.w; i++)
        {
            for (int k = 0; k < label.c; k++)
            {
                float val = get_pixel(label, i, j, k);
                set_pixel(a, i+c, j+r, k, rgb[k] * val);
            }
        }
    }
}

// get image color
float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio); 
    int j = ceil(ratio);  
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];

    return r;
}

// display image
void show_image(image p, const char *name)
{
    fprintf(stderr,"Not compiled with OpenCV, saving to %s.png instead.\n", name);
    save_image(p, name);
}

// save image top function
void save_image(image im, const char *name)
{
    save_image_png(im, name);
}

// data: one pixel(three channels); im.data: all pixels for one channel, next channel, etc.
void save_image_png(image im, const char *name)
{
    char buffer[256];
    // save picture name into buffer
    sprintf(buffer, "%s.png", name);
    unsigned char *data = (unsigned char *)calloc(im.w*im.h*im.c, sizeof(char));
    //
    //printf("ch0;\n");
    for(int k = 0; k < im.c; k++)
    {
        for(int i = 0; i < im.w*im.h; i++)
        {
            data[i*im.c + k] = (unsigned char) (255 * im.data[i + k*im.w*im.h]);
        }
    }
    //printf("ch1;\n");
    int success = stbi_write_png(buffer, im.w, im.h, im.c, data, im.w*im.c);
    //printf("ch2;\n");
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buffer);
}

// free allocated memory
void free_image(image m)
{
    if(m.data)
    {
        free(m.data);
    }
}
