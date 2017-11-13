//========================================================================
// Image header file
//========================================================================
// @brief: function prototype & struct type definition

#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>

#include "box.h"
#include "utilities.h"
#include "blas.h"

#define LABEL_SIZE 8
#define LABEL_TYPE 128

// height, weight, channel and data
typedef struct image
{
	int h;
	int w;
	int c;
	float *data;
} image;

//======================================================================================
// Read && resize images
//======================================================================================
// return 8*128*image
// load labels 8(different size), 32~126 (different type)
// store information&value of labels: w,h,c,*data
image **load_alphabet();
// pass value
image load_image_color(char *filename, int w, int h);
// load image top function
image load_image(char *filename, int w, int h, int c);
// return im.data: w(width); h(height); z(depth,channel)
image load_image_stb(char *filename, int channels);
// make image top function
image make_image(int w, int h, int c);
// make an empty image
image make_empty_image(int w, int h, int c);
// resize the given image (w*h)
image resize_image(image im, int w, int h);
// pick up pixel in m.data: x - width, y - height, c - channel
float get_pixel(image m, int x, int y, int c);
// fetch extra pixels
float get_pixel_extend(image m, int x, int y, int c);
// check the validity of data && store data into image
void set_pixel(image m, int x, int y, int c, float val);
// add value to pixels
void add_pixel(image m, int x, int y, int c, float val);

//======================================================================================
// Draw detections & save etc.
//======================================================================================
// draw detecting results
void draw_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **labels, int classes);
// get label
image get_label(image **characters, char *string, int size);
// splite image
image tile_images(image a, image b, int dx);
// border/wrap up image
image border_image(image a, int border);
// copy image
image copy_image(image p);
// embed image (image data transmission)
void embed_image(image source, image dest, int dx, int dy);
// merge images
void composite_image(image source, image dest, int dx, int dy);
// get width of boxes
void draw_box_width(image a, int x1, int y1, int x2, int y2,int w, float r, float g, float b);
// draw one box
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);
// draw labels
void draw_label(image a, int r, int c, image label, const float *rgb);
// get image color
float get_color(int c, int x, int max);
// display image
void show_image(image p, const char *name);
// save image top function
void save_image(image p, const char *name);
// rearrange the output image
void save_image_png(image im, const char *name);
// free allocated memory
void free_image(image p);

#endif /* SRC_IMAGE_H_ */
