//========================================================================
// Box header file
//========================================================================
// @brief: function prototype & special type definition

#ifndef SRC_BOX_H_
#define SRC_BOX_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// bounding box
typedef struct box
{
	float x;
	float y;
	float w;
	float h;
} box;

// distance of bounding boxes
typedef struct dbox
{
	float dx;
	float dy;
	float dw;
	float dh;
} dbox;

// box for sort
typedef struct sortable_box
{
	int index;
	int classes;
	float **probs;
}sortable_box;

// sort boxes
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
// compare function for qsort
int nms_comparator(const void *pa, const void *pb);
// intersection/union
float box_iou(box a, box b);
// overlap area
float box_intersection(box a, box b);
// overlap length (width, height, etc.)
// x1, x2 midpoint of the boxes
float overlap(float x1, float w1, float x2, float w2);
// union area = total - intersection
float box_union(box a, box b);
// select boxes contains a confidence larger than the threshhold
void do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh);
// transfer float type to stuct box
box float_to_box(float *f);

#endif /* SRC_BOX_H_ */
