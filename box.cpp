//========================================================================
// Box
//========================================================================
// @brief: sort boxes according to the confidence 

#ifndef SRC_BOX_CPP_
#define SRC_BOX_CPP_

#include "box.h"

// sort boxes
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh)
{
    sortable_box *s = (sortable_box *)calloc(total, sizeof(sortable_box));
    for (int i = 0; i < total; i++)
    {
        s[i].index   = i;
        s[i].classes = 0;
        s[i].probs   = probs;
    }
    for (int k = 0; k < classes; k++)
    {
        for (int i = 0; i < total; i++)
        {
            s[i].classes = k;
        }
        qsort(s, total, sizeof(sortable_box), nms_comparator);
        for (int i = 0; i < total; i++)
        {
            if (probs[s[i].index][k] == 0)
            {
                continue;
            }
            box a = boxes[s[i].index];
            for (int j = i+1; j < total; j++)
            {
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh)
                {
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}

// compare function for qsort
int nms_comparator(const void *pa, const void *pb)
{
    sortable_box a = *(sortable_box *)pa;
    sortable_box b = *(sortable_box *)pb;
    float diff = a.probs[a.index][b.classes] - b.probs[b.index][b.classes];
    if (diff < 0)      return 1;
    else if (diff > 0) return -1;
    return 0;
}

//
float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a,b);
}

// overlap area
float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0)
    {
        return 0;
    }
    float area = w*h;
    return area;
}

// overlap length (width, height, etc.)
// x1, x2 midpoint of the boxes
float overlap(float x1, float w1, float x2, float w2)
{
    float l1     = x1 - w1/2;
    float l2     = x2 - w2/2;
    float left   = l1 > l2 ? l1 : l2;
    float r1     = x1 + w1/2;
    float r2     = x2 + w2/2;
    float right  = r1 < r2 ? r1 : r2;

    return right - left;
}

// union area = total - intersection
float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;

    return u;
}

// select boxes contains a confidence larger than the threshhold
void do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh)
{
    sortable_box *s = (sortable_box *)calloc(total, sizeof(sortable_box));
    for (int i = 0; i < total; i++)
    {
        s[i].index   = i;
        s[i].classes = classes;
        s[i].probs   = probs;
    }
    qsort(s, total, sizeof(sortable_box), nms_comparator);
    for (int i = 0; i < total; i++)
    {
        if (probs[s[i].index][classes] == 0)
        {
            continue;
        }
        box a = boxes[s[i].index];
        for (int j = i+1; j < total; j++)
        {
            box b = boxes[s[j].index];
            if (box_iou(a, b) > thresh)
            {
                for (int k = 0; k < classes+1; k++)
                {
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}

// store value into box
box float_to_box (float *f)
{
    box b;
    b.x = f[0];
    b.y = f[1];
    b.w = f[2];
    b.h = f[3];

    return b;
}

#endif /* SRC_BOX_CPP_ */
