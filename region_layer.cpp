//========================================================================
// Region layer
//========================================================================
// @brief: get predictions (final bounding boxes)

#include "region_layer.h"

// make region layer
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l;
    init_layer(l);

    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.classes = classes;
    l.coords = coords;
    l.cost = (float *)calloc(1, sizeof(float));
    l.biases = (float *)calloc(n*2, sizeof(float));
    l.bias_updates = (float *)calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30*(5);
    l.delta = (float *)calloc(batch*l.outputs, sizeof(float));
    l.output = (float *)calloc(batch*l.outputs, sizeof(float));
    //
    for(int i = 0; i < n*2; i++){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    //l.backward = backward_region_layer;
    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}

// get bounding boxes
//                          l      1      1        0.24            probs       boxes       0                   0                 0.5
void get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map, float tree_thresh)
{
    //
    float *predictions = l.output;
/*
    // print output
    printf("l.outputs:%d;\n",l.outputs);
    for (int i = 0; i < 100; i++)
    {
        printf("predictions[%d]:%.12f;\n",i,predictions[i]);
    }
*/

    //cover l.w * l.h grids  l.n = 5 5boxes  l.classes = 20
    for (int i = 0; i < l.w*l.h; i++){
        int row = i / l.w;
        int col = i % l.w;
        for(int n = 0; n < l.n; n++){
            int index = i*l.n + n;
            int p_index = index * (l.classes + 5) + 4;
            float scale = predictions[p_index];
            int box_index = index * (l.classes + 5);
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;

            int class_index = index * (l.classes + 5) + 5;
            if(l.softmax_tree)
            {
                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
                if(map)
                {
                    for(int j = 0; j < 200; j++)
                    {
                        float prob = scale*predictions[class_index+map[j]];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                }
                else
                {
                    int j =  hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh);
                    probs[index][j] = (scale > thresh) ? scale : 0;
                    probs[index][l.classes] = scale;
                }
            }
            else
            {
                for(int j = 0; j < l.classes; ++j){
                    float prob = scale*predictions[class_index+j];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                }
            }
            if(only_objectness)
            {
                probs[index][0] = scale;
            }
        }
    }
}

// get bounding box (single)
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;
    b.y = (j + logistic_activate(x[index + 1])) / h;
    b.w = exp(x[index + 2]) * biases[2*n]   / w;
    b.h = exp(x[index + 3]) * biases[2*n+1] / h;

    return b;
}

// region layer top function
void forward_region_layer(const layer l, network_state state)
{
    int i,j,b,t,n;
    int size = l.coords + l.classes + 1;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
/*
    // input of region layer
    for (int x = 0; x < 100; x++)
    {
        printf("l.output[%d]:%.12f;\n",x,l.output[x]);
    }
*/


#ifndef GPU
    flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
#endif
    for (b = 0; b < l.batch; ++b){
        for(i = 0; i < l.h*l.w*l.n; ++i){
            int index = size*i + b*l.outputs;
            l.output[index + 4] = logistic_activate(l.output[index + 4]);
        }
    }


#ifndef GPU
    if (l.softmax_tree){
        for (b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w*l.n; ++i){
                int index = size*i + b*l.outputs;
                softmax_tree(l.output + index + 5, 1, 0, 1, l.softmax_tree, l.output + index + 5);
            }
        }
    } else if (l.softmax){
        for (b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w*l.n; ++i){
                int index = size*i + b*l.outputs;
                softmax(l.output + index + 5, l.classes, 1, l.output + index + 5);
            }
        }
    }
/*
    // output of region layer
    for (int x = 0; x < 100; x++)
    {
        printf("l.output[%d]:%.12f;\n",x,l.output[x]);
    }
*/

#endif
    if(!state.train)
    {
        //printf("return here???\n");
        return;
    }
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        if(l.softmax_tree){
            int onlyclass = 0;
            for(t = 0; t < 30; ++t){
                box truth = float_to_box(state.truth + t*5 + b*l.truths);
                if(!truth.x) break;
                int class_s = state.truth[t*5 + b*l.truths + 4];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int index = size*n + b*l.outputs + 5;
                        float scale =  l.output[index-1];
                        l.delta[index - 1] = l.noobject_scale * ((0 - l.output[index - 1]) * logistic_gradient(l.output[index - 1]));
                        float p = scale*get_hierarchy_probability(l.output + index, l.softmax_tree, class_s);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int index = size*maxi + b*l.outputs + 5;
                    delta_region_class(l.output, l.delta, index, class_s, l.classes, l.softmax_tree, l.class_scale, &avg_cat);
                    if(l.output[index - 1] < .3) l.delta[index - 1] = l.object_scale * ((.3 - l.output[index - 1]) * logistic_gradient(l.output[index - 1]));
                    else  l.delta[index - 1] = 0;
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if(onlyclass) continue;
        }
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
                    box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                    float best_iou = 0;
                    for(t = 0; t < 30; ++t){
                        box truth = float_to_box(state.truth + t*5 + b*l.truths);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    avg_anyobj += l.output[index + 4];
                    l.delta[index + 4] = l.noobject_scale * ((0 - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
                    if (best_iou > l.thresh) {
                        l.delta[index + 4] = 0;
                    }

                    if(*(state.net.seen) < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n]/l.w;
                        truth.h = l.biases[2*n+1]/l.h;
                        delta_region_box(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, .01);
                    }
                }
            }
        }
        for(t = 0; t < 30; ++t){
            box truth = float_to_box(state.truth + t*5 + b*l.truths);

            if(!truth.x) break;
            float best_iou = 0;
            int best_index = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            //printf("index %d %d\n",i, j);
            for(n = 0; n < l.n; ++n){
                int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
                box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                if(l.bias_match){
                    pred.w = l.biases[2*n]/l.w;
                    pred.h = l.biases[2*n+1]/l.h;
                }
                //printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_index = index;
                    best_iou = iou;
                    best_n = n;
                }
            }
            //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);

            float iou = delta_region_box(truth, l.output, l.biases, best_n, best_index, i, j, l.w, l.h, l.delta, l.coord_scale);
            if(iou > .5) recall += 1;
            avg_iou += iou;

            //l.delta[best_index + 4] = iou - l.output[best_index + 4];
            avg_obj += l.output[best_index + 4];
            l.delta[best_index + 4] = l.object_scale * (1 - l.output[best_index + 4]) * logistic_gradient(l.output[best_index + 4]);
            if (l.rescore) {
                l.delta[best_index + 4] = l.object_scale * (iou - l.output[best_index + 4]) * logistic_gradient(l.output[best_index + 4]);
            }


            int class_s = state.truth[t*5 + b*l.truths + 4];
            if (l.map) class_s = l.map[class_s];
            delta_region_class(l.output, l.delta, best_index + 5, class_s, l.classes, l.softmax_tree, l.class_scale, &avg_cat);
            ++count;
            ++class_count;
        }
    }
    //printf("\n");
#ifndef GPU
    flatten(l.delta, l.w*l.h, size*l.n, l.batch, 0);
#endif
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

// extra region classes
void delta_region_class(float *output, float *delta, int index, int class_s, int classes, tree *hier, float scale, float *avg_cat)
{
    if(hier)
    {
        float pred = 1;
        while(class_s >= 0)
        {
            pred *= output[index + class_s];
            int g = hier->group[class_s];
            int offset = hier->group_offset[g];
            for(int i = 0; i < hier->group_size[g]; i++)
            {
                delta[index + offset + i] = scale * (0 - output[index + offset + i]);
            }
            delta[index + class_s] = scale * (1 - output[index + class_s]);

            class_s = hier->parent[class_s];
        }
        *avg_cat += pred;
    }
    else
    {
        for(int n = 0; n < classes; n++)
        {
            delta[index + n] = scale * (((n == class_s)?1 : 0) - output[index + n]);
            if(n == class_s)
            {
                *avg_cat += output[index + n];
            }
        }
    }
}

// extra region boxes
float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0] = scale * (tx - logistic_activate(x[index + 0])) * logistic_gradient(logistic_activate(x[index + 0]));
    delta[index + 1] = scale * (ty - logistic_activate(x[index + 1])) * logistic_gradient(logistic_activate(x[index + 1]));
    delta[index + 2] = scale * (tw - x[index + 2]);
    delta[index + 3] = scale * (th - x[index + 3]);

    return iou;
}
