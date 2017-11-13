//========================================================================
// yolo_detector header file
//========================================================================
// @brief: Application for detecting objects

#include "yolo_detector.h"

/*
    // transfer value
    // argv[0] - datacfg     : cfg/coco.data
    // argv[1] - cfgfile     : cfg/tiny-yolo.cfg
    // argv[2] - weightfile  : tiny-yolo.weights
    // argv[3] - filename    : data/dog.jpg
    // argv[4] - thresh      : 0.24
    // argv[5] - hier_thresh : 0.5
*/

void detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh)
{
    // load datacfg
    printf("datacfg:%s\n",datacfg);
    list *options   = read_data_cfg(datacfg);

/*
    // print options
        node *pnode = options->front;
        kvp  *dis;
        int counter = 0;
        printf("Option size:%d\n",options->size);
        while (pnode)
        {
            dis = (kvp *)pnode->val;
            printf("NO. %d: (1)key: %s; (2)value: %s;\n",counter,(char*)dis->key,(char*)dis->val);
            counter++;
            pnode = pnode->next;
        }
*/

    char *name_list = option_find_str(options, "names", "data/names.list");
    //printf("name_list:%s\n",name_list);

    // name_list: data/coco.names
    char **names    = get_labels(name_list);

/*
    int size_names = sizeof(names);
    for (int i = 0; i < 80; i++)
    {
        printf("names NO. %d: %s;\n",i,names[i]);
    }
*/
    // read labels
    image **alphabet = load_alphabet();

    // load cfgfile
    network net = parse_network_cfg(cfgfile);

    // load weitht file
    if (weightfile)
    {
        load_weights(&net,weightfile);
    }

    // setup net.batch = 1
    set_batch_network(&net, 1);
    char buffer[255];
    char *input = buffer;
    float nms = 0.4;

    // start timer
    Timer timer("yolo_detector");

    while (1)
    {
        // copy image name
        if (filename)
        {
            strncpy (input,filename,256);
        }
        else
        {
            printf("Please enter image path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input)
            {
                return;
            }
            strtok(input,"\n");
        }
        //
        image im      = load_image_color(input,0,0);
        image sized   = resize_image(im, net.w, net.h);
        //printf("sized.h:%d; sized.w:%d; sized.c:%d;\n",sized.h,sized.w,sized.c);
/*
        // print input image
        for (int m = 0; m < 100; m++)
        {
            printf("im.data[%d]:%.12f;\n",m,im.data[m]);
        }
*/
        // region layer
        layer l       = net.layers[net.n-1];
        //
        box *boxes    = (box *)calloc(l.w * l.h * l.n, sizeof(box));
        float **probs = (float **)calloc(l.w * l.h * l.n, sizeof(float *));
        for (int i = 0; i < l.w*l.h*l.n; i++)
        {
            probs[i] = (float *)calloc(l.classes+1, sizeof(float)); // ???
        }

        //
        float *X = sized.data;

        Timer timer("Total time");
        // start prediction
        printf("Start prediction...\n");
        timer.start();
        network_predict(net,X);
        timer.stop();
        printf("Prediction finishes!\n");

        // draw region boxes
        printf("Getting region boxes...\n");
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
/*
        // verify **probs
        float sum = 0.0;
        printf("max_i:%d; max_j:%d;\n",l.w * l.h * l.n,l.classes+1);
        for (int i = 0; i < l.w * l.h * l.n; i++)
        {
            for (int j = 0; j < l.classes+1; j++)
            {
                if (probs[i][j] != 0)
                {
                    printf("probs[%d][%d]:%.12f;\n",i,j,probs[i][j]);
                }
                sum += probs[i][j];
            }
        }
        printf("sum:%.12f;\n",sum);
*/
        //
        if (l.softmax_tree && nms)
        {
            //printf("Enter 111111\n");
            do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        }
        else if (nms)
        {
            //printf("Enter 222222\n");
            do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        }
/*
        // verify **probs
                float sum = 0.0;
                printf("max_i:%d; max_j:%d;\n",l.w * l.h * l.n,l.classes+1);
                for (int i = 0; i < l.w * l.h * l.n; i++)
                {
                    for (int j = 0; j < l.classes+1; j++)
                    {
                        if (probs[i][j] != 0)
                        {
                            printf("probs[%d][%d]:%.12f;\n",i,j,probs[i][j]);
                        }
                        sum += probs[i][j];
                    }
                }
                printf("sum:%.12f;\n",sum);
*/


        printf("Start draw predictions...\n");
        draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
        printf("Save & print images...\n");
        save_image(im, "predictions");
        show_image(im, "predictions");
        //printf("ch0\n");

        // free memory
        free_image(im);
        free_image(sized);
        //printf("ch1\n");
        free_ptrs((void **)probs, l.w*l.h*l.n);
        //printf("ch2\n");
        free(boxes);
        //printf("ch3\n");
        // where did we modify the value filename to jump out of the while loop?????
        if (filename)
        {
            break;
        }
    }
    printf("Exit program.\n");
}


