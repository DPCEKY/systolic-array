//========================================================================
// Parser
//========================================================================
// @brief: parse and store configs

#ifndef SRC_PARSER_CPP_
#define SRC_PARSER_CPP_

#include "parser.h"

// parse amxpooling layer
maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", (size-1)/2);
    //printf("\nstride: %d; size: %d; padding: %d;\n",stride,size,padding);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    //printf("\nh: %d; w: %d; c: %d; batch: %d;\n",h,w,c,batch);
    if(!(h && w && c))
    {
        error("Layer before maxpool layer must output image.");
    }

    maxpool_layer layer = make_maxpool_layer(batch,h,w,c,size,stride,padding);
    return layer;
}

// copy matrix
void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = (float *)calloc(rows*cols, sizeof(float));
    //
    for(int x = 0; x < rows; x++)
    {
        for(int y = 0; y < cols; y++)
        {
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

// parse region layers
layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.max_boxes = option_find_int_quiet(options, "max",30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.rescore = option_find_int_quiet(options, "rescore",0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_int_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match",0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file)
    {
        l.softmax_tree = read_tree(tree_file);
    }
    char *map_file = option_find_str(options, "map", 0);
    if (map_file)
    {
        l.map = read_map(map_file);
    }

    char *a = option_find_str(options, "anchors", 0);
    if(a)
    {
        int len = strlen(a);
        int n = 1;
        //
        for(int i = 0; i < len; i++){
            if (a[i] == ',')
            {
                n++;
            }
        }
        for(int i = 0; i < n; i++){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

// parse convolutional layer
convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c))
    {
        error("Layer before convolutional layer must output image.");
    }
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,size,stride,padding,activation, batch_normalize, binary, xnor, params.net.adam);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);
    if(params.net.adam)
    {
        layer.B1 = params.net.B1;
        layer.B2 = params.net.B2;
        layer.eps = params.net.eps;
    }

    return layer;
}

// get string type
LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[region]")==0) return REGION;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[reorg]")==0) return REORG;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[route]")==0) return ROUTE;
    return BLANK;
}

// get larning rate policy
learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random")==0) return RANDOM;
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);

    return CONSTANT;
}

// config parser
void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;

    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam)
    {
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .000001);
    }

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

    if(!net->inputs && !(net->h && net->w && net->c))
    {
        error("No input parameters supplied");
    }

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    if(net->policy == STEP)
    {
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    }
    else if (net->policy == STEPS)
    {
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if(!l || !p)
        {
            error("STEPS policy must have steps and scales in cfg file");
        }

        int len = strlen(l);
        int n = 1;
        //
        for(int i = 0; i < len; i++)
        {
            if (l[i] == ',')
            {
                n++;
            }
        }
        int *steps = (int *) calloc(n, sizeof(int));
        float *scales = (float *)calloc(n, sizeof(float));
        for(int i = 0; i < n; i++)
        {
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    }
    else if (net->policy == EXP)
    {
        net->gamma = option_find_float(options, "gamma", 1);
    }
    else if (net->policy == SIG)
    {
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    }
    else if (net->policy == POLY || net->policy == RANDOM)
    {
        net->power = option_find_float(options, "power", 1);
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

// free section
void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

// tiny-yolo.cfg
network parse_network_cfg(char *filename)
{
    // read cfg lines into a list
    // list:    size, *front(start 'node'), *back(end 'node')
    // node:    val(*'section'), *next, *prev
    // section: *type, *option('list')
    // list:    size, *front(start 'node'), *back(end 'node')
    // node:    val(*'kvp'), *next, *prev
    // kvp:     *key, *val, used?(init 0 - unused)
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n)
    {
        error("Config file has no sections");
    }
    // network within net(general setup) + 16 layers(9 conv + 6 maxpool + 1 region) in tiny-yolo
    network net = make_network(sections->size - 1);
    size_params params; // why define this?

    // traverse the sections in the top list
    section *s = (section *)n->val;
    list *options = s->options;
    //if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, &net);

    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    params.batch = net.batch;
    params.time_steps = net.time_steps;
    params.net = net;

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while(n)
    {
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;

        layer l;
        init_layer(l);

        //printf("\n(1).workspace_size:%d\n;",l.workspace_size);

        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL)
        {
            l = parse_convolutional(options, params);
        }
        else if(lt == MAXPOOL)
        {
            l = parse_maxpool(options, params);
        }
        else if(lt == REGION)
        {
           l = parse_region(options, params);
        }

        //printf("\n(2).workspace_size:%d\n;",l.workspace_size);
        /*}else if(lt == LOCAL){
            l = parse_local(options, params);
        }else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }else if(lt == RNN){
            l = parse_rnn(options, params);
        }else if(lt == GRU){
            l = parse_gru(options, params);
        }else if(lt == CRNN){
            l = parse_crnn(options, params);
        }else if(lt == CONNECTED){
            l = parse_connected(options, params);
        }else if(lt == CROP){
            l = parse_crop(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
        }else if(lt == REGION){
            l = parse_region(options, params);
        }else if(lt == DETECTION){
            l = parse_detection(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net.hierarchy = l.softmax_tree;
        }else if(lt == NORMALIZATION){
            l = parse_normalization(options, params);
        }else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }else if(lt == REORG){
            l = parse_reorg(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == ROUTE){
            l = parse_route(options, params, net);
        }else if(lt == SHORTCUT){
            l = parse_shortcut(options, params, net);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params);
            l.output = net.layers[count-1].output;
            l.delta = net.layers[count-1].delta;
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }*/
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        // check unused kvps
        option_unused(options);
        net.layers[count] = l;
        //printf("n:%d; l.workspace_size: %d;\n",count,l.workspace_size);
        if (l.workspace_size > workspace_size)
        {
            workspace_size = l.workspace_size;
        }
        free_section(s);
        n = n->next;
        count++;
        if(n)
        {
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    net.outputs = get_network_output_size(net);  // output size
    //printf("net.outputs:%d;\n",net.outputs);
    net.output = get_network_output(net);        // output value???
    if(workspace_size)
    {
        //printf("workspace_size:%ld;\n", workspace_size);
        //??????????
        //net.workspace = (float *)calloc(1, workspace_size);
        net.workspace = (float *)calloc(workspace_size,sizeof(float));
        //net.workspace = (float *)sds_alloc(workspace_size * sizeof(float));
    }
    return net;
}

// read configs
list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    //if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *sections = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0)
    {
        nu++;
        strip(line);
        switch(line[0])
        {
            case '[':
                current = (section *)malloc(sizeof(section));
                list_insert(sections, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return sections;
}

// laod weights
void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary)
    {
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l.n*l.c*l.size*l.size;
    fread(l.biases, sizeof(float), l.n, fp);
    //printf("num:%d; l.n:%d;\n",num,l.n);
    if (l.batch_normalize && (!l.dontloadscales))
    {
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
        //printf("enter here1\n");
        if(0)
        {
            //
            for(int i = 0; i < l.n; i++)
            {
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(int i = 0; i < l.n; i++)
            {
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
        if(0)
        {
            fill_cpu(l.n, 0, l.rolling_mean, 1);
            fill_cpu(l.n, 0, l.rolling_variance, 1);
        }
    }
    fread(l.weights, sizeof(float), num, fp);
    //l.adam = 0;
    if(l.adam)
    {
        fread(l.m, sizeof(float), num, fp);
        fread(l.v, sizeof(float), num, fp);
        //printf("enter here2\n");
    }
    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped)
    {
        transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
        //printf("enter here3\n");
    }
/*
    // print weights
    for(int j = 300; j < 400; j++)
    {
        printf("l.weights[%d]:%.12f;\n",j,l.weights[j]);
    }
*/
    //if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
}

// batchnorm weights
void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);

}

// connected weights
void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose)
    {
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales))
    {
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }

}

// load weights top function
void load_weights_upto(network *net, char *filename, int cutoff)
{
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    //if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    fread(net->seen, sizeof(int), 1, fp);
    int transpose = (major > 1000) || (minor > 1000);
    //
    for(int i = 0; i < net->n && i < cutoff; i++){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL)
        {
            //printf("layer %d: CONVOLUTIONAl;\n",i);
            load_convolutional_weights(l, fp);
        }
        if(l.type == CONNECTED)
        {
            load_connected_weights(l, fp, transpose);
        }
        if(l.type == BATCHNORM)
        {
            load_batchnorm_weights(l, fp);
        }
        if(l.type == CRNN)
        {
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == RNN)
        {
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if(l.type == GRU)
        {
            load_connected_weights(*(l.input_z_layer), fp, transpose);
            load_connected_weights(*(l.input_r_layer), fp, transpose);
            load_connected_weights(*(l.input_h_layer), fp, transpose);
            load_connected_weights(*(l.state_z_layer), fp, transpose);
            load_connected_weights(*(l.state_r_layer), fp, transpose);
            load_connected_weights(*(l.state_h_layer), fp, transpose);
        }
        if(l.type == LOCAL)
        {
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), size, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

// load weights top function
void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, net->n);
}

#endif /* SRC_PARSER_CPP_ */
