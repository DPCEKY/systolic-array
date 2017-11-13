//========================================================================
// Inits
//========================================================================
// @brief: initilization of struct

#include "inits.h"

// init layer ***pass by reference
void init_layer(layer &l)
{
    /*
    l.type = 0;
    l.activation = 0;
    l.cost_type = 0;
    */
    //
    l.forward  = 0;
    l.backward = 0;
    l.update   = 0;
    l.forward_gpu  = 0;
    l.backward_gpu = 0;
    l.update_gpu   = 0;

    //
    l.batch_normalize = 0;
    l.shorcut = 0;
    l.batch = 0;
    l.forced = 0;
    l.flipped = 0;
    l.inputs = 0;
    l.outputs = 0;
    l.truths = 0;
    l.h = 0;
    l.w = 0;
    l.c = 0;
    l.out_h = 0;
    l.out_w = 0;
    l. out_c = 0;
    l. n = 0;
    l. max_boxes = 0;
    l. groups = 0;
    l. size = 0;
    l. side = 0;
    l. stride = 0;
    l. reverse = 0;
    l. pad = 0;
    l. sqrt = 0;
    l. flip = 0;
    l. index = 0;
    l. binary = 0;
    l. xnor = 0;
    l. steps = 0;
    l. hidden = 0;
    l. dot = 0;
    l. angle = 0;
    l. jitter = 0;
    l. saturation = 0;
    l. exposure = 0;
    l. shift = 0;
    l. ratio = 0;
    l. softmax = 0;
    l. classes = 0;
    l. coords = 0;
    l. background = 0;
    l. rescore = 0;
    l. objectness = 0;
    l. does_cost = 0;
    l. joint = 0;
    l. noadjust = 0;
    l. reorg = 0;
    l. log = 0;
    // repeat in network ???
    l. adam = 0;
    l. B1 = 0;
    l. B2 = 0;
    l. eps = 0;
    l. t = 0;
    //
    l. alpha = 0;
    l. belta = 0;
    l. kappa = 0;
    //
    l. coord_scale = 0;
    l. object_scale = 0;
    l. noobject_scale = 0;
    l. class_scale = 0;
    l. bias_match = 0;
    l. random = 0;
    l. thresh = 0;
    l. classfix = 0;
    l. absolute = 0;
    //
    l. dontload = 0;
    l. dontloadscales = 0;
    //
    l. temperature = 0;
    l. probability = 0;
    l. scale = 0;
    //
    l.cweights = 0;
    l.indexes = 0;
    l.input_layers = 0;
    l.input_sizes = 0;
    l. map = 0;
    l.rand = 0;
    l.cost = 0;
    l.state = 0;
    l.prev_state = 0;
    l.forgot_state = 0;
    l.forgot_delta = 0;
    l.state_delta = 0;
    //
    l.concat = 0;
    l.concat_delta = 0;
    //
    l.binary_weights = 0;
    //
    l.biases = 0;
    l.bias_updates = 0;
    //
    l.scales = 0;
    l.scale_updates = 0;
    //
    l.weights = 0;
    l.weight_updates = 0;
    //
    l.col_image = 0;
    l.delta = 0;
    l.output = 0;
    l.squared = 0;
    l.norms = 0;
    //
    l.spatial_mean = 0;
    l.mean = 0;
    l.variance = 0;
    //
    l.mean_delta = 0;
    l.variance_delta = 0;
        //
    l.rolling_mean = 0;
    l.rolling_variance = 0;
        //
    l.x = 0;
    l. x_norm = 0;
    l.m = 0;
    l.v = 0;
    //
    l.z_cpu = 0;
    l.r_cpu = 0;
    l.h_cpu = 0;
    //
    l.binary_input = 0;
    //
    l.input_layer = 0;
    l.self_layer = 0;
    l.output_layer = 0;
    //
    l.input_gate_layer = 0;
    l.state_gate_layer = 0;
    l.input_save_layer = 0;
    l.state_save_layer = 0;
    l.input_state_layer = 0;
    l.state_state_layer = 0;
    //
    l.input_z_layer = 0;
    l.state_z_layer = 0;
    //
    l.input_r_layer = 0;
    l.state_r_layer = 0;
    //
    l.input_h_layer = 0;
    l.state_h_layer = 0;
    //
    l.softmax_tree = 0;
    //
    l.workspace_size = 0;
}


