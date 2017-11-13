//========================================================================
// yolo_detector header file
//========================================================================
// @brief: function prototype definition

#ifndef SRC_YOLO_DETECTOR_H_
#define SRC_YOLO_DETECTOR_H_

#include <stdio.h>
#include <stdlib.h>

#include "timer.h"
#include "network.h"
#include "region_layer.h"
#include "utilities.h"
#include "parser.h"
#include "box.h"
#include "option_list.h"
#include "data.h"
#include "timer.h"

void detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh);

#endif /* SRC_YOLO_DETECTOR_H_ */
