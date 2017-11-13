//========================================================================
// testbench.cpp
//========================================================================
// @brief: testbench for yolo detector

#include <stdio.h>
#include <stdlib.h>

#include "yolo_detector.h"

int main (int argc, char **argv)
{
/*
	// transfer value
	// argv[0]: yolo_detector_test
	// argv[1]: detect
	// argv[2]: cfg/yolo.cfg
	// argv[3]: yolo.weights
	// argv[4]: data/dog.jpg
*/
	if (strcmp(argv[1],"detect") == 0)
	{
		float thresh = 0.24;
		char  *filename = (argc > 4) ? argv[4] : 0;
		detector("cfg/coco.data", argv[2], argv[3], filename, thresh, 0.5);
	}
	else
	{
		printf("Invalid input, program stop...");
	}
	return 0;
}
