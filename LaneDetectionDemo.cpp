/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com
Description: Lane Detector Demo.

*/

#include "LaneDetectionLib.h"
#include<opencv2\imgproc\imgproc.hpp>

#define CFG "config1.xml"
//#define CFG "config2.xml"
//#define CFG "config3.xml"
//#define CFG "config4.xml"
//#define CFG "config5.xml"

int main(void)
{
	string video_path;
	VehicleVision::params lparams;
	FileStorage fs(CFG, FileStorage::READ); 
	
	if (!fs.isOpened())
		{
			cout << "Could not open the configuration file. \n" << endl;
			exit(-1);
		}

		FileNode& fn = fs["Settings"];
		
		
		fn["videofile" ] >> video_path;
		
		fn["SCANNING_STEP" ] >> lparams.SCANNING_STEP;
		fn["LINE_THRESH" ] >> lparams.LINE_THRESH;
		fn["EDGE_THRESH" ] >> lparams.EDGE_THRESH;
		fn["MARGINS" ] >> lparams.MARGINS;
		fn["LANE_DIST_THRESH" ] >> lparams.LANE_DIST_THRESH;
		fn["HOUGH_LINE_THRESH" ] >> lparams.HOUGH_LINE_THRESH;
		fn["LINE_LENGTH" ] >> lparams.LINE_LENGTH;
		fn["LINE_GAP" ] >> lparams.LINE_GAP;
		fn["SLOPE_THRESH" ] >> lparams.SLOPE_THRESH;
		fn["INTERCEPT_THRESH" ] >> lparams.INTERCEPT_THRESH;
		fn["MAX_LANE_FRAMES" ] >> lparams.MAX_LANE_FRAMES;
		fn["GrayThresh" ] >> lparams.GrayThresh;
		fs.release();                                         

	VehicleVision::LaneDetector ld(video_path,2,0);
	ld.initParams(lparams);
	ld.detect();
	return 0;
}

