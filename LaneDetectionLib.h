/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com
Description: Lane Detector Class Object from a video sequence.

*/


#pragma once
#include <SDKDDKVer.h>

#include <stdio.h>
#include <conio.h>
#include <tchar.h>


#ifdef LANEDETECTIONLIB_EXPORTS
#define LANEDETECTIONLIB_API __declspec(dllexport)
#else
#define LANEDETECTIONLIB_API __declspec(dllimport)
#endif

#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace VehicleVision
{

	typedef struct params
	{
	
		int SCANNING_STEP; // step to traverse image for finding lane pixel
		int LINE_THRESH; // threshold to reject a line (in degrees)
		int EDGE_THRESH;		  // threshold to declare a pixel lane pixel'
		int MARGINS;			  // margins for skipping edges of an image
		int LANE_DIST_THRESH;	  // maximum distance of a candidate pixel from Lane pixel

		int HOUGH_LINE_THRESH;		// line detection thresh
		int LINE_LENGTH;	// keep only lines larger than this threshold
		int LINE_GAP;   // threshold to join lines

		int SLOPE_THRESH;   // 'm' in y=mx+c (how much the slope of a line should be allowed to deviate in order to still retain the detection in previous image)
		int INTERCEPT_THRESH;  // 'c' in y=mx+c (how much the intercept of a line should be allowed to deviate in order to still retain the detection in previous image)
		int MAX_LANE_FRAMES;	// How many frames before a detected lane is declared lost.

		Scalar colorMin;   //Minimum color value to specify for color filtering (RGB/HSV)
		Scalar colorMax;   //Maximum color value to specigy for color filtering.
		
		int GrayThresh;
		//params():SCANNING_STEP(5),LINE_THRESH(10),EDGE_THRESH(250),MARGINS(10),LANE_DIST_THRESH(5),HOUGH_LINE_THRESH(50),LINE_LENGTH(50),LINE_GAP(150),SLOPE_THRESH(0.2f),INTERCEPT_THRESH(20),MAX_LANE_FRAMES(30),colorMin(100,100,100),colorMax(255,255,255) {};
	}params;

	class  LaneDetector
	{
	private:
		std::string input_video_path;
		std::string output_video_path;
		cv::VideoCapture input;
		cv::VideoWriter output;
		int filterType;   // 0: RGB 1:HSV 2:GrayThresh
		bool doEnhancement;
		params lparams;
	public:
		
		LANEDETECTIONLIB_API LaneDetector() {};
		void LANEDETECTIONLIB_API initParams();
		void LANEDETECTIONLIB_API initParams(params lparams);
		LANEDETECTIONLIB_API LaneDetector(std::string video_path,int filterType=0,bool doEnhancement=false,bool UseCamera=false,bool WriteOutput=false,string out_path="");
		void LANEDETECTIONLIB_API detect(void);
		
	};

}