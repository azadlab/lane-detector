/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com
Description: Lane Detector Class Object.

*/


#include "LaneDetectionLib.h"
#include<opencv2\imgproc\imgproc.hpp>
#include <math.h>
#include "utils.h"

#undef MIN
#undef MAX
#define MAX(a,b) ((a)<(b)?(b):(a))
#define MIN(a,b) ((a)>(b)?(b):(a))

namespace VehicleVision
{

	struct Lane {
	Lane(){}
	Lane(CvPoint x1, CvPoint x2, float ang, float m, float c): p1(x1),p2(x2),theta(ang),
		numVotes(0),isSeen(false),isDetected(false),slope(m),intercept(c) { }

	CvPoint p1, p2;
	int numVotes;
	bool isSeen, isDetected;
	float theta, slope, intercept;
};

struct Status {
	Status():clean(true),lost(0){}
	ExpMovingAverage slope, intercept;
	bool clean;
	int lost;
};
Status rightLane, leftLane;


	void LaneDetector::initParams()
	{

		lparams.SCANNING_STEP=5;
		lparams.LINE_THRESH=10;
		lparams.EDGE_THRESH=250;
		lparams.MARGINS=10;
		lparams.LANE_DIST_THRESH=5;
		lparams.HOUGH_LINE_THRESH=50;
		lparams.LINE_LENGTH=50;
		lparams.LINE_GAP=150;
		lparams.SLOPE_THRESH=0.2f;
		lparams.INTERCEPT_THRESH=20;
		lparams.MAX_LANE_FRAMES=30;
		lparams.colorMin=Scalar(100,100,100);
		lparams.colorMax=Scalar(255,255,255);
		lparams.GrayThresh = 230;
	}
	
	void LaneDetector::initParams(params lparams)
	{
		this->lparams = lparams;
	}
	LaneDetector::LaneDetector(std::string video_path,int filterType,bool doEnhancement,bool UseCamera,bool WriteOutput,string out_path)
	{
		
		initParams();
		if(UseCamera)
		{
			input = VideoCapture(0);
		}
		else
		{
			this->input_video_path = video_path;
			this->lparams = lparams;
			input = VideoCapture(input_video_path); 
		}

		if (!input.isOpened()) {
			fprintf(stderr, "Error: Can't open video\n");
			return;
		}
		this->filterType = filterType;
		this->doEnhancement = doEnhancement;
		cv::Size vid_size;
		vid_size.height = (int) input.get(CV_CAP_PROP_FRAME_HEIGHT);
		vid_size.width = (int) input.get(CV_CAP_PROP_FRAME_WIDTH);
		float fps = (int) input.get(CV_CAP_PROP_FPS);
		
		if(WriteOutput)
		{
			this->output_video_path = out_path;
			output = VideoWriter(output_video_path,-1,fps,vid_size);
			if(!output.isOpened())
			{
			fprintf(stderr, "Error: Can't write output video\n");
			return;
			}
		}

	}


void searchLanePixelInRow(IplImage *img, int xBegin, int xEnd, int y, std::vector<int>& intensities,params lparams)
{
    
	const int imgRow = y * img->width * img->nChannels;
	unsigned char* ptr = (unsigned char*)img->imageData;

    int pix_step = (xEnd < xBegin) ? -1: 1;
    int pix_range = (xEnd > xBegin) ? xEnd-xBegin+1 : xBegin-xEnd+1;

    for(int i = xBegin; pix_range>0; i += pix_step, pix_range--)
    {
		if(ptr[imgRow + i] <= lparams.EDGE_THRESH) 
			continue; 

        int Idx = i + pix_step;

		while(pix_range > 0 && ptr[imgRow+Idx] > lparams.EDGE_THRESH){
            Idx += pix_step;
            pix_range--;
        }

		if(ptr[imgRow+Idx] <= lparams.EDGE_THRESH) {
            intensities.push_back(i);
        }

        i = Idx; 
    }
}


void validateLane(std::vector<Lane> lanes, IplImage* edgeImg, bool isRightLane,params lparams) {

	Status* side = isRightLane ? &rightLane : &leftLane;

	int width = edgeImg->width;
	int height = edgeImg->height;
	const int ystart = 0;
	const int yend = height-1;
	const int xend = isRightLane ? (width-lparams.MARGINS) : lparams.MARGINS;
	int midX = width/2;
	int midY = edgeImg->height/2;
	unsigned char* ptr = (unsigned char*)edgeImg->imageData;

	int* numVotes = new int[lanes.size()];
	for(int i=0; i<lanes.size(); i++) 
		numVotes[i++] = 0;

	for(int y=yend; y>=ystart; y-=lparams.SCANNING_STEP) {
		std::vector<int> pix_response;
		searchLanePixelInRow(edgeImg, midX, xend, y, pix_response,lparams);

		if (pix_response.size() > 0) {
			int horiz_response = pix_response[0]; 

			float min_dist = 1e10;
			float minX = 1e10;
			int match = -1;
			for (int j=0; j<lanes.size(); j++) {
				
				float dist = dist2line(cvPoint2D32f(lanes[j].p1.x, lanes[j].p1.y), 
									cvPoint2D32f(lanes[j].p2.x, lanes[j].p2.y), 
									cvPoint2D32f(horiz_response, y));

				
				int horizLine = (y - lanes[j].intercept) / lanes[j].slope;
				int mid_dist = abs(midX - horizLine); 

				if (match == -1 || (dist <= min_dist && mid_dist < minX)) {
					min_dist = dist;
					match = j;
					minX = mid_dist;
					break;
				}
			}

			if (match != -1) {
				numVotes[match] += 1;
			}
		}
	}

	int bestMatch = -1;
	int mini = 1e10;
	for (int i=0; i<lanes.size(); i++) {
		int horizLine = (midY - lanes[i].intercept) / lanes[i].slope;
		int dist = abs(midX - horizLine); 

		if (bestMatch == -1 || (numVotes[i] > numVotes[bestMatch] && dist < mini)) {
			bestMatch = i;
			mini = dist;
		}
	}

	if (bestMatch != -1) {
		Lane* best = &lanes[bestMatch];
		float slope_diff = fabs(best->slope - side->slope.get());
		float intercept_diff = fabs(best->slope - side->intercept.get());

		bool isvalidUpdate = (slope_diff <= lparams.SLOPE_THRESH && intercept_diff <= lparams.INTERCEPT_THRESH) || side->clean;
		
		if (isvalidUpdate) {
			side->slope.add(best->slope);
			side->intercept.add(best->intercept);
			side->clean = false;
			side->lost = 0;
		} else {

			side->lost++;
			if (side->lost >= lparams.MAX_LANE_FRAMES && !side->clean) {
				side->clean = true;
			}
		}

	} else {

		side->lost++;
		if (side->lost >= lparams.MAX_LANE_FRAMES && !side->clean) {
			side->clean = true;
			side->slope.clear();
			side->intercept.clear();
		}
	}

	delete[] numVotes;
}

	void findLanes(Vector<Vec4i> lines,Mat edges,Mat frame,params lparams)
	{

		std::vector<Lane> left, right,mid;

		for(int i = 0; i < lines.size(); i++ )
		{
			
			Vector<CvPoint> line;
			line.push_back(cvPoint(lines[i][0],lines[i][1]));
			line.push_back(cvPoint(lines[i][2],lines[i][3]));
			int dx = line[1].x - line[0].x;
			int dy = line[1].y - line[0].y;
			float angle = atan2f(dy, dx) * 180/CV_PI;

			if (fabs(angle) <= lparams.LINE_THRESH) { 
				continue;
			}

		
			dx = (dx == 0) ? 1 : dx; 
			float slope = dy/(float)dx;
			float intercept = line[0].y - slope*line[0].x;

			int midX = (line[0].x + line[1].x) / 2;
			int midY = (line[0].y + line[1].y) / 2;
		
			float d1 = fabs((float)edges.cols/4-midX);
			float d2 = fabs((float)3*edges.cols/4-midX);
			float d3 = fabs((float)5*edges.cols/16-midX);

			if(d3<d1 && d3<d2)
				mid.push_back(Lane(line[0],line[1],angle,slope,intercept));
			else
			if (d1 < d2) {
				left.push_back(Lane(line[0], line[1], angle, slope,intercept));
			} else if (d1 > d2) {
				right.push_back(Lane(line[0], line[1], angle, slope, intercept));
			}
		}

		IplImage* ipl_edges = cvCloneImage(&(IplImage)edges);
		
		validateLane(left, ipl_edges, false,lparams);
		validateLane(right, ipl_edges, true,lparams);


	}

	void EqualizeImage(Mat img,Mat &img_hist_equalized,bool isColor)
	{
	    if(isColor)
	   {
		   vector<Mat> channels; 
		   cvtColor(img, img_hist_equalized, CV_BGR2YCrCb); 
		   split(img_hist_equalized,channels); 
		   equalizeHist(channels[0], channels[0]); 
		   merge(channels,img_hist_equalized); 
		   cvtColor(img_hist_equalized, img_hist_equalized, CV_YCrCb2BGR); 
		   
	   }
	   else
	      equalizeHist(img, img_hist_equalized); 

	   
	}

	void LaneDetector::detect()
	{
		
		
		Mat frame;

		Size frame_size = Size(input.get(CV_CAP_PROP_FRAME_WIDTH), 3*input.get(CV_CAP_PROP_FRAME_HEIGHT)/4);
		Mat tmp_frame = Mat(frame_size,CV_8UC3);
		Mat edges = Mat(frame_size, CV_8UC1);
		float fps = (int) input.get(CV_CAP_PROP_FPS);
		int key_pressed = 0;
		while(key_pressed != 27) {


		input>>frame;
//		frameNo++;

		if(frame.empty())
		{
			return;
		}

		if(key_pressed==32)
			cvWaitKey(0);

		frame(Rect(0,frame_size.height/3,frame_size.width,frame_size.height)).copyTo(tmp_frame);
		Mat hsv_frame;
		Mat enh_frame(tmp_frame);
		switch(this->filterType)
		{
		case 0:  //RGB
			if(doEnhancement)
				EqualizeImage(tmp_frame,enh_frame,1);
			inRange(enh_frame,lparams.colorMin,lparams.colorMax,edges);
			break;
		case 1: //HSV
			if(doEnhancement)
				EqualizeImage(tmp_frame,enh_frame,1);
			hsv_frame = Mat(frame_size,CV_8UC3);
			cvtColor(enh_frame, hsv_frame, CV_BGR2HSV); 
			inRange(hsv_frame,lparams.colorMin,lparams.colorMax,edges);
			break;
		case 2: //GrayThresh
			cvtColor(tmp_frame,tmp_frame,CV_BGR2GRAY);
			enh_frame = tmp_frame;
			if(doEnhancement)
				EqualizeImage(tmp_frame,enh_frame,0);
			
			edges = enh_frame > lparams.GrayThresh;
		//imshow("Edges",edges);
			//cvWaitKey(0);	
			break;
		}
			
	//GaussianBlur( edges, edges, Size( 5, 5 ), 0, 0 );
	//Canny(edges, edges, 1, 100);

		double rho = 1;
		double theta = CV_PI/180;
		vector<Vec4i> lines;
		HoughLinesP(edges, lines, rho, theta, lparams.HOUGH_LINE_THRESH, lparams.LINE_LENGTH, lparams.LINE_GAP );
		if(lines.size()>0)
		{
			findLanes(lines, edges, tmp_frame,lparams);
			int br = frame_size.height/6;
			int xR = (br-rightLane.intercept.get())/rightLane.slope.get();
			int xL = (br-leftLane.intercept.get())/leftLane.slope.get();
		
		if(fabs(rightLane.slope.get())>0 && fabs(rightLane.intercept.get())>0)
			line(frame, cvPoint(xR, frame_size.height/3+br), cvPoint(frame.cols, frame.rows/2+(rightLane.slope.get() * frame.cols + rightLane.intercept.get())), CV_RGB(255, 0, 0), 4);
		if(fabs(leftLane.slope.get())>0 && fabs(leftLane.intercept.get())>0)
			line(frame, cvPoint(xL, frame_size.height/3+br), cvPoint(0, frame.rows/2+leftLane.intercept.get()), CV_RGB(255, 0, 0), 4);
		
			// show middle line
			float xInterL1 = (frame_size.height-leftLane.intercept.get())/leftLane.slope.get();
			float xInterR1 = (frame_size.height-rightLane.intercept.get())/rightLane.slope.get();
			float midInter1 = (xInterL1+xInterR1)/2;

		if(fabs(rightLane.slope.get())>0 && fabs(rightLane.intercept.get())>0 && fabs(leftLane.slope.get())>0 && fabs(leftLane.intercept.get())>0)
			line(frame, cvPoint((xR+xL)/2,frame.rows/2+br), cvPoint(midInter1,frame.rows), CV_RGB(255,255, 0), 4);
		}
		else
		{
			rightLane.clean=true;
			leftLane.clean=true;
		}
		imshow("Lanes", frame);
		imshow("Edges",edges);
		if(!output_video_path.empty())
			output.write(frame);
		
		key_pressed = cvWaitKey(1000/fps);

		}
	}
	
}
