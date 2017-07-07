#ifndef FRAME_H
#define FRAME_H

#include <string>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Eigen>

using namespace cv;
using namespace std;



class Frame
{
public:

	// dense scene info
	int frame_index;
	Mat rgb;
	Mat depth;
	Mat gray;
	// sparse feature info
	std::vector<KeyPoint> keypoints;
	std::vector<unsigned char> feature_tracked_flag;
	Mat descriptor;
	std::vector<float> depth_value;


	// pose information, represented using Lie algebra, based on MRPT
	std::vector<Eigen::Vector3d> local_points;
	// time stamp
	double time_stamp;
	int tracking_success;
	int blur_flag;
	int is_keyframe;

	Eigen::Vector3d pos;
	Eigen::Vector3d orientation;
	// queryIdx
	
	Frame()
	{
		frame_index = 0;
		keypoints.clear(); 
		descriptor.release(); 
		depth_value.clear();
		rgb.release();
		depth.release();
		local_points.clear();
		feature_tracked_flag.clear();
		tracking_success = 0;
		blur_flag = 0;
		is_keyframe = 0;
	}
};



#endif
