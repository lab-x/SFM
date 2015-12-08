#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "View.h"
#include "funUtils.h"

using namespace std;
class ViewPair
{
private:
	View *m_view1, *m_view2;
	vector<vector<cv::DMatch>> m_vv_matchingKps;




public:
    cv::Mat m_FMatrix, m_EMatrix;
	vector<cv::DMatch> m_matchingKps;
	ViewPair(View* view1ptr, View* view2ptr);
	~ViewPair();

	void matchFeat(cv::DescriptorMatcher* matcherType);
	void displayMatch();

	void computeFMatrix();
	void autoCalib();
	void computeEMatrix();
	void extractCameraMatrix();


	void computePPp(cv::Mat& P, cv::Mat& Pp); // not needed anymore -> projective ambiguity
	//void triangulation(vector<cv::Point3f>& recons3D);//my triangulation
	void triangulation(cv::Mat& out);//opencv triangulation
	void setManualMatch(vector<cv::Point2f>& vkp1, vector<cv::Point2f>& vkp2);
};