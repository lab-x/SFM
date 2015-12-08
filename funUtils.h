#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>


using namespace std;


cv::Mat vec2crossMat(cv::Mat& vec);
cv::Mat solveAx0(cv::Mat& A);
cv::Mat createA_DLT(cv::Mat& P1, cv::Mat& P2, cv::Point2f pt1, cv::Point2f pt2);
cv::Mat Kruppa(cv::Mat& F);


void get4P(cv::Mat& E, vector<cv::Mat>& Pcan);
cv::Point2f Pt3ToPt2(cv::Point3f& p3);