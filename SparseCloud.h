//
// Created by derue on 04/12/15.
//

#ifndef SFM_SPARSECLOUD_H
#define SFM_SPARSECLOUD_H

#include "Mesh.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "ViewPair.h"
#include "pba/pba.h"

using namespace std;

class SparseCloud {

private:
    cv::Mat m_F,m_E; // Fundamental and Essential
    cv::Mat m_K; // Calibration

    cv::Ptr<cv::Feature2D> m_feat2D; // keypoint type
    cv::DescriptorMatcher* m_matcher; // matcher type

    vector<View> m_vView;

    // bundler variable
    vector<CameraT> m_vCamera;
    vector<Point3D> point3D_data;     //3D point(iput/output)
    vector<Point2D> point2D_data;   //measurment/projection vector
    vector<int> camidx, pt3Didx;  //index of camera/point for each projection


    ParallelBA pba;





public:
    vector<Mesh::Point> m_cloud3D; // reconstructed 3D points

    SparseCloud();
    ~SparseCloud();

    void initialize(cv::Mat& v1, cv::Mat& v2);
    void addView(cv::Mat& vnew);
    void bundleAdjust();

    void reconstruct3DPt(vector<KpCustom>& kp1, vector<KpCustom>& kp2,vector<cv::DMatch>& match, cv::Mat& P1, cv::Mat& P2, vector<Mesh::Point>& outPt3D);
    void generateCloud();




};


#endif //SFM_SPARSECLOUD_H
