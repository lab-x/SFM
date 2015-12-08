#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Mesh.h"
#include "pba/DataInterface.h"

using namespace std;

struct KpCustom{
	cv::KeyPoint* m_kp_ptr;
	cv::Point3f m_xy_hom; // homogeneous coordinates
	Mesh::Color m_kpColor;
	Mesh::Point* m_pt3D_ptr;
    int m_pt3Didx = -1; // id of reconstructed 3D point

	KpCustom():m_kp_ptr(nullptr),m_kpColor(),m_pt3D_ptr(nullptr),m_xy_hom(cv::Point3f(0,0,0)){}
};


class View
{
private:


	//vector<cv::Point2f> m_newp1m, m_newp2m;
public:
    vector<cv::Point2f> m_kpm;// coordinates of matching points
	cv::Mat m_image;
	vector<cv::KeyPoint> m_vkp;
	vector<KpCustom> m_vkpCustom;
	cv::Mat m_kpDescr;
    cv::Mat m_KMatrix;
	cv::Mat m_Rt;
    cv::Mat m_P; // P = K[R|t]

	CameraT* camera;



	View(cv::Mat& image, cv::Ptr<cv::Feature2D>& feat2D);
	~View();

    void normalizeIso();
	void normalizeCalib();

    void cam2view();
    void view2cam();



};