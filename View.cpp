#include "View.h"

using namespace std;
using namespace cv;

View::View(Mat& image,Ptr<Feature2D> &feat2D)
{
	m_image = image;
	m_Rt = Mat(3, 4, CV_32F,Scalar(0));

	feat2D->detectAndCompute(image, Mat(),m_vkp, m_kpDescr);

    m_vkpCustom.resize(m_vkp.size());
    for(int i=0; i<m_vkpCustom.size(); i++){
        cv::Vec3b kpcolor = image.at<cv::Vec3b>((int)m_vkp[i].pt.y,(int)m_vkp[i].pt.x);
        m_vkpCustom[i].m_kpColor.b = kpcolor[0];
        m_vkpCustom[i].m_kpColor.g = kpcolor[1];
        m_vkpCustom[i].m_kpColor.r = kpcolor[2];

        m_vkpCustom[i].m_xy_hom.x = m_vkp[i].pt.x;
        m_vkpCustom[i].m_xy_hom.y = m_vkp[i].pt.y;
        m_vkpCustom[i].m_xy_hom.z = 1;
    }
}

View::~View()
{

}

void View::normalizeCalib()
{
    float finv = 1/m_KMatrix.at<float>(0,0);

    for(int i=0; i<m_vkpCustom.size(); i++){
        m_vkpCustom[i].m_xy_hom.x*=finv;
        m_vkpCustom[i].m_xy_hom.y*=finv;
    }

}

void View::normalizeIso() {
    float w = m_image.cols;
    float h = m_image.rows;
    float N = w*h;
    cv::Point2f center(w / 2.f, h / 2.f);
    float totNorm = 0;
    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++){
            cv::Point2f xy(x, y);
            totNorm += norm(center - xy);
        }
    }
    float s = totNorm / (sqrt(2)*N);


    //normalize kp coordinates
    for (auto& kp : m_vkpCustom){
        kp.m_xy_hom.x -= center.x;
        kp.m_xy_hom.y -= center.y;
        kp.m_xy_hom /= s;
        kp.m_xy_hom.z = 1;
    }

}

void View::cam2view() {
    m_KMatrix.at<float>(0,0) = camera->f;
    m_KMatrix.at<float>(1,1) = camera->f;
    m_KMatrix.at<float>(1,1) = 1;

    for(int i=0; i<3; i++){
        m_Rt.at<float>(i,3) = camera->t[i];
        for(int j=0; j<3; j++){
            m_Rt.at<float>(i,j)=camera->m[i][j];
        }
    }
}

void View::view2cam() {
    camera->f = m_KMatrix.at<float>(0,0);
    for(int i=0; i<3; i++){
        camera->t[i]=m_Rt.at<float>(i,3);
        for(int j=0; j<3; j++){
            camera->m[i][j]=m_Rt.at<float>(i,j);
        }
    }
}
