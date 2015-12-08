#include "ViewPair.h"

using namespace std;
using namespace cv;
ViewPair::ViewPair(View* view1ptr,View* view2ptr)
{
	m_view1 = view1ptr;
	m_view2 = view2ptr;
}

ViewPair::~ViewPair()
{
}

void ViewPair::matchFeat(DescriptorMatcher* matcher)
{

	float thr_ratio = 0.75;
	matcher->knnMatch(m_view1->m_kpDescr, m_view2->m_kpDescr, m_vv_matchingKps, 2);
	
	//David lowe ratio test
	for (int i = 0; i < m_vv_matchingKps.size(); i++)
	{
		float ratio = m_vv_matchingKps[i][0].distance / m_vv_matchingKps[i][1].distance;
		if (ratio < thr_ratio) m_matchingKps.push_back(m_vv_matchingKps[i][0]);
	}


}

void ViewPair::computeFMatrix()
{
    m_view1->normalizeIso();
    m_view2->normalizeIso();

    // fill 2d coordinates of matching points in the 2 views
    m_view1->m_kpm.resize(m_matchingKps.size());
    m_view2->m_kpm.resize(m_matchingKps.size());
    for (int i = 0; i < m_matchingKps.size(); i++)
    {
        m_view1->m_kpm[i] = Pt3ToPt2(m_view1->m_vkpCustom[m_matchingKps[i].queryIdx].m_xy_hom);
        m_view2->m_kpm[i] = Pt3ToPt2(m_view2->m_vkpCustom[m_matchingKps[i].trainIdx].m_xy_hom);
    }

	CV_Assert(m_view1->m_kpm.size() >= 8);
	m_FMatrix = findFundamentalMat(m_view1->m_kpm, m_view2->m_kpm, FM_RANSAC);
	m_FMatrix.convertTo(m_FMatrix, CV_32F);
	cout << "FMatrix " << m_FMatrix << endl;
	cout << "det F " << determinant(m_FMatrix) << endl;
	//correctMatches(m_FMatrix, m_p1m, m_p2m, m_newp1m, m_newp2m);

}




void ViewPair::autoCalib() {
    m_view1->m_KMatrix = Kruppa(m_FMatrix);
    m_view2->m_KMatrix = m_view1->m_KMatrix;
    cout << "K " << m_view1->m_KMatrix  << endl;
    // Normalize homogeneous coordinates , not needed if E matrix found from cv
    //m_view1->normalizeCalib();
    //m_view2->normalizeCalib();

}

void ViewPair::computeEMatrix() {
	/*m_EMatrix = m_KMatrix.t()*m_FMatrix*m_KMatrix;
	cout << "my E " << m_EMatrix << endl;*/

    m_EMatrix = cv::findEssentialMat(m_view1->m_kpm,m_view2->m_kpm,m_view1->m_KMatrix.at<float>(0,0),Point2d(0,0),RANSAC);
    cout<<"cv E"<<m_EMatrix<<endl;
    m_EMatrix.convertTo(m_EMatrix,CV_32F);

}

// find [R|t]
void ViewPair::extractCameraMatrix() {

    // find [R|t]1
    float *P_ptr = (float *) m_view1->m_Rt.data;
    for (int i = 0; i < 3; i++){
        P_ptr[i * 4 + i] = 1; //P1=[I|0]
    }

    // find P2 = [R|t]2 since from E

    vector<cv::Mat> Pcand(4);
    get4P(m_EMatrix, Pcand); //ok

    // only one point is needed to remove ambiguity but should be an inlier and give a point in front of both camera

    float f = m_view1->m_KMatrix.at<float>(0,0);
    int idxPoint = 0;
    bool notFound = 1;
    while(notFound) {
        CV_Assert(idxPoint<m_matchingKps.size());
        cv::Point3f p1 = m_view1->m_vkpCustom[m_matchingKps[idxPoint].queryIdx].m_xy_hom; //  p1 = K1^-1 x
        cv::Point3f p2 = m_view2->m_vkpCustom[m_matchingKps[idxPoint].trainIdx].m_xy_hom;



        for (int i = 0; i < 4; i++) {
            //cv::Mat A = createA_DLT(m_view1->m_Rt, Pcand[i], Point2f(p1.x, p1.y), Point2f(p2.x, p2.y));
            //cv::Mat Q = solveAx0(A); // column vector 4x1

            cv::Mat Q;
            vector<cv::Point2f> p1v(1,Pt3ToPt2(p1)/f), p2v(1,Pt3ToPt2(p2)/f);
            //p1v.push_back();
            //p2v.push_back(Pt3ToPt2(p2));

            cv::triangulatePoints(m_view1->m_Rt,Pcand[i],p1v,p2v,Q);

            CV_Assert(Q.isContinuous());
            float *Q_ptr = (float *) Q.data;

            float c1 = Q_ptr[2] * Q_ptr[3];
            cv::Mat PQ = Pcand[i] * Q; // columnn vector 4x1
            float c2 = PQ.at<float>(2, 0) * Q_ptr[3];
            if (c1 > 0 && c2 > 0) {
                m_view2->m_Rt = Pcand[i];
                notFound = false;
                break;
            }
            idxPoint++;
        }
    }
}


void ViewPair::triangulation(Mat& recons3D)//vector<Point3f>& recons3D)
{
	computePPp(m_view1->m_Rt, m_view2->m_Rt);
	//recons3D.resize(m_newp1m.size());
	/*for (int i = 0; i < recons3D.size(); i++)
	{
		Mat A = createA_DLT(m_view1->P,m_view2->P,m_newp1m[i],m_newp2m[i]);
		Mat pt3DM = solveAx0(A);
		float* pt3D_ptr = (float*)pt3DM.data;
		recons3D[i] = Point3f(pt3D_ptr[0], pt3D_ptr[1], pt3D_ptr[2]);
	}*/


	//triangulatePoints(m_view1->m_Rt, m_view2->m_Rt, m_newp1m, m_newp2m, recons3D);

}

void ViewPair::computePPp(Mat& P, Mat& Pp)
{
	float* P_ptr = (float*)P.data;
	for (int i = 0; i < 3; i++) P_ptr[i * 4 + i] = 1;

	// find ep
	Mat Ft;
	transpose(m_FMatrix, Ft);
	Mat ep = solveAx0(Ft);
	float* ep_ptr = (float*)ep.data;
	Mat ep_cross = vec2crossMat(ep);


	Mat Pptmp = ep_cross*m_FMatrix;


	float* Pptmp_ptr = (float*)Pptmp.data;

	float* Pp_ptr = (float*)Pp.data;
	for (int i = 0; i < 3; i++)
	{
		int i4 = 4 * i;
		int i3 = 3 * i;
		Pp_ptr[i4] = Pptmp_ptr[i3];
		Pp_ptr[i4+1] = Pptmp_ptr[i3+1];
		Pp_ptr[i4+2] = Pptmp_ptr[i3+2];
		Pp_ptr[i4 + 3] = ep_ptr[i];
	}

	cout << "Pp " << P << endl;

	Mat K, R, t;
	decomposeProjectionMatrix(Pp, K, R, t);

	cout << "Kp " <<K<< endl;

	decomposeProjectionMatrix(P, K, R, t);
	cout << "K " << K << endl;


}



void ViewPair::setManualMatch(vector<cv::Point2f>& vkp1, vector<cv::Point2f>& vkp2){
	CV_Assert(vkp1.size() == vkp2.size());
	m_view1->m_kpm.resize(vkp1.size());
    m_view2->m_kpm.resize(vkp2.size());
	for (int i = 0; i < vkp1.size(); i++){
        m_view1->m_kpm[i] = vkp1[i];
        m_view2->m_kpm[i] = vkp2[i];
	}
}

void ViewPair::displayMatch()
{
	Mat out;
	drawMatches(m_view1->m_image, m_view1->m_vkp, m_view2->m_image, m_view2->m_vkp, m_matchingKps, out);
	imshow("match", out);
}



