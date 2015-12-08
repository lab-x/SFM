#include "funUtils.h"


using namespace std;
using namespace cv;

Mat vec2crossMat(Mat& vec)
{
	CV_Assert(vec.isContinuous());
	float* vec_ptr = (float*)vec.data;
	Mat crossMat(3, 3, CV_32F, Scalar(0));
	float* crossMat_ptr = (float*)crossMat.data;

	crossMat_ptr[1] = -vec_ptr[2];
	crossMat_ptr[2] = vec_ptr[1];
	crossMat_ptr[3] = vec_ptr[2];
	crossMat_ptr[5] = -vec_ptr[0];
	crossMat_ptr[6] = -vec_ptr[1];
	crossMat_ptr[7] = vec_ptr[0];

	return crossMat;
}


Mat solveAx0(Mat& A)
{
	Mat w, u, vt;
	SVD::compute(A, w, u, vt);
	cout << "vt" << vt << endl;
	cout << "w" << w << endl;
	cout << "u" << u << endl;

	float* u_ptr = (float*)u.data;
	Mat x(u.rows,1, CV_32F,Scalar(0));
	float*x_ptr = (float*)x.data;
	for (int i = 0; i < u.rows; i++) x_ptr[i] = u_ptr[u.cols*i+(u.cols-1)];
	return x; // colomn vector !
}

Mat createA_DLT(Mat& P1, Mat& P2, Point2f pt1, Point2f pt2)
{
	Mat A(4, 4, CV_32F, Scalar(0));
	float* A_ptr = (float*)A.data;
	vector<Mat> row(4);
	row[0] = pt1.x*P1.row(2) - P1.row(0);
	row[1] = pt1.y*P1.row(2) - P1.row(1);
	row[2] = pt2.x*P2.row(2) - P2.row(0);
	row[3] = pt2.y*P2.row(2) - P2.row(1);

	for (int i = 0; i < 4; i++){
		float* row_ptr = (float*)row[i].data;
		int i4 = 4 * i;
		for (int j = 0; j < 4; j++){
			A_ptr[i4 + j] = row_ptr[j];
		}
	}

	//cout << "A " << A << endl;
	return A;
}

void isoNormalize(int w, int h, vector<cv::Point2f>& vkp)
{
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
	for (auto& kp : vkp){
		kp -= center;
		kp /= s;
	}

}



inline float getABC(float s, float u0, float u1, float u2, float v0,float v1, float v2, float& a,float& b,float& c)
{

	s*=s;
	u0*=u0;
	u1*=u1;
	u2*=u2;

	v0*=v0;
	v1*=v1;
	v2*=v2;

	a = s*(u0+u1)*(v0+v1);
	b = s*(v2*(u0+u1)+u2*(v0+v1));
	c = s*u2*v2;
}

inline float getABC2(cv::Mat u2M, cv::Mat v1M, cv::Mat v2M, float& a, float& b, float& c)
{
    Vec3f u2(u2M.at<float>(0,0),u2M.at<float>(1,0),u2M.at<float>(2,0));
    Vec3f v1(v1M.at<float>(0,0),v1M.at<float>(1,0),v1M.at<float>(2,0));
    Vec3f v2(v2M.at<float>(0,0),v2M.at<float>(1,0),v2M.at<float>(2,0));

    cv::pow(u2,2,u2);


    a = (u2[0]+u2[1])*(v1[0]*v2[0]+v1[1]*v2[1]);
    b = ((u2[0]+u2[1])*v1[2]*v2[2])+(v1[0]*v2[0]+v1[1]*v2[1])*u2[2];
    c = u2[2]*v1[2]*v2[2];

}


cv::Mat Kruppa(cv::Mat& F)
{
	cv::Mat U,D,Vt; // ! D is column vector

	cv::SVDecomp(F,D,U,Vt);

    cv::Mat V = Vt.t();

	cout<<"D "<<D<<endl;
	cout<<"Vt "<<Vt<<endl;
	cout<<"U "<<U<<endl;

	float s0 = D.at<float>(0,0);
	float s1 = D.at<float>(1,0);
	float s2 = D.at<float>(2,0);

	float a0,b0,c0;
	getABC(s0,U.at<float>(0,0),U.at<float>(1,0),U.at<float>(2,0),
		   Vt.at<float>(0,0),Vt.at<float>(0,1),Vt.at<float>(0,2),a0,b0,c0);

	float a1,b1,c1;
	getABC(s1,U.at<float>(0,1),U.at<float>(1,1),U.at<float>(2,1),
		   Vt.at<float>(1,0),Vt.at<float>(1,1),Vt.at<float>(1,2),a1,b1,c1);

	float a,b,c;

	a = a1-a0;
	b = b1-b0;
	c = c1-c0;

	float rho = b*b-4*a*c;
	float srho = sqrt(rho);

	float alpha2p = (-b+srho)/(2*a);
	float alpha2m = (-b-srho)/(2*a);

	float alpha2 = std::max(alpha2p,alpha2m);
	if(alpha2>=0)
    {
        float alpha = sqrt(alpha2);

        Mat K=Mat::eye(3,3,CV_32F)*alpha;
        K.at<float>(2,2)=1;
        return K;
    }
    else
    {
        getABC2(U.col(1),V.col(0) , V.col(1), a0, b0,c0);
        getABC2(V.col(0) , U.col(0),U.col(1) ,a1, b1,c1);

        a = s1*a0+s0*a1;
        b = s1*b0+s0*b1;
        c = s1*c0+s0*c1;

        rho = b*b-4*a*c;
        srho = sqrt(rho);

        alpha2p = (-b+srho)/(2*a);
        alpha2m = (-b-srho)/(2*a);

        alpha2 = std::max(alpha2p,alpha2m);

        CV_Assert(alpha2>=0);
        float alpha = sqrt(alpha2);

        Mat K=Mat::eye(3,3,CV_32F)*alpha;
        K.at<float>(2,2)=1;
        return K;

    }
}

// correct gives same Ra Rb than
void get4P(cv::Mat&E, vector<cv::Mat>& Pcan) {
	cv::Mat U,I,Vt;
	cv::SVDecomp(E,I,U,Vt);

	float dTab[]={0,1,0,
				  -1,0,0,
				  0,0,1};
	cv::Mat D(3,3,CV_32F,dTab);

	cv::Mat Ra = U*D*Vt;
	cv::Mat Rb = U*D.t()*Vt;

	cv::Mat tu = U.col(U.cols-1);//last singular vector

	cv::hconcat(Ra,tu,Pcan[0]);
	cv::hconcat(Ra,-tu,Pcan[1]);
	cv::hconcat(Rb,tu,Pcan[2]);
	cv::hconcat(Rb,-tu,Pcan[3]);
}




cv::Point2f Pt3ToPt2(cv::Point3f& p3) {
    return Point2f(p3.x,p3.y);
}