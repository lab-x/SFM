//
// Created by derue on 04/12/15.
//

#include "SparseCloud.h"

SparseCloud::SparseCloud()
{
    m_feat2D = cv::xfeatures2d::SIFT::create();
    m_matcher = new cv::BFMatcher;

}
SparseCloud::~SparseCloud()
{
    delete m_matcher;

}

void SparseCloud::initialize(cv::Mat& im1, cv::Mat& im2)
{

    m_vView.push_back(View(im1,m_feat2D));
    m_vView.push_back(View(im2,m_feat2D));

    View& v1 = m_vView[0];
    View& v2 = m_vView[1];

    ViewPair v1v2(&v1,&v2);
    v1v2.matchFeat(m_matcher);
    v1v2.computeFMatrix();

    v1v2.autoCalib();
    v1v2.computeEMatrix();
    v1v2.extractCameraMatrix();

    // first reconstruction
    vector<Mesh::Point> outPt3D;
    v1.m_P = v1.m_KMatrix*v1.m_Rt; // P = K[R t]
    v2.m_P = v2.m_KMatrix*v2.m_Rt;
    reconstruct3DPt(v1.m_vkpCustom,v2.m_vkpCustom,v1v2.m_matchingKps,v1.m_P,v2.m_P, outPt3D);


    //transfer of data for Bundler

    m_vCamera.push_back(CameraT());
    m_vCamera.push_back(CameraT());
    v1.camera = &m_vCamera[0];
    v2.camera = &m_vCamera[1];
    v1.view2cam();
    v2.view2cam();


    point3D_data.resize(outPt3D.size());
    point2D_data.resize(2*outPt3D.size());
    pt3Didx.resize(point2D_data.size());
    camidx.resize(point2D_data.size());
    for(int i=0; i<outPt3D.size(); i++){
        point3D_data[i].SetPoint(outPt3D[i].pos.x,outPt3D[i].pos.y,outPt3D[i].pos.z);
        point2D_data[i].SetPoint2D(v1.m_kpm[i].x,v1.m_kpm[i].y);
        pt3Didx[i] = i;
        camidx[i] = 0;
    }
    for(int i=0; i<outPt3D.size(); i++){
        point2D_data[i+outPt3D.size()].SetPoint2D(v2.m_kpm[i].x,v2.m_kpm[i].y);
        pt3Didx[i+outPt3D.size()] = i;
        camidx[i+outPt3D.size()] = 1;
    }

    //Bundle adjustement
    bundleAdjust();

    //adjust my parameters
    v1.cam2view();
    v2.cam2view();


    for(int i=0; i<outPt3D.size(); i++){
        outPt3D[i].pos.x = point3D_data[i].xyz[0];
        outPt3D[i].pos.y = point3D_data[i].xyz[1];
        outPt3D[i].pos.z = point3D_data[i].xyz[2];
    }

    // add final point3d to the cloud
    //addToCloud(outPt3D);
}


void SparseCloud::reconstruct3DPt(vector<KpCustom>& kp1, vector<KpCustom>& kp2,vector<cv::DMatch>& match, cv::Mat& P1, cv::Mat& P2, vector<Mesh::Point>& outPt3D) {

    outPt3D.resize(match.size());
    cout<<"P1"<<P1<<endl;
    cout<<"P2"<<P2<<endl;

    cv::Mat pt4D;

    cv::triangulatePoints(P1,P2,m_vView[0].m_kpm,m_vView[1].m_kpm,pt4D); // opencv method

    for(int i=0; i< match.size(); i++){

        int i1 = match[i].queryIdx;
        int i2 = match[i].trainIdx;

        cv::Mat pt3DM = pt4D.col(i);
        //cout<<pt3DM<<endl;
        float* pt3D_ptr = (float*)pt3DM.data;
        Mesh::Point newPt3D;
        newPt3D.pos.x = pt3D_ptr[0];
        newPt3D.pos.y = pt3D_ptr[1];
        newPt3D.pos.z = pt3D_ptr[2];

        newPt3D.col.r = ((int)kp1[i1].m_kpColor.r+(int)kp2[i2].m_kpColor.r)/2;
        newPt3D.col.g = ((int)kp1[i1].m_kpColor.g+(int)kp2[i2].m_kpColor.g)/2;
        newPt3D.col.b = ((int)kp1[i1].m_kpColor.b+(int)kp2[i2].m_kpColor.b)/2;

        kp1[i1].m_pt3Didx=i;
        kp2[i2].m_pt3Didx=i;

        //cout<<(int)newPt3D.pos.x<<endl;

        outPt3D[i] = newPt3D;
    }



//======== my method (not working so well)
/*

    for(int i=0; i< match.size(); i++){

        int i1 = match[i].queryIdx;
        int i2 = match[i].trainIdx;


        cv::Mat A = createA_DLT(P1,P2,cv::Point2f(kp1[i1].m_xy_hom.x,kp1[i1].m_xy_hom.y),cv::Point2f(kp2[i2].m_xy_hom.x,kp2[i2].m_xy_hom.y));
        cv::Mat pt3DM = solveAx0(A);

        cout<<pt3DM<<endl;

        float* pt3D_ptr = (float*)pt3DM.data;
        Mesh::Point newPt3D;
        newPt3D.pos.x = pt3D_ptr[0];
        newPt3D.pos.y = pt3D_ptr[1];
        newPt3D.pos.z = pt3D_ptr[2];

        newPt3D.col.r = ((int)kp1[i1].m_kpColor.r+(int)kp2[i2].m_kpColor.r)/2;
        newPt3D.col.g = ((int)kp1[i1].m_kpColor.g+(int)kp2[i2].m_kpColor.g)/2;
        newPt3D.col.b = ((int)kp1[i1].m_kpColor.b+(int)kp2[i2].m_kpColor.b)/2;

        //cout<<(int)newPt3D.pos.x<<endl;

        outPt3D.push_back(newPt3D);
    }
    */
}





void SparseCloud::addView(cv::Mat &vnew) {
    m_vView.push_back(View(vnew,m_feat2D));
    View& v = m_vView.back();
    v.m_KMatrix = m_vView[1].m_KMatrix; // do not forget to update K with bundle adjustement
    v.normalizeIso();

    vector<vector<int>> matchMat(v.m_vkpCustom.size());
    for(int i=0; i<matchMat.size(); i++) matchMat[i].resize(m_vView.size()-1,-1);

    //create all pair
    for(int i=0; i<m_vView.size()-1; i++) {
        ViewPair vi_v(&m_vView[i], &v); // vi (query) search for NN in v (train)
        vi_v.matchFeat(m_matcher);


        //build array of match between all viewpair
        for (int j = 0; j < vi_v.m_matchingKps.size(); j++) {
            int idxvi = vi_v.m_matchingKps[j].queryIdx;
            int idxv = vi_v.m_matchingKps[j].trainIdx;

            matchMat[idxv][i] = idxvi;
        }
    }

    //separate the keypoints in 3 categories : discard, getP and newPoint
    vector<cv::Point3f> pt3D_P;
    vector<cv::Point2f> pt2D_P;

    vector<cv::Point2f> pt2D_ov,pt2D_v;
    vector<int> idx_ov,idx_pt2d_ov,idx_pt2d_v;

    for(int kp_v_idx=0; kp_v_idx<matchMat.size(); kp_v_idx++){
        vector<int> idxKpToKeep_in_otherView;
        vector<int> idx3D_sel;
        for(int v_idx = 0; v_idx<matchMat[kp_v_idx].size(); v_idx++){
            int kpidx_in_v_idx = matchMat[kp_v_idx][v_idx];
            if(kpidx_in_v_idx!=-1){ // check if match
                idxKpToKeep_in_otherView.push_back(v_idx);
                int idPt3D = m_vView[v_idx].m_vkpCustom[kpidx_in_v_idx].m_pt3Didx;
                if(idPt3D!=-1){
                    idx3D_sel.push_back(idPt3D);
                }
            }

        }
        bool toDiscard=false;
        if(!idxKpToKeep_in_otherView.empty()){
            if(idx3D_sel.empty()&&idxKpToKeep_in_otherView.size()==1){
                //newPoint
                // - pt2d of v + its idx
                // - pt2d of otherView + its idx
                // - idx of otherView

                int v_idx = idxKpToKeep_in_otherView[0];
                int id_kp_ov = matchMat[kp_v_idx][v_idx];
                pt2D_ov.push_back(Pt3ToPt2(m_vView[v_idx].m_vkpCustom[id_kp_ov].m_xy_hom));
                pt2D_v.push_back(Pt3ToPt2(v.m_vkpCustom[kp_v_idx].m_xy_hom));
                idx_ov.push_back(v_idx);
                idx_pt2d_ov.push_back(id_kp_ov);
                idx_pt2d_v.push_back(kp_v_idx);

            }
            else if(!idx3D_sel.empty()){
                for(int i=1; i<idx3D_sel.size(); i++){
                    if(idx3D_sel[0]!=idx3D_sel[i]) {
                        toDiscard = true;
                        break;
                    }
                }
                if(!toDiscard){
                    //getP
                    // -need idx3D

                    pt3D_P.push_back(cv::Point3f(point3D_data[idx3D_sel[0]].xyz[0],point3D_data[idx3D_sel[0]].xyz[1],point3D_data[idx3D_sel[0]].xyz[2]));
                    // -need kp_v_idx
                    cout<<"P3D "<<pt3D_P.back()<<endl;
                    pt2D_P.push_back(Pt3ToPt2(v.m_vkpCustom[kp_v_idx].m_xy_hom));
                    cout<<"P2D "<<pt2D_P.back()<<endl;

                }
            }
        }
    }

    cout<<"npoint to reconstruct : "<<pt3D_P.size()<<endl;
    cout<<"nnew point : "<<pt2D_ov.size()<<endl;


    //recoverP(toGetP)
    cv::Mat Rvec,R,t;
    cv::solvePnPRansac(pt3D_P,pt2D_P,v.m_KMatrix,cv::Mat(),Rvec,t);

    cv::Rodrigues(Rvec,R);
    cv::hconcat(R,t,v.m_Rt);

    cout<<"Rt "<<v.m_Rt<<endl;

    v.m_Rt.convertTo(v.m_Rt,CV_32F);
    v.m_P = v.m_KMatrix*v.m_Rt;

    m_vCamera.push_back(CameraT());
    v.camera = &m_vCamera.back();

    //transfer to bundler variable
    v.view2cam();

    //triangulate newpoint
    for(int i=0; i<pt2D_ov.size(); i++){
        cv::Mat newPt3D;
        cv::triangulatePoints(m_vView[idx_ov[i]].m_P,v.m_P,vector<cv::Point2f>(1,pt2D_ov[i]),vector<cv::Point2f>(1,pt2D_v[i]),newPt3D);

        point2D_data.push_back(Point2D(pt2D_ov[i].x,pt2D_ov[i].y));
        camidx.push_back(idx_ov[i]);


        point2D_data.push_back(Point2D(pt2D_v[i].x,pt2D_v[i].y));
        camidx.push_back(m_vCamera.size()-1);

        float* newPt3D_p = (float*)newPt3D.data;
        point3D_data.push_back(Point3D{newPt3D_p[0],newPt3D_p[1],newPt3D_p[2]});
        pt3Didx.push_back(point3D_data.size()-1);
        pt3Didx.push_back(point3D_data.size()-1);

        //assign idx point3d to keypoint of each view
        m_vView[idx_ov[i]].m_vkpCustom[idx_pt2d_ov[i]].m_pt3Didx = (int)point3D_data.size()-1;
        v.m_vkpCustom[idx_pt2d_v[i]].m_pt3Didx = (int)point3D_data.size()-1;
    }

    bundleAdjust();
    v.cam2view();

}

void SparseCloud::bundleAdjust()
{
    pba.SetCameraData(m_vCamera.size(),  &m_vCamera[0]);                        //set camera parameters
    pba.SetPointData(point3D_data.size(), &point3D_data[0]);                            //set 3D point data
    pba.SetProjection(point2D_data.size(), &point2D_data[0], &pt3Didx[0], &camidx[0]);//set the projections
    pba.RunBundleAdjustment();

}

void SparseCloud::generateCloud() {

    m_cloud3D.resize(point3D_data.size());
    for(int i=0; i<point3D_data.size(); i++)
    {
        point3D_data[i].GetPoint(m_cloud3D[i].pos.x,m_cloud3D[i].pos.y,m_cloud3D[i].pos.z);
    }
}
