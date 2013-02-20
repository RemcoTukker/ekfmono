#include "feature.h"

#include <iostream>

using namespace cv;

feature::feature(Point2f& uv, const Eigen::VectorXf& x1to7, Mat& patch, int step, int positioninfilter,
                 Eigen::Matrix<float, 6, 13>& dydxv, Eigen::Matrix<float, 6, 3>& dydhd, Eigen::Vector3f& n, cv::Mat& K , cv::Mat& distCoef)
{
    //mind you that x1to7 should have 7 elements and newFeature 6 elements (find a way to do this explicitly)

    patch.copyTo(patch_when_initialized);
    patch_when_matching = Mat::zeros(13,13,CV_64F);

    rwcInit = x1to7.head<3>();
    RwcInit = x1to7.segment<4>(3);
    //x1to7.colRange(cv::Range(0,3)).copyTo(r_wc_when_initialized);
    //x1to7.colRange(cv::Range(3,7)).copyTo(R_wc_when_initialized);

    uv_when_initialized = uv;
    half_patch_size_when_initialized = 20;
    half_patch_size_when_matching = 6;

    predicted = false;
    timesPredicted = 0;
    measured = false;
    timesMeasured = 0;

    keepaway = true;
    he = Eigen::Vector2f(uv.x , uv.y);

    init_frame = step;
    init_measurement = uv;
    cartesian = false;
    ///TODO fix this: //yi = newFeature;
    high_innovation_inlier = false;
    low_innovation_inlier = false;

    state_size = 6;
    measurement_size = 2;
    Re = Eigen::Matrix2f::Identity();

    position = positioninfilter;


    //all fields stored; now calculate the derivatives that will be sent back

    //first undistort
    vector<Point2f> src;
    src.push_back( uv );
    vector<Point2f> undistorted;
    undistortPoints(src , undistorted, K , distCoef, Mat());

    Point3f h_LR(undistorted[0].x, undistorted[0].y, 1);

    //std::cout << "h_LR " << h_LR << std::endl;

    //reproject point to get jacobians
    Mat tvec = Mat::zeros(1,3,CV_64F), rvec = Mat::zeros(1,3,CV_64F); //carefull with changing this, might mess up jacobians?
    vector<Point2f> projectedLocation;
    vector<Point3f> coordinates;
    coordinates.push_back(h_LR);
    Mat jacobians;
    projectPoints( coordinates, rvec, tvec, K, distCoef, projectedLocation, jacobians);
    Mat dhrl_dh;

    Mat dh_dhrl = jacobians.colRange(3,6).clone();
    Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > dhdhrldouble(dh_dhrl.ptr<double>(0));
    Eigen::Matrix<float, 2, 3> dhdhrl = dhdhrldouble.cast<float>();

    //std::cout << "dhdhrl " <<jacobians.colRange(3,6)<< std::endl;

    Eigen::Vector4f q = x1to7.segment(3, 4);     //camera rotation quaternion (r x y z for now)
    Eigen::Quaternionf q2(q(0),q(1),q(2),q(3)); //Note that we cannot just use q b/c of different order; can we make it prettier?
    Eigen::Vector3f hLR(h_LR.x , h_LR.y, h_LR.z);
    n = q2 * hLR;
    float nx = n(0), ny = n(1), nz = n(2);

    //start calculating derivatives
    Eigen::RowVector3f dthetadgw(nz / (nx*nx + nz*nz), 0, -nx / (nx*nx + nz*nz));
    Eigen::RowVector3f dphidgw((nx*ny) / ((nx*nx+ny*ny+nz*nz) * sqrt(nx*nx + nz*nz)),
                               -sqrt(nx*nx+nz*nz)/(nx*nx+ny*ny+nz*nz), (nz*ny) / ((nx*nx+ny*ny+nz*nz)*sqrt(nx*nx + nz*nz)));

    //std::cout << "dthetadgw " << dthetadgw << std::endl;
    //std::cout << "dphidgw " << dphidgw << std::endl;

    Eigen::Matrix<float, 3, 4> dgwdqwr;
    dRqtimesabydq(q, hLR, dgwdqwr);

    //std::cout << "dgwdqwr " << dgwdqwr << std::endl;

    dydxv = Eigen::Matrix<float, 6, 13>::Zero();
    dydxv.topLeftCorner<3,3>() = Eigen::Matrix3f::Identity();
    dydxv.block<1,4>(3,3) = dthetadgw * dgwdqwr;
    dydxv.block<1,4>(4,3) = dphidgw * dgwdqwr;  //dydxv done!

    Eigen::Matrix<float, 5, 3> dyprimedgw = Eigen::Matrix<float, 5, 3>::Zero();
    dyprimedgw.row(3) = dthetadgw;
    dyprimedgw.row(4) = dphidgw;

    Eigen::Matrix<float, 3, 3> dgcdhrl;
    //dgcdhrl << 1/h_LR.x, 0, -h_LR.x/(h_LR.z*h_LR.z), 0, 1/h_LR.y, -h_LR.y/(h_LR.z*h_LR.z), 0, 0, 0;
    float theta = atan2(nx, nz);
    float phi = atan2(-ny, sqrt(nx*nx + nz*nz) );
    Eigen::Vector3f mi(cos(phi)*sin(theta), -sin(phi), cos(phi)*cos(theta));
    Eigen::Vector3f hc = q2.inverse() * mi;
    float hx = hc(0); float hy = hc(1); float hz = hc(2);

    dgcdhrl << 1/hz, 0, -hx/(hz*hz), 0, 1/hz, -hy/(hz*hz), 0, 0, 0;
    Eigen::Matrix<float, 3, 2> dhdhrlrightinverse = dhdhrl.transpose() * ( dhdhrl * dhdhrl.transpose() ).inverse();

    dydhd = Eigen::Matrix<float, 6, 3>::Zero();
    dydhd(5,2) = 1;
    dydhd.topLeftCorner<5,2>() = dyprimedgw * q2.toRotationMatrix() * dgcdhrl * dhdhrlrightinverse ; //dydhd done!
    //dgw -> w for world coordinates, dgc -> c for camera coordinates,

}

feature::~feature()
{
    //dtor
}

bool feature::updatePrediction(Eigen::VectorXf & xkkp, int store , Mat& K , Mat& distCoef)
{
    Eigen::Vector3f twc = xkkp.segment(0, 3);   //camera translation in world coordinates
    Eigen::Vector4f q = xkkp.segment(3, 4);     //camera rotation quaternion (r x y z for now)
    Eigen::Quaternionf q2(q(0),q(1),q(2),q(3)); //Note that we cannot just use q b/c of different order; can we make it prettier?

    Eigen::Vector3f hrel;
    Eigen::Vector3f trans;
    Eigen::Vector3f y = xkkp.segment(position, 3);

    if (cartesian)
    {
        trans = y - twc;
        hrel = q2.inverse() * trans;  //feature coordinates relative to camera

    }
    else
    {
        float rho = xkkp(position + 5);   //this x_k_km1 stuff, the features..
        float phi = xkkp(position + 4);
        float theta = xkkp(position + 3); //are the same as in x_k_k (thus no need to copy?)

        Eigen::Vector3f m( cos(phi) * sin(theta), -sin(phi), cos(phi) * cos(theta) );
        trans = (y - twc) * rho + m;
        hrel = q2.inverse() * trans; //is this right? transpose == inverse, right?
    }

    if (hrel(2) < 0) return false; //if feature is not in front of camera

    Mat tvec = Mat::zeros(1,3,CV_64F), rvec = Mat::zeros(1,3,CV_64F); //we can use translation as calculated above, and rotation too?
    vector<Point2d> projectedLocation;
    vector<Point3d> coordinates;
    coordinates.push_back(Point3d(hrel(0),hrel(1),hrel(2) ));

    projectPoints( coordinates, rvec, tvec, K, distCoef, projectedLocation);

    switch (store)
    {
    case 1:
        he = Eigen::Vector2f(projectedLocation[0].x , projectedLocation[0].y);
        break;
    case 2:
        h2e = Eigen::Vector2f(projectedLocation[0].x , projectedLocation[0].y);
        break;
    default:
        exit(11);
    }
    return true;
}


bool feature::updatePredictionAndDerivatives(Eigen::VectorXf & xkkp, Mat& K , Mat& distCoef)
{
    Eigen::Vector3f twc = xkkp.segment(0, 3);   //camera translation in world coordinates
    Eigen::Vector4f q = xkkp.segment(3, 4);     //camera rotation quaternion (r x y z for now)
    Eigen::Quaternionf q2(q(0),q(1),q(2),q(3)); //Note that we cannot just use q b/c of different order; can we make it prettier?
    Eigen::Vector3f hrel;
    Eigen::Vector3f trans;
    Eigen::Vector3f y = xkkp.segment(position, 3);

    if (cartesian)
    {
        trans = y - twc;
        hrel = q2.inverse() * trans;  //feature coordinates relative to camera

    }
    else
    {
        float rho = xkkp(position + 5);   //this x_k_km1 stuff, the features..
        float phi = xkkp(position + 4);
        float theta = xkkp(position + 3); //are the same as in x_k_k (thus no need to copy?)

        Eigen::Vector3f m( cos(phi) * sin(theta), -sin(phi), cos(phi) * cos(theta) );
        trans = (y - twc) * rho + m;
        hrel = q2.inverse() * trans; //is this right? transpose == inverse, right?
    }

    if (hrel(2) < 0) return false; //if feature is not in front of camera

    Mat tvec = Mat::zeros(1,3,CV_64F), rvec = Mat::zeros(1,3,CV_64F); //we can use translation as calculated above, and rotation too?
    vector<Point2d> projectedLocation;
    vector<Point3d> coordinates;
    coordinates.push_back(Point3d(hrel(0),hrel(1),hrel(2) ));

    Mat jacobians;
    projectPoints( coordinates, rvec, tvec, K, distCoef, projectedLocation, jacobians);

    he = Eigen::Vector2f(projectedLocation[0].x , projectedLocation[0].y);

    //also calculate derivative H  ( = dh/dx|predictedstate )   2xn matrix cause h returns x and y coordinates
    Mat dh_dhrl = jacobians.colRange(3,6).clone();

    Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > dhdhrl(dh_dhrl.ptr<double>(0));

    Eigen::Matrix<float, 3, 4> dgwdqwr;
    Eigen::Vector4f qconj = -q; //conjugate is inverse for quaternions
    qconj(0) = q(0);
    dRqtimesabydq(qconj, trans, dgwdqwr); //can we not do this prettier?
    Eigen::Matrix4f dqbarbydq = Eigen::Matrix4f::Zero();
    dqbarbydq.diagonal() << 1,-1,-1,-1;
    Eigen::Matrix<float, 2, 4> dhdqwr = dhdhrl.cast<float>() * dgwdqwr * dqbarbydq;  //add some explanation here, what _are_ we doing?

    //std::cout << "dhrldqwr: " << dgwdqwr * dqbarbydq << std::endl;

    Eigen::Matrix<float, 2, Eigen::Dynamic> Hie(2,xkkp.rows());

    Eigen::Matrix<float, 2, 3> dhdrw = dhdhrl.cast<float>() * -q2.inverse().toRotationMatrix();

    if (cartesian)  //now parametrization specific stuff
    {
        Hie << dhdrw, dhdqwr, Eigen::MatrixXf::Zero(2,position - 7),
        dhdhrl.cast<float>() * q2.inverse().toRotationMatrix(), Eigen::MatrixXf::Zero(2, xkkp.rows() - position - 3);
        ///something wrong? dhdrw same as what we calculate and put in middle of Hie ?
    }
    else
    {
        float lambda = xkkp(position + 5);  //different parametrization here, nasty
        float phi = xkkp(position + 4);
        float theta = xkkp(position + 3);

        Eigen::Matrix<float, 3, 6> dhrldy;
        dhrldy << q2.inverse().toRotationMatrix() * lambda ,
        q2.inverse() * Eigen::Vector3f( cos(phi)*cos(theta), 0, -cos(phi)*sin(theta) ) ,
        q2.inverse() * Eigen::Vector3f( -sin(phi)*sin(theta), -cos(phi), -sin(phi)*cos(theta) ) ,
        q2.inverse() * ( y - twc ) ; //strange that rho and mi are not here now.. Same as in matlab code, but correct?

        Hie << dhdrw * lambda, dhdqwr, Eigen::MatrixXf::Zero(2,position - 7),
        dhdhrl.cast<float>() * dhrldy, Eigen::MatrixXf::Zero(2, xkkp.rows() - position - 6);
    }

    He = Hie;

    return true; //feature is in front of camera and everything is calculated
}

void feature::dRqtimesabydq(const Eigen::Vector4f & quat, const Eigen::Vector3f & n, Eigen::Matrix<float, 3, 4> & res) //this works
{
    float q0 = 2*quat(0), qx = 2*quat(1), qy = 2*quat(2), qz = 2*quat(3);
    Eigen::Matrix3f dRbydq0;
    dRbydq0 << q0, -qz, qy,   qz, q0, -qx,   -qy, qx, q0 ;
    Eigen::Matrix3f dRbydqx;
    dRbydqx << qx, qy, qz,   qy, -qx, -q0,   qz, q0, -qx ;
    Eigen::Matrix3f dRbydqy;
    dRbydqy << -qy, qx, q0,   qx, qy, qz,   -q0, qz, -qy ;
    Eigen::Matrix3f dRbydqz;
    dRbydqz << -qz, -q0, qx,   q0, -qz, qy,   qx, qy, qz ;

    res << dRbydq0 * n, dRbydqx * n, dRbydqy * n, dRbydqz * n;

}
