#include "feature.h"

feature::feature(cv::Point2f& uv, const Eigen::VectorXf& x1to7, cv::Mat& patch, int step, const Eigen::VectorXf& newFeature, int positioninfilter)
{   //mind you that x1to7 should have 7 elements and newFeature 6 elements (find a way to do this explicitly)


    patch.copyTo(patch_when_initialized);
    patch_when_matching = cv::Mat::zeros(13,13,CV_64F);

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

    init_frame = step;
    init_measurement = uv;
    cartesian = false;
    yi = newFeature;
    high_innovation_inlier = false;
    low_innovation_inlier = false;

    //z h H S stay empty
    //z.release();
    //h.release();
    //H.release();
    //S.release();
    state_size = 6;
    measurement_size = 2;
    Re = Eigen::Matrix2f::Identity();

    position = positioninfilter;

}

feature::~feature()
{
    //dtor
}
