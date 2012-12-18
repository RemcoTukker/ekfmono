#include "feature.h"

feature::feature(cv::Point2d & uv, cv::Mat & x1to7, cv::Mat & patch, int step, cv::Mat & newFeature)
{
    patch.copyTo(patch_when_initialized);
    patch_when_matching = cv::Mat::zeros(13,13,CV_64F);
    x1to7.colRange(cv::Range(0,3)).copyTo(r_wc_when_initialized);
    x1to7.colRange(cv::Range(3,7)).copyTo(R_wc_when_initialized);
    uv_when_initialized = uv;
    half_patch_size_when_initialized = 20;
    half_patch_size_when_matching = 6;

    predicted = 0;
    measured = 0;

    init_frame = step;
    init_measurement = uv;
    cartesian = false;
    newFeature.copyTo(yi);
    high_innovation_inlier = false;
    low_innovation_inlier = false;
    individually_compatible = false;

    //z h H S stay empty
    state_size = 6;
    measurement_size = 2;
    R = cv::Mat::eye(2,2, CV_64F);

}

feature::~feature()
{
    //dtor
}
