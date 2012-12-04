#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/opencv.hpp>

class feature
{
    public:
        feature(cv::Point2d & uv, cv::Mat & x1to7, cv::Mat & patch, int step, cv::Mat & newFeature);
        ~feature();

        cv::Mat patch_when_initialized;
        cv::Mat patch_when_matching;
        cv::Mat r_wc_when_initialized;
        cv::Mat R_wc_when_initialized;
        cv::Point2d uv_when_initialized;

        int half_patch_size_when_initialized;
        int half_patch_size_when_matching;
        int predicted;
        int measured;
        int init_frame;
        cv::Point2d init_measurement; //same as uv_when_initialized? (only one is column vector, other one is row vector in matlab)

        bool cartesian;
        cv::Mat yi; // this is the 6-d feature as it is appended to the x-vector (why necessary?)
        bool individually_compatible;
        bool low_innovation_inlier;
        bool high_innovation_inlier;

        cv::Mat h; //predicted measurement
        cv::Mat z; //latest measurement
        cv::Mat H;
        cv::Mat S;

        cv::Mat h2; //for ransac hypothesis

        int state_size; //seems unnecessary
        int measurement_size; //seems unnecessary
        cv::Mat R;

    protected:
    private:
};

#endif // FEATURE_H
