#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/opencv.hpp>
#include "Eigen/Dense"

class feature
{
    public:
        feature(cv::Point2f& uv, const Eigen::VectorXf& x1to7, cv::Mat& patch, int step, int positioninfilter,
                Eigen::Matrix<float, 6, 13>& dydxv, Eigen::Matrix<float, 6, 3>& dydhd, Eigen::Vector3f& n,
                cv::Mat& K , cv::Mat& distCoef);
        ~feature();

        bool updatePrediction(Eigen::VectorXf & xkkp, int store, cv::Mat& K , cv::Mat& distCoef);
        bool updatePredictionAndDerivatives(Eigen::VectorXf & xkkp, cv::Mat& K , cv::Mat& distCoef); //returns true if in front of camera
        void dRqtimesabydq(const Eigen::Vector4f & quat, const Eigen::Vector3f & n, Eigen::Matrix<float, 3, 4> & res);

        cv::Mat patch_when_initialized;
        cv::Mat patch_when_matching;

        Eigen::Vector3f rwcInit;
        Eigen::Vector4f RwcInit;

        cv::Point2f uv_when_initialized;

        int half_patch_size_when_initialized;
        int half_patch_size_when_matching;
        bool predicted;
        int timesPredicted;
        bool measured;
        bool keepaway;
        int timesMeasured;
        int init_frame;
        cv::Point2d init_measurement; //same as uv_when_initialized? (only one is column vector, other one is row vector in matlab)

        cv::Rect lastSearchArea;

        bool cartesian;
        Eigen::Matrix<float, 6, 1> yi; // this is the 6-d feature as it is appended to the x-vector (why necessary?)
        bool low_innovation_inlier;
        bool high_innovation_inlier;

        //cv::Mat h; //predicted measurement
        //cv::Mat z; //latest measurement
        //cv::Mat H;
        //cv::Mat S;

        //cv::Mat h2; //for ransac hypothesis

        Eigen::Vector2f ze;  //latest measurement
        Eigen::Vector2f he;  //predicted measurement
        Eigen::Vector2f h2e; //for ransac hypothesis
        Eigen::MatrixXf He;
        Eigen::Matrix2f Se;
        Eigen::Matrix2f Re;


        int state_size; //seems unnecessary
        int measurement_size; //seems unnecessary
        //cv::Mat R;

        int position;

    protected:
    private:
};

#endif // FEATURE_H
