#ifndef EKF_H
#define EKF_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "feature.h"

using namespace cv;

class EKF
{
    public:
        EKF();
        ~EKF();
        void mapManagement(Mat & frame);
        void ekfPrediction();
        void searchICmatches(Mat & frame);
    protected:

    private:
        void multiplyQuaternion(const Mat& q1,const Mat& q2, Mat& q);
        void dqomegadtbydomega(const Mat & wW, const double delta_t, Mat & res);
        double dq0_by_domegaA(double omegaA, double omega, double delta_t); // these 3 are only required for dqomegadtbydomega
        double dqA_by_domegaA(double omegaA, double omega, double delta_t); // so maybe move to a separate class or something
        double dqA_by_domegaB(double omegaA, double omegaB, double omega, double delta_t); // or make it inlines
        void jacob_undistor_fm(Point2d coor, Mat & res);
        void dRq_times_a_by_dq(const Mat & quat, const Mat & n, Mat & res);
        Mat quaternion2rotmatrix(Mat & quat);
        void convertToCartesian();
        void deleteFeatures();
        void addAndUpdateFeatures(Mat & frame);
        vector<feature> features_info;
        vector< vector <double> > trajectory;
        vector<int> measurements;
        vector<int> predicted_measurements;
        int min_number_of_features_in_image;

        int step;

        Mat x_k_k;
        Mat p_k_k;

        Mat x_k_km1;
        Mat p_k_km1;

        Mat K;
        Mat distCoef;
        //Mat measurements;
        //Mat predictedMeasurements;

        double sigma_a;
        double sigma_alpha;
        double sigma_image_noise;

};

#endif // EKF_H
