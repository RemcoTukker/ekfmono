#include <iostream>
#include <cstdio>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "EKF.h"
#include "feature.h"

using namespace std;
using namespace cv;
/*
 //test for code in feature
int main()
{
    Mat distCoef = (Mat_<double>(1, 4) <<  -0.13, -0.06,0,0); //this is close enough for unibrain fire-i (test sequence)
    double Kdata[] = {2.1735 / 0.0112, 0, 1.7945 / 0.0112,  0, 2.1735 / 0.0112, 1.4433 / 0.0112,  0,0,1};
    int nRows = 240; int nCols = 320;

    Mat K = Mat(3,3,CV_64F, Kdata).clone();
    Eigen::VectorXf xkk;
    xkk.setZero(13);
    double eps = 1e-15;
    xkk << 1, 2, 3, 0.570563, 0.570563, -0.051482, 0.588443, 0, 0, 0, eps,eps,eps;

    Eigen::Matrix<float, 6, 13> dydxv;
     Eigen::Matrix<float, 6, 3> dydhd;
      Eigen::Vector3f n;

    //Point2f point(46,116);
    Point2f point(147, 170);

    feature test(point, xkk, K, 0, 13, dydxv, dydhd, n, K, distCoef );

    float nx = n(0); float ny = n(1); float nz = n(2);

        xkk.conservativeResize(xkk.size() + 6 );
        xkk.tail<6>() << xkk(0), xkk(1), xkk(2), atan2(nx, nz), atan2(-ny, sqrt(nx*nx + nz*nz) ), 1;

    std::cout << dydxv << std::endl;
    std::cout << dydhd << std::endl;
    std::cout << n << std::endl;

    std::cout << "feature added" << std::endl;

     std::cout << test.he << std::endl;

    test.updatePredictionAndDerivatives(xkk , K, distCoef);

    std::cout << test.he << std::endl;
    std::cout << test.He << std::endl;

}
*/


int main()
{

   // get a new frame from camera and store it as first reference point
    Mat frame;
    Mat frameGray, oldGray;

    //this should work but most of the time gives segfaults now (re-install opencv with ffmpeg properly)
    //VideoCapture cap("http://10.10.1.157:8080/videofeed"); // open ip camera

    VideoCapture cap(0);  //to test with local camera
    if(!cap.isOpened())  // check if we succeeded
    {
       cout << "Camera init failed!" << endl;
        return -1;
    }
    sleep(2); //camera is a bit slow to start streaming...

    cap >> frame;
    cvtColor(frame, frameGray, CV_BGR2GRAY);


    //to test with sequence

    //string base = "/home/remco/SLAM/ekfmonoslam/trunk/sequences/ic/rawoutput";
    //frameGray = imread("/home/remco/SLAM/ekfmonoslam/trunk/sequences/ic/rawoutput0000.pgm",0);

    //int width = frameGray.size().width;
    //int height = frameGray.size().height;
    frameGray.copyTo(oldGray);

    //set some variables for fps count
    int fps = 0;
    char fpsbuffer[10];
    time_t previousTime = time(NULL);

    Point2f center(frameGray.cols / 2, frameGray.rows / 2);

    //reasonable settings for a webcam
    // K = [fx 0 cx; 0 fy cy; 0 0 1]
    // cx = 640/2  cy = 480/2  OK, image center ... fx=fy = cx / tan(60 / 2 * pi / 180) ?

    EKF filter;

    //start main loop
    for(;;)
    //for(int i = 0; i< 500;i++)
    {
        fps++;
        if (time(NULL) != previousTime)
        {
            sprintf( fpsbuffer, "%d", fps );
            //sprintf( fpsbuffer, "%d", points[1].size() );
            previousTime = time(NULL);
            fps = 0;
        }

        filter.mapManagement(frameGray);

        filter.ekfPrediction();

        //std::swap(points[1], points[0]); //swap previously detected points to [0], [1] will hold new locations
        swap(frameGray, oldGray); // swap previous frame to oldGray to make place for new frame

        //get a new frame
        cap >> frame;
        cvtColor(frame, frameGray, CV_BGR2GRAY);
        //char numstr[5]; // enough to hold all numbers up to 64-bits
        //sprintf(numstr, "%04d", i);
        //string result = base + numstr + ".pgm";
        //frameGray = imread(result,0);

        filter.searchICmatches(frameGray);
        filter.ransacHypotheses();
        filter.updateLIInliers();
        filter.rescueHIInliers();
        filter.updateHIInliers();
        filter.visualize(frameGray, fpsbuffer);

        // to create an FPS meter
        //frameGray.copyTo(frame);

        //putText(frame, fpsbuffer, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255,255)); //put FPS text

        //give output
        //imshow("detect", frame);

        if(waitKey(30) >= 0) break;
        //sleep(2);

    }
    // the camera will be deinitialized automatically in VideoCapture destructor

    //sleep(100);

    return 0;
}


