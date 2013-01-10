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

int main()
{

    //this should work but most of the time gives segfaults now (re-install opencv with ffmpeg properly)
    //VideoCapture cap("http://10.10.1.157:8080/videofeed"); // open ip camera

    //VideoCapture cap(0);  //to test with local camera
    //if(!cap.isOpened())  // check if we succeeded
    //{
    //   cout << "Camera init failed!" << endl;
    //    return -1;
    //}
    //sleep(2);
    //to test with sequence

    // get a new frame from camera and store it as first reference point
    Mat frame, frameBig;
    Mat frameGray, oldGray;

    //cap >> frame;
    //cap >> frameBig;
    //resize(frameBig, frame, Size(), 0.5, 0.5);
    //cvtColor(frame, frameGray, CV_BGR2GRAY);

    string base = "/home/remco/SLAM/ekfmonoslam/trunk/sequences/ic/rawoutput";
    frameGray = imread("/home/remco/SLAM/ekfmonoslam/trunk/sequences/ic/rawoutput0000.pgm",0);

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
    //for(;;)
    for(int i = 0; i< 400;i++)
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
        //cap >> frame;
        //cap >> frameBig;
        //resize(frameBig, frame, Size(), 0.5, 0.5);
        //cvtColor(frame, frameGray, CV_BGR2GRAY);
        char numstr[5]; // enough to hold all numbers up to 64-bits
        sprintf(numstr, "%04d", i);
        string result = base + numstr + ".pgm";
        frameGray = imread(result,0);

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
        sleep(2);

    }
    // the camera will be deinitialized automatically in VideoCapture destructor

    //sleep(100);

    return 0;
}


