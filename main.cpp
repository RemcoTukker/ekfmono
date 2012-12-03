#include <iostream>
#include <cstdio>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "EKF.h"
#include "feature.h"

using namespace std;
using namespace cv;

int main()
{

    //this should work but most of the time gives segfaults now (re-install opencv with ffmpeg properly)
    //VideoCapture cap("http://10.10.1.157:8080/videofeed"); // open ip camera

    VideoCapture cap(0);  //to test with local camera

    if(!cap.isOpened())  // check if we succeeded
    {
        cout << "Camera init failed!" << endl;
        return -1;
    }

    // get a new frame from camera and store it as first reference point
    Mat frame;
    Mat frameGray, oldGray;
    cap >> frame;
    int width = frame.size().width;
    int height = frame.size().height;
    cvtColor(frame, frameGray, CV_BGR2GRAY);
    frameGray.copyTo(oldGray);

    //set some variables for fps count
    int fps = 0;
    char fpsbuffer[10];
    time_t previousTime = time(NULL);

    Point2f center(width / 2, height / 2);

    //reasonable settings for a webcam
    // K = [fx 0 cx; 0 fy cy; 0 0 1]
    // cx = 640/2  cy = 480/2  OK, image center ... fx=fy = cx / tan(60 / 2 * pi / 180) ?

    EKF filter;

    namedWindow("detect",1);

    //start main loop
    for(;;)
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


        filter.searchICmatches(frameGray);



        // to create an FPS meter
        putText(frame, fpsbuffer, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255,255)); //put FPS text

        //give output
        imshow("detect", frame);

        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor


    return 0;
}


