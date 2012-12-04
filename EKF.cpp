#include "EKF.h"

using namespace cv;

EKF::EKF() //ctor
{
    min_number_of_features_in_image = 25;

    sigma_a = 0.007;
    sigma_alpha = 0.007;
    sigma_image_noise = 1.0;

    double eps = .00000000001;

    //init cam
    double Kdata[] = {640/(2*tan(60*3.14/360)) ,0, 640/2,  0, 640/(2*tan(60*3.14/360)), 480/2,  0,0,1}; //is this OK? esp. focal lengths..
    K = Mat(3,3,CV_64F, Kdata).clone();  //fix this better! find out theory and so on...
    distCoef = (Mat_<double>(1, 4) <<  0.06333, 0.0139, 0, 0); //TODO get better distortion coefficients!

    //init state

    double xdata[] = {0,0,0, 1,0,0,0, 0,0,0, eps,eps,eps};
    x_k_k = Mat(1,13,CV_64F, xdata).clone();

    //init covariance matrix
    p_k_k = Mat::zeros(13, 13, CV_64F);
    p_k_k.at<double>(0,0) = eps;
    p_k_k.at<double>(1,1) = eps;
    p_k_k.at<double>(2,2) = eps;

    p_k_k.at<double>(3,3) = eps;
    p_k_k.at<double>(4,4) = eps;
    p_k_k.at<double>(5,5) = eps;
    p_k_k.at<double>(6,6) = eps;

    p_k_k.at<double>(7,7) = 0.025*0.025;
    p_k_k.at<double>(8,8) = 0.025*0.025;
    p_k_k.at<double>(9,9) = 0.025*0.025;

    p_k_k.at<double>(10,10) = 0.025*0.025;
    p_k_k.at<double>(11,11) = 0.025*0.025;
    p_k_k.at<double>(12,12) = 0.025*0.025;

    // assume linear velocity for now

    step = 1;

}

EKF::~EKF()
{
    //dtor
}

void EKF::deleteFeatures()
{
    //TODO: would be better to go backwards for speed. Also, clean up the ugliness
    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ) //note, no ++it on purpose here
    {
        int position = 13;
        if ((it->measured < 0.5*it->predicted) && (it->predicted > 5))
        {
            int fsize;
            if (it->cartesian)
                fsize = 3;
            else
                fsize = 6;

            int xsize = x_k_k.size().width;
            Mat newx, newp;

            if ((position + fsize) == xsize) //if it is the last feature we are deleting
            {
                newx = x_k_k.colRange(Range(0,position));
                newp = p_k_k.colRange(Range(0,position)).rowRange(Range(0,position));
            }
            else
            {
                newx = x_k_k.colRange(Range(0,position));
                hconcat(newx, x_k_k.colRange(Range(position + fsize, xsize)), newx);

                Mat newp1;
                hconcat(p_k_k.colRange(Range(0,position)),p_k_k.colRange(Range(position + fsize, xsize)),newp1 );
                vconcat(newp1.rowRange(Range(0,position)),newp1.rowRange(Range(position + fsize, xsize)),newp );
            }

            newx.copyTo(x_k_k);
            newp.copyTo(p_k_k);

            it = features_info.erase(it);
        }
        else
        {
            if (it->cartesian)
                position = position + 6;
            else
                position = position + 3;
            ++it;
        }
    }
}

void EKF::addAndUpdateFeatures(Mat & frame)
{
    int measuredFeatures = 0;
    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if (it->low_innovation_inlier || it->high_innovation_inlier)
        {
            measuredFeatures++;
            it->measured++;
        }

        if (! it->h.empty() ) it->predicted++;

        it->high_innovation_inlier = false;
        it->low_innovation_inlier = false;
        it->individually_compatible = false;
        it->h.release();
        it->z.release();
        it->H.release();
        it->S.release();
        it->h2.release(); // required for ransac hypothesis

    }

    convertToCartesian(); //something goes wrong in here i think...

    //start adding new features
    int max_attempts = 50;
    int initialized = 0, attempts = 0;
    while ((initialized + measuredFeatures < min_number_of_features_in_image) && ( attempts < max_attempts ))
    {
        attempts++;
        bool success = false;

        //extract fast corners and check if it is far enough away from others (predicted from EKF)

        //make a random small box further than 21 pixels away from borders, 60x40 large
        int x = 21 + rand() % (frame.size().width - 42 - 60 );
        int y = 21 + rand() % (frame.size().height - 42 - 40 );
        Mat searchBox(frame, Rect(x,y,60,40) );
        //extract FAST corners here
        vector<KeyPoint> result;
        FAST(searchBox, result, 40, true);

        Point2f FASTresult;

        if (result.size() > 0)
        {
            success = true;

            int maxResponse = 0;
            int maxIdx = 0;
            for (int sre = 0; sre < result.size(); sre++)
            {
                if (result[sre].response > maxResponse)
                {
                    maxResponse = result[sre].response;
                    maxIdx = sre;
                }
            }

            FASTresult.x = result[maxIdx].pt.x + x;
            FASTresult.y = result[maxIdx].pt.y + y;

            vector<Point2f> resultVector;
            resultVector.push_back(FASTresult);

            //use cornerSubPix to get better initial estimates
            // Set the needed parameters to find the refined corners
            Size winSize = Size( 5, 5 );
            Size zeroZone = Size( -1, -1 );
            TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 20, 0.01 );

            // Calculate the refined corner locations
            cornerSubPix( frame, resultVector, winSize, zeroZone, criteria );

            FASTresult = resultVector[0];

            //check if feature is far enough away from existing features (either same box or minimum distance)
            int position = 13;
            Mat t_wc = x_k_k.colRange(0, 3);  //t_wc
            Mat quat = x_k_k.colRange(Range(3,7));
            Mat rotMat = quaternion2rotmatrix(quat); //r_wc

            for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
            {
                //we should use projectPoints here instead! (and also use the jacobians that it can calculate somewhere later!)

                //in fact, we may also compare with last measured position in case of slow movements; may be faster

                Mat hrl;
                vector<Point2f> projectedLocation;

                if (it->cartesian)
                {
                    Mat yi = x_k_k.colRange(position, position + 3);
                    Mat r_cw;
                    invert(rotMat, r_cw);  /// its possible to check return value if necessary; also, can we directly use it in projectPoints?
                    hrl = r_cw * (yi.t() - t_wc.t()); //this should give point in camera coordinates.. does it?

                    position = position + 3;

                    Mat tvec = Mat::zeros(1,3,CV_64F), rvec = Mat::zeros(1,3,CV_64F);
                    vector<Point3f> coordinates;
                    coordinates.push_back(Point3f(hrl.at<double>(0),hrl.at<double>(1),hrl.at<double>(2) ));
                    projectPoints( coordinates, rvec, tvec, K, distCoef, projectedLocation);
                }
                else
                {
                    Mat yi = x_k_k.colRange(position, position + 3);
                    double rho = x_k_k.at<double>(0,position + 5);
                    double theta = x_k_k.at<double>(0,position + 3);
                    double phi = x_k_k.at<double>(0,position + 4);
                    Mat mi = (Mat_<double>(3,1) << cos(phi) * sin(theta), -sin(phi), cos(phi) * cos(theta) );
                    hrl = rotMat.t() * ( (yi.t() - t_wc.t() )*rho + mi );

                    position = position + 6;  ///TODO: check how we can make this prettier; rotation is 0 in the beginning,
                                                ///gotta change that if we want to test this..
                    Mat tvec = Mat::zeros(1,3,CV_64F), rvec = Mat::zeros(1,3,CV_64F);
                    vector<Point3f> coordinates;
                    coordinates.push_back(Point3f(hrl.at<double>(0),hrl.at<double>(1),hrl.at<double>(2) ));
                    projectPoints( coordinates, rvec, tvec, K, distCoef, projectedLocation);
                }

                Point2f location(projectedLocation[0]);

                std::cout << location << std::endl;

                if ( (norm(location - FASTresult) < 20) && (hrl.at<double>(0,3) > 0) ) //should also be in front of camera
                {
                    success = false;
                    break;
                }

            }
            //perhaps use projectPoints here, after having collected the points 3d coordinates (is likely slower though)
        }

        if (success)
        {

            //add feature to x_k_k

            //first undistort
            vector<Point2d> src;
            src.push_back( FASTresult );

            vector<Point2d> undistorted;
            undistortPoints(src , undistorted, K , distCoef, Mat(), K);

            Mat h_LR  = (Mat_<double>(3, 1) << -(K.at<double>(0,2) - undistorted[0].x) / K.at<double>(0,0),
                         -(K.at<double>(1,2) - undistorted[0].y) / K.at<double>(1,1), 1);

            //rotation quaternion
            Mat quat = x_k_k.colRange(Range(3,7));
            Mat rotMat = quaternion2rotmatrix(quat);

            Mat n = rotMat * h_LR;
            double nx = n.at<double>(0,0);
            double ny = n.at<double>(1,0);
            double nz = n.at<double>(2,0);

            Mat newFeature  = (Mat_<double>(1, 6) << x_k_k.at<double>(0,0), x_k_k.at<double>(0,1), x_k_k.at<double>(0,2),
                               atan2(nx, nz), atan2(-ny, sqrt(nx*nx + nz*nz) ), 1); //last parameter = initial_rho = 1

            hconcat(x_k_k, newFeature, x_k_k);

            //add feature to p_k_k
            Mat dtheta_dgw = (Mat_<double>(1, 3) << nz / (nx*nx + nz*nz), 0, -nx / (nx*nx + nz*nz) );
            Mat dphi_dgw = (Mat_<double>(1, 3) << (nx*ny) / ((nx*nx+ny*ny+nz*nz) * sqrt(nx*nx + nz*nz)),
                            -sqrt(nx*nx+nz*nz)/(nx*nx+ny*ny+nz*nz), (nz+ny) / ((nx*nx+ny*ny+nz*nz)*sqrt(nx*nx + nz*nz)) );

            Mat dgwdqwr;
            dRq_times_a_by_dq(quat, n, dgwdqwr);

            Mat dtheta_dqwr = dtheta_dgw * dgwdqwr;
            Mat dphi_dqwr = dphi_dgw * dgwdqwr;
            Mat dy_dqwr = Mat::zeros(3, 4, CV_64F);
            vconcat(dy_dqwr, dtheta_dqwr, dy_dqwr);
            vconcat(dy_dqwr, dphi_dqwr, dy_dqwr);
            vconcat(dy_dqwr, Mat::zeros(1,4,CV_64F), dy_dqwr);

            Mat dy_dxv = Mat::eye(6,3, CV_64F);
            hconcat(dy_dxv, dy_dqwr, dy_dxv);
            hconcat(dy_dxv, Mat::zeros(6,6,CV_64F), dy_dxv); //finally, first component done!

            Mat dyprima_dgw = Mat::zeros(3,3,CV_64F);
            vconcat(dyprima_dgw, dtheta_dgw, dyprima_dgw);
            vconcat(dyprima_dgw, dphi_dgw, dyprima_dgw);
            Mat dgc_dhu = (Mat_<double>(3, 2) << 1/K.at<double>(0,0), 0, 0, 1/K.at<double>(1,1), 0, 0 ); //two columns, three rows

            Mat dhu_dhd = Mat::zeros(2,2,CV_64F);
            jacob_undistor_fm(src[0], dhu_dhd);

            Mat dyprima_dhd = dyprima_dgw*rotMat*dgc_dhu*dhu_dhd;
            hconcat(dyprima_dhd, Mat::zeros(5,1,CV_64F) , dyprima_dhd);
            Mat dy_dhd = (Mat_<double>(1, 3) << 0, 0, 1);
            vconcat(dyprima_dhd, dy_dhd, dy_dhd);  //finally, another component!

            Mat Padd = (Mat_<double>(3, 3) << sigma_image_noise*sigma_image_noise, 0, 0, 0, sigma_image_noise*sigma_image_noise, 0, 0, 0, 1);

            //assemble new p_k_k
            Mat P_xv = p_k_k.rowRange(0,13).colRange(0,13);
            int psize = p_k_k.size().width;

            Mat rightadd = P_xv*dy_dxv.t();
            Mat bottomadd = dy_dxv * P_xv;

            if (psize != 13)
            {
                Mat P_xvy = p_k_k.rowRange(0,13).colRange(13,psize);
                Mat P_yxv = p_k_k.rowRange(13,psize).colRange(0,13);
                vconcat(rightadd, P_yxv*dy_dxv.t(), rightadd);
                hconcat(bottomadd, dy_dxv * P_xvy, bottomadd);
            }

            hconcat(p_k_k, rightadd, p_k_k);
            hconcat(bottomadd, dy_dxv*P_xv*dy_dxv.t() + dy_dhd*Padd*dy_dhd.t(), bottomadd);
            vconcat(p_k_k, bottomadd , p_k_k);   //finally done with updating p_k_k! yay!

            //add feature to info vector
            Mat patch = frame.colRange( (int) src[0].x - 20, (int) src[0].x + 21).rowRange( (int) src[0].y - 20, (int) src[0].y + 21) ;
            feature newFeatureInfo(src[0], x_k_k, patch, step, newFeature );  //heavy lifting is done in constructor
            //NB: Should the first argument be the distorted point (as in matlab code) or the undistorted point? its probably right

            features_info.push_back(newFeatureInfo);

            initialized++;


            //lets check results after first init here
            //std::cout << x_k_k << std::endl;
            //std::cout << p_k_k << std::endl;
            std::cout << FASTresult << undistorted[0] << std::endl;

           // exit(1);
        }

    }
    exit(1);
}

void EKF::dRq_times_a_by_dq(const Mat & quat, const Mat & n, Mat & res)
{
    double q0 = quat.at<double>(0), qx = 2*quat.at<double>(1), qy = 2*quat.at<double>(2), qz = 2*quat.at<double>(3);
    Mat dRbydq0 = (Mat_<double>(3, 3) << q0, -qz, qy,   qz, q0, -qx,   -qy, qx, q0 );
    Mat dRbydqx = (Mat_<double>(3, 3) << qx, qy, qz,   qy, -qx, -q0,   qz, q0, -qx );
    Mat dRbydqy = (Mat_<double>(3, 3) << -qy, qx, q0,   qx, qy, qz,   -q0, qz, -qy );
    Mat dRbydqz = (Mat_<double>(3, 3) << -qz, -q0, qx,   q0, -qz, qy,   qx, qy, qz );

    res = dRbydq0 * n;  //check if cv::matMulDeriv cannot do the work for us, here and in other places..
    hconcat(res, dRbydqx * n, res);
    hconcat(res, dRbydqy * n, res);
    hconcat(res, dRbydqz * n, res);  //now it should be a 3 row 4 column matrix
}

void EKF::jacob_undistor_fm(Point2d coor, Mat & res)
{
    //undistortion Jacobian stuff (Real-Time 3D SLAM with Wide-Angle Vision, Andrew J. Davison, Yolanda Gonzalez Cid and Nobuyuki Kita, IAV 2004.)
    double d = 0.0056;  //this d is probably wildly inaccurate!!!!  Also, different dx and dy might be good!
    double xd = (coor.x - K.at<double>(0,2)) * d;
    double yd = (coor.y - K.at<double>(1,2)) * d;
    double rd2 = xd*xd+yd*yd, rd4 = rd2 * rd2;
    double k1 = distCoef.at<double>(0,0), k2 = distCoef.at<double>(0,1);
    res.at<double>(0,0) = (1 + k1*rd2 + k2*rd4) + (coor.x - K.at<double>(0,2)) * (k1 + 2*k2*rd2) * (2*(coor.x - K.at<double>(0,2))*d*d);
    res.at<double>(1,1) = (1 + k1*rd2 + k2*rd4) + (coor.y - K.at<double>(1,2)) * (k1 + 2*k2*rd2) * (2*(coor.y - K.at<double>(1,2))*d*d);
    res.at<double>(0,1) = (coor.x - K.at<double>(0,2)) * (k1 + 2*k2*rd2) * (2*(coor.y - K.at<double>(1,2))*d*d);
    res.at<double>(1,0) = (coor.y - K.at<double>(1,2)) * (k1 + 2*k2*rd2) * (2*(coor.x - K.at<double>(0,2))*d*d);

    //did i do matrix coordinates right?
    //Mat dhu_dhd = (Mat_<double>(2, 2) << uu_ud, uu_vd, vu_ud, vu_vd);

}

void EKF::mapManagement( Mat & frame )
{
    step++;

    deleteFeatures();

    addAndUpdateFeatures(frame);

}

void EKF::convertToCartesian()
{
    float linearity_index_threshold = 0.1;

    int position = 13;

    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if (it->cartesian)
        {
            position = position + 3;
            continue; // skip to next feature if this one is already cartesian
        }

        float std_rho = sqrt( p_k_k.at<double>(position + 5,position + 5) );
        float rho = x_k_k.at<double>(position + 5);
        float std_d = std_rho / (rho*rho);

        float theta = x_k_k.at<double>(position + 3);
        float phi = x_k_k.at<double>(position + 4);
        float cphi = cos(phi);

        double midata[] = {cphi * sin(theta), -sin(phi), cphi*cos(theta)};
        Mat mi = Mat(3,1, CV_64F, midata).clone();  //right orientation? should be column vector
        Mat x_c1 = x_k_k.colRange(Range(position, position + 3)); //is this +2 right? should become a 3-vector
        Mat x_c2 = x_k_k.colRange(Range(0,3)); //prolly need to put 3 in both those ranges

        //this should give the cartesian equivalent of the inverse parametrization used so far
        double pdata[] = {x_k_k.at<double>(position) + midata[0]/rho,
                          x_k_k.at<double>(position + 1) + midata[1]/rho,
                          x_k_k.at<double>(position + 2) + midata[2]/rho
                         };
        Mat p = Mat(1,3, CV_64F, pdata).clone();

        float d_c2p = norm(p - x_c2); //this will prolly crash because p is 3 elements and x_c2 only 2 (should be 3 though!)

        //parallax
        Mat p1 = p - x_c1;
        Mat p2 = p - x_c2;

        Mat cos_alpham = (p1 * p2.t()) / (norm(p1) * norm(p2) );
        float cos_alpha = cos_alpham.at<double>(0,0);

        float linearity_index = 4 * std_d * cos_alpha / d_c2p;

        if (linearity_index < linearity_index_threshold)  //convert to cartesion
        {

            //change x_k_k
            Mat newx1, newx2;
            int xsize = x_k_k.size().width;

            hconcat(x_k_k.colRange(Range(0,position)), p, newx1 );
            if ((position + 6) != xsize) //if it is not the last feature we are deleting
            {
                hconcat(newx1, x_k_k.colRange(Range(position + 6, xsize)),newx2 ); //also add the rest
            }
            x_k_k = newx2.clone();

            //change p_k_k
            Mat dm_dtheta = (Mat_<double>(3, 1) << cos(phi)*cos(theta), 0, -cos(phi)*sin(theta));
            Mat dm_dphi = (Mat_<double>(3, 1) << -sin(phi)*sin(theta), -cos(phi), -sin(phi)*cos(theta));
            Mat J = Mat::eye(3, 3, CV_64F);
            hconcat(J, dm_dtheta / rho, J);
            hconcat(J, dm_dphi / rho, J);
            hconcat(J, -mi / (rho * rho), J);

            Mat J_all = Mat::eye(xsize - 3, xsize, CV_64F);
            Mat sub = J_all.colRange(position, position + 6).rowRange(position, position + 3);
            J.copyTo(sub);

            Mat J_all_t;
            transpose(J_all, J_all_t);
            p_k_k = J_all * p_k_k * J_all_t;  //now we should have a smaller p_k_k

            it->cartesian = true;
            position = position + 3;

            return; //for the moment, only convert one feature per step (sorry!)
        }
        else  //leave feature
        {
            position = position + 6;
        }

    }

}


Mat EKF::quaternion2rotmatrix(Mat & quat)  //only works on <double> Mat
{
    double q1 = quat.at<double>(0);
    double q2 = quat.at<double>(1);
    double q3 = quat.at<double>(2);
    double q4 = quat.at<double>(3);
    Mat res = (Mat_<double>(3, 3) << q1*q1 + q2*q2 - q3*q3 - q4*q4, 2 * (q2*q3 - q1*q4), 2 * (q4*q2 + q1*q3),
               2 * (q2*q3 + q1*q4), q1*q1 -q2*q2 + q3*q3 - q4*q4, 2 * (q3*q4 - q1*q2),
               2 * (q4*q2 - q1*q3), 2 * (q3*q4 + q1 * q2), q1*q1 - q2*q2 - q3*q3 + q4*q4 );
    return res;
}

void EKF::ekfPrediction() //here, fill the m1 copies of p_k_k and x_k_k (the predictions)
{
    float delta_t = 1;
    //camera motion prediction (constant velocity model for now)
    Mat rW = x_k_k.colRange(Range(0,3));
    Mat qWR = x_k_k.colRange(Range(3,7));
    Mat vW = x_k_k.colRange(Range(7,10));
    Mat wW = x_k_k.colRange(Range(10,13));
    x_k_km1 = rW + vW * delta_t;
    Mat q2;
    double theta = norm(wW*delta_t);
    if (theta < 0.00001)  //we can do quaternion stuff better with Eigen!
        q2 = (Mat_<double>(4,1) << 1,0,0,0);
    else
        q2 = (Mat_<double>(4,1) << cos(theta / 2) , wW.at<double>(0) * sin(theta / 2) * delta_t / theta,
              wW.at<double>(1) * sin(theta / 2) * delta_t / theta, wW.at<double>(2) * sin(theta / 2) * delta_t / theta);

    Mat qprod(1,4,CV_64F);
    multiplyQuaternion(qWR, q2, qprod);
    hconcat(x_k_km1, qprod, x_k_km1);
    hconcat(x_k_km1, vW, x_k_km1);
    hconcat(x_k_km1, wW, x_k_km1);

    //add features prediction (just let them stay at the same place)
    int sizex = x_k_k.size().width;
    if (sizex > 13)
        hconcat(x_k_km1, x_k_k.colRange(Range(13, sizex)) , x_k_km1);

    //and predict p_k_km1

    //state transition equation derivatives
    Mat F = Mat::eye(13,13,CV_64F);

    double q2r = q2.at<double>(0), q2x = q2.at<double>(1), q2y = q2.at<double>(2), q2z = q2.at<double>(3);
    Mat dq3bydq2 = (Mat_<double>(4,4) << q2r, -q2x, -q2y, -q2z, q2x, q2r, q2z, -q2y, q2y, -q2z, q2r, q2x, q2z, q2y, -q2x, q2r);
    Mat tmp = F(Rect(3,3,4,4));
    dq3bydq2.copyTo(tmp);
    tmp = F(Rect(7,0,3,3));
    Mat eye = Mat::eye(3,3,CV_64F);
    eye.copyTo(tmp);
    double q1r = qWR.at<double>(0), q1x = qWR.at<double>(1), q1y = qWR.at<double>(2), q1z = qWR.at<double>(3);
    Mat dq3bydq1 = (Mat_<double>(4,4) << q1r, -q1x, -q1y, -q1z, q1x, q1r, -q1z, q1y, q1y, q1z, q1r, -q1x, q1z, -q1y, q1x, q1r);
    Mat res;
    dqomegadtbydomega(wW,delta_t,res);
    Mat prod = dq3bydq1 * res;
    tmp = F(Rect(10,3,3,4));
    prod.copyTo(tmp);  //F is finally done
    //state noise
    double linear = (sigma_a*delta_t)*(sigma_a*delta_t);
    double angular = (sigma_alpha*delta_t)*(sigma_alpha*delta_t);
    Mat Pn = Mat::zeros(6,6,CV_64F);
    Pn.at<double>(0,0) = linear;
    Pn.at<double>(1,1) = linear;
    Pn.at<double>(2,2) = linear;
    Pn.at<double>(3,3) = angular;
    Pn.at<double>(3,3) = angular;
    Pn.at<double>(3,3) = angular;

    //calculate Q
    Mat G = Mat::zeros(13,6,CV_64F); //hey! dimensions turned around here or down there? argh! (height width here)
    tmp = G(Rect(0,7,3,3));
    eye.copyTo(tmp);
    tmp = G(Rect(3,10,3,3));
    eye.copyTo(tmp);
    tmp = G(Rect(0,0,3,3));
    Mat eye2 = Mat::eye(3,3,CV_64F) * delta_t;
    eye2.copyTo(tmp);
    tmp = G(Rect(3,3,3,4));
    prod.copyTo(tmp);
    Mat Q = G*Pn*G.t();

    //assemble p_k_km1
    int sizep = p_k_k.size().width;
    p_k_km1 = F * p_k_k.rowRange(Range(0,13)).colRange(Range(0,13)) * F.t() + Q;
    if (sizep > 13)
    {
        hconcat(p_k_km1, F * p_k_k.rowRange(Range(0,13)).colRange(Range(13, sizep)), p_k_km1);
        Mat bottom = p_k_k.rowRange(Range(13, sizep)).colRange(Range(0,13)) * F.t();
        hconcat(bottom, p_k_k.rowRange(Range(13, sizep)).colRange(Range(13,sizep)) , bottom);
        vconcat(p_k_km1, bottom, p_k_km1); //done!
    }

}

void EKF::dqomegadtbydomega(const Mat & wW, const double delta_t, Mat & res)
{
    double omegamod = norm(wW);
    res = Mat::zeros(4,3,CV_64F);
    res.at<double>(0,0) = dq0_by_domegaA(wW.at<double>(0), omegamod, delta_t);
    res.at<double>(0,1) = dq0_by_domegaA(wW.at<double>(1), omegamod, delta_t);
    res.at<double>(0,2) = dq0_by_domegaA(wW.at<double>(2), omegamod, delta_t);
    res.at<double>(1,0) = dqA_by_domegaA(wW.at<double>(0), omegamod, delta_t);
    res.at<double>(1,1) = dqA_by_domegaB(wW.at<double>(0), wW.at<double>(1), omegamod, delta_t);
    res.at<double>(1,2) = dqA_by_domegaB(wW.at<double>(0), wW.at<double>(2), omegamod, delta_t);
    res.at<double>(2,0) = dqA_by_domegaB(wW.at<double>(1), wW.at<double>(0), omegamod, delta_t);
    res.at<double>(2,1) = dqA_by_domegaA(wW.at<double>(1), omegamod, delta_t);
    res.at<double>(2,2) = dqA_by_domegaB(wW.at<double>(1), wW.at<double>(2), omegamod, delta_t);
    res.at<double>(3,0) = dqA_by_domegaB(wW.at<double>(2), wW.at<double>(0), omegamod, delta_t);
    res.at<double>(3,1) = dqA_by_domegaB(wW.at<double>(2), wW.at<double>(1), omegamod, delta_t);
    res.at<double>(3,2) = dqA_by_domegaA(wW.at<double>(2), omegamod, delta_t);

}

double EKF::dq0_by_domegaA(double omegaA, double omega, double delta_t)
{
    return (-delta_t / 2.0)*(omegaA / omega)*sin(omega * delta_t / 2.0);
}

double EKF::dqA_by_domegaA(double omegaA, double omega, double delta_t)
{
    return  (delta_t / 2.0) * omegaA * omegaA / (omega * omega) * cos(omega * delta_t / 2.0)
            + (1.0 / omega) * (1.0 - omegaA * omegaA / (omega * omega)) * sin(omega * delta_t / 2.0);
}

double EKF::dqA_by_domegaB(double omegaA, double omegaB, double omega, double delta_t)
{
    return (omegaA*omegaB / (omega*omega)) * ( (delta_t / 2.0) * cos(omega*delta_t / 2.0) - (1.0 / omega)*sin(omega * delta_t / 2.0) );
}

void EKF::multiplyQuaternion(const Mat& q1,const Mat& q2, Mat& q) //only works on <double> Mat
{
    // First quaternion q1 (x1 y1 z1 r1)
    const double x1=q1.at<double>(1);
    const double y1=q1.at<double>(2);
    const double z1=q1.at<double>(3);
    const double r1=q1.at<double>(0);

    // Second quaternion q2 (x2 y2 z2 r2)
    const double x2=q2.at<double>(1);
    const double y2=q2.at<double>(2);
    const double z2=q2.at<double>(3);
    const double r2=q2.at<double>(0);


    q.at<double>(1)=x1*r2 + r1*x2 + y1*z2 - z1*y2;   // x component
    q.at<double>(2)=r1*y2 - x1*z2 + y1*r2 + z1*x2;   // y component
    q.at<double>(3)=r1*z2 + x1*y2 - y1*x2 + z1*r2;   // z component
    q.at<double>(0)=r1*r2 - x1*x2 - y1*y2 - z1*z2;   // r component
}

void EKF::searchICmatches(Mat & frame)
{

    //predict feature locations (note, we did this before while adding features!
    //however, we now use EKF prediction to calculate predicted points instead of actual points;
    //thus, it probably wasnt really necessary while adding features, as we can use last measured location there
    // (only if they were measured in fact though :-( )
    int position = 13;
    Mat t_wc = x_k_km1.colRange(Range(0, 3));  //t_wc
    Mat quat = x_k_km1.colRange(Range(3,7));
    Mat rotMat = quaternion2rotmatrix(quat); //r_wc

    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        Mat hrl;
        Mat yi;
        Mat distvect;

        if (it->cartesian)
        {
            yi = x_k_km1.colRange(position, position + 3);
            Mat r_cw;
            invert(rotMat, r_cw);  // its possible to check return value if necessary; also, can we directly use it in projectPoints?
            distvect = yi.t() - t_wc.t();
            hrl = r_cw * distvect; //this should give point in camera coordinates.. does it?

            position = position + 3;
        }
        else
        {
            yi = x_k_km1.colRange(position, position + 3);     //actually, if i remember correctly
            double rho = x_k_km1.at<double>(0,position + 5);   //this x_k_km1 stuff, the features..
            double theta = x_k_km1.at<double>(0,position + 3); //are the same as in x_k_k (thus no need to copy?)
            double phi = x_k_km1.at<double>(0,position + 4);
            Mat mi = (Mat_<double>(3,1) << cos(phi) * sin(theta), -sin(phi), cos(phi) * cos(theta) );
            distvect = (yi.t() - t_wc.t() )*rho + mi;
            hrl = rotMat.t() * distvect;

            position = position + 6;
        }

        Mat tvec = Mat::zeros(1,3,CV_64F), rvec = Mat::zeros(1,3,CV_64F);
        vector<Point2d> projectedLocation;
        vector<Point3d> coordinates;
        coordinates.push_back(Point3d(hrl.at<double>(0),hrl.at<double>(1),hrl.at<double>(2) ));

        projectPoints( coordinates, rvec, tvec, K, distCoef, projectedLocation);

        //projectedLocation[0].x; //+= 320; //because projectPoints assumes (0,0) straight ahead // hrm, not?
        //projectedLocation[0].y;// += 240;         //probably something is going wrong...

        Mat location = (Mat_<double>(2,1) << projectedLocation[0].x , projectedLocation[0].y );

        if ( hrl.at<double>(0,3) > 0 ) //should also be in front of camera
        {
            ///TODO: probably its important that they also are within column / row range; skipped that so far, so add that after all!
            location.copyTo(it->h);

            //also calculate derivative H  ( = dh/dx|predictedstate )   2xn matrix cause h returns x and y coordinates

            Mat Hi;

            double fku = K.at<double>(0,0), fkv = K.at<double>(1,1);
            double hrlx = hrl.at<double>(0), hrly = hrl.at<double>(1), hrlz = hrl.at<double>(2);
            Mat dhu_dhrl = (Mat_<double>(2,3) << fku/hrlz, 0 , -hrlx*fku / (hrlz*hrlz), 0, fkv/hrlz, -hrly*fkv/(hrlz*hrlz) );

            Mat invdhd_dhu = Mat::zeros(2,2,CV_64F), dhd_dhu = Mat::zeros(2,2,CV_64F);
            Point2d loc;
            loc.x = location.at<double>(0);
            loc.y = location.at<double>(1);
            jacob_undistor_fm( loc , invdhd_dhu);
            invert(invdhd_dhu, dhd_dhu); //is this actuallt the same as what we calculated before in function jacob_babla ? dont think so..

            Mat dh_dhrl = dhd_dhu * dhu_dhrl; //add distortion derivative

            Mat dqbar_by_dq = (Mat_<double>(4,4) << 1,0,0,0, 0,-1,0,0, 0,0,-1,0, 0,0,0,-1);
            Mat dRqabydq;
            Mat quatconj = -quat; //is this allright? not changing original etc..?
            quatconj.at<double>(0) = quat.at<double>(0);
            dRq_times_a_by_dq(quatconj, distvect, dRqabydq ); //second argument should be column vector
            Mat dhrl_dqwr = dRqabydq * dqbar_by_dq;
            Mat dh_dqwr = dh_dhrl * dhrl_dqwr;

            if (it->cartesian)  //now parametrization specific stuff
            {

                Mat dhrl_dy;
                invert(rotMat, dhrl_dy); //calculated before but not stored (r_cw)
                Mat dhrl_drw = -dhrl_dy;

                Mat dh_drw = dh_dhrl * dhrl_drw;

                Mat dh_dxv = dh_drw;
                hconcat(dh_dxv, dh_dqwr, dh_dxv);
                hconcat(dh_dxv, Mat::zeros(2,6,CV_64F), dh_dxv);
                Mat dh_dy = dh_dhrl * dhrl_dy;

                Hi = dh_dxv;
                if (position - 3 - 13 > 0)  //amount of zeros to be added before
                    hconcat(Hi, Mat::zeros(2, position - 3 - 13, CV_64F), Hi);

                hconcat(Hi, dh_dy, Hi);

                if (p_k_km1.size().width - Hi.size().width > 0) //amount of zeros to be added after
                    hconcat(Hi, Mat::zeros(2, p_k_km1.size().width - Hi.size().width, CV_64F), Hi);
            }
            else
            {
                double lambda = x_k_km1.at<double>(position - 1);  //different parametrization here, is that right?
                double phi = x_k_km1.at<double>(position - 2);
                double theta = x_k_km1.at<double>(position - 3);

                Mat Rrw;
                invert(rotMat, Rrw);
                Mat dhrl_drw = Rrw * -lambda;

                Mat dhrl_dy = lambda * Rrw;
                Mat dmi1 = (Mat_<double>(3,1) << cos(phi)*cos(theta), 0, -cos(phi)*sin(theta) ); //column vector!
                Mat dmi_dthetai = Rrw * dmi1;
                hconcat(dhrl_dy, dmi_dthetai, dhrl_dy);
                Mat dmi2 = (Mat_<double>(3,1) << -sin(phi)*sin(theta), -cos(phi), -sin(phi)*cos(theta) ); //column vector!
                Mat dmi_dphii = Rrw * dmi2;
                hconcat(dhrl_dy, dmi_dphii, dhrl_dy);
                Mat lastpart = Rrw * (yi.t() - t_wc.t() );
                hconcat(dhrl_dy, lastpart, dhrl_dy);
                //dhrl_dy finished

                Mat dh_drw = dh_dhrl * dhrl_drw;

                Mat dh_dxv = dh_drw;
                hconcat(dh_dxv, dh_dqwr, dh_dxv);
                hconcat(dh_dxv, Mat::zeros(2,6,CV_64F), dh_dxv);
                Mat dh_dy = dh_dhrl * dhrl_dy;

                Hi = dh_dxv;
                if (position - 6 - 13 > 0)  //amount of zeros to be added before
                    hconcat(Hi, Mat::zeros(2, position - 6 - 13, CV_64F), Hi);

                hconcat(Hi, dh_dy, Hi);

                if (p_k_km1.size().width - Hi.size().width > 0) //amount of zeros to be added after
                    hconcat(Hi, Mat::zeros(2, p_k_km1.size().width - Hi.size().width, CV_64F), Hi);

            }

            Hi.copyTo(it->H);
            //finally, H calculated

            //now also calculate S
            Mat Si = Hi * p_k_km1 * Hi.t() + it->R;
            Si.copyTo(it->S);
        }

    }

    //warp patches according to predicted motion (predict_features_appearance)
///TODO


    //Find correspondences in the search regions using normalized cross-correlation
    double correlation_threshold = 0.8;
    double chi_095_2 = 5.9915;

    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if (it->h.empty()) continue; //apparently, this feature was not predicted

        Scalar TSS = trace(it->S), DSS = determinant(it->S);
        double TS = TSS.val[0];
        double DS = DSS.val[0];
        double underroot = TS*TS/4 - DS;
        if (underroot < 0) continue; //something is very wrong in that case

        double EigS1 = TS / 2 + sqrt(underroot);
        double EigS2 = TS / 2 - sqrt(underroot);

///(this catches everything? s/t wrong there!!)
        //if ((EigS1 > 100) || (EigS2 > 100)) continue; //do not search if ellipse is too big

        Mat invS;
        invert(it->S, invS);

        //everything OK, lets search!

        ///TODO we can do much better here by searching only in 95% region, as in matlab; for now, I simply used openCV template matching
        ///over a rectangle
        /// and of course used warped patch instead of patch stored in beginning

        Mat patch_when_matching = it->patch_when_initialized.rowRange(Range(14,26)).colRange(Range(14,26));
        patch_when_matching.copyTo(it->patch_when_matching); //make matching patch small (to be removed eventually)

        double half_search_region_size_x = ceil(2*sqrt(it->S.at<double>(0,0)));
        double half_search_region_size_y = ceil(2*sqrt(it->S.at<double>(1,1)));

        int xmin = it->h.at<double>(0) - half_search_region_size_x - it->half_patch_size_when_matching;
        int xmax = it->h.at<double>(0) + half_search_region_size_x + it->half_patch_size_when_matching;
        int ymin = it->h.at<double>(1) - half_search_region_size_y - it->half_patch_size_when_matching;
        int ymax = it->h.at<double>(1) + half_search_region_size_y + it->half_patch_size_when_matching;

        if (xmin < 0) xmin = 0; //check if we're not going over border
        if (ymin < 0) ymin = 0;
        if (xmax >= frame.cols) xmax = frame.cols - 1;
        if (ymax >= frame.rows) ymax = frame.rows - 1;

        if ((ymax - ymin < patch_when_matching.rows) || (xmax - xmin < patch_when_matching.cols) ) continue; //search area is not big enough, because feature is near edge

        Mat searchArea = frame(Rect(xmin, ymin, xmax - xmin, ymax - ymin));
        Mat result( searchArea.cols - patch_when_matching.cols + 1, searchArea.rows - patch_when_matching.rows + 1, CV_32FC1);
        matchTemplate( searchArea, patch_when_matching, result, 3 ); //3 for normalized crosscorrelation //ekfmonoslam might use 5 instead?!
        normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() ); //this okay?

        double minVal;
        double maxVal;
        Point minLoc;
        Point maxLoc;
        minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
        double maximum_correlation = maxVal;

        if (maximum_correlation > correlation_threshold)
        {
            it->individually_compatible = true;
            Mat z = (Mat_<double>(2,1) << maxLoc.x + xmin, maxLoc.y + ymin); //this going allright?
            z.copyTo(it->z); //new positions
        }

    }

}

void EKF::ransacHypotheses()
{

    double p_at_least_one_spurious_free = 0.99;
    double threshold = sigma_image_noise;
    int hyp_iterations = 1000; // max number of iterations
    //int n_hyp = 1000;  // initial value for early stopping, will be updated
    int max_hypothesis_support = -1; // will be updated

    int num_IC_matches = 0;
    vector<int> IC_matches;
    for(int i = 0; i < features_info.size(); i++)
    {
        if (features_info[i].individually_compatible)
        {
            num_IC_matches++;
            IC_matches.push_back(i);
        }
    }

    if (num_IC_matches == 0) exit(5); //make a good exit strategy here! this situation can happen, dont want program to crash then

    for (int i = 1; i <= hyp_iterations; i++) //matlab loop works different from c loop; no stopping condition in matlab
    {
        //select a random match
        int pos = (rand() % num_IC_matches);
        int randposition = IC_matches[pos];

        //1 match EKF state update
        Mat zi = features_info[randposition].z;
        Mat hi = features_info[randposition].h;
        Mat Hi = features_info[randposition].H;

        Mat S = Hi*p_k_km1*Hi.t() + features_info[randposition].R;
        Mat invS;
        invert(S, invS);
        Mat K1 = p_k_km1*Hi.t()*invS;

        Mat xi = x_k_km1.t() + K1*(zi - hi ) ; //xi has wrong orientation now

        //compute hypothesis support (using commented method in matlab as "compute hypothesis support fast" is heavy on matlab stuff)
        //first, use xi for predicting camera measurements (done 2x before using different state vector) and store in h2
        int position = 13;
        Mat t_wc = xi.rowRange(Range(0, 3));  //t_wc
        Mat quat = xi.rowRange(Range(3,7));
        Mat rotMat = quaternion2rotmatrix(quat); //r_wc

        for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
        {
            //possible improvement; we dont have to predict all points, only those that have been measured
            Mat hrl;
            Mat yi;
            Mat distvect;

            if (it->cartesian)
            {
                yi = xi.rowRange(Range(position, position + 3));
                Mat r_cw;
                invert(rotMat, r_cw);  // its possible to check return value if necessary; also, can we directly use it in projectPoints?
                distvect = yi - t_wc;
                hrl = r_cw * distvect; //this should give point in camera coordinates.. does it?

                position = position + 3;
            }
            else
            {
                yi = xi.rowRange(Range(position, position + 3));
                double rho = xi.at<double>(position + 5);
                double theta = xi.at<double>(position + 3);
                double phi = xi.at<double>(position + 4);
                Mat mi = (Mat_<double>(3,1) << cos(phi) * sin(theta), -sin(phi), cos(phi) * cos(theta) );
                distvect = (yi - t_wc )*rho + mi;
                hrl = rotMat.t() * distvect;

                position = position + 6;
            }

            Mat tvec = Mat::zeros(1,3,CV_64F), rvec = Mat::zeros(1,3,CV_64F);
            vector<Point2d> projectedLocation;
            vector<Point3d> coordinates;
            coordinates.push_back(Point3d(hrl.at<double>(0),hrl.at<double>(1),hrl.at<double>(2) ));

            projectPoints( coordinates, rvec, tvec, K, distCoef, projectedLocation); //project this point
            //and store it in h2
            Mat location = (Mat_<double>(2,1) << projectedLocation[0].x , projectedLocation[0].y );
            location.copyTo(it->h2);

        }

        //then calculate support for this hypothesis, based on predicted measurements h2 and actual measurements z
        int hyp_support = 0;
        vector <int> inliers;
        for(int j = 0; j < features_info.size(); j++)
        {
            if (features_info[j].z.empty()) continue;
            double nu1 = features_info[j].z.at<double>(0) - features_info[j].h2.at<double>(0);
            double nu2 = features_info[j].z.at<double>(1) - features_info[j].h2.at<double>(1);
            if ( sqrt(nu1*nu1 + nu2*nu2) < threshold )
            {
                hyp_support++;
                inliers.push_back(j);
            }
        }

        if (hyp_support > max_hypothesis_support)
        {
            max_hypothesis_support = hyp_support;

            //kind if ugly way to "set as most supported hypothesis"
            for(int j = 0; j < features_info.size(); j++)
            {
                features_info[j].low_innovation_inlier = false;
            }
            for(int j = 0; j < inliers.size(); j++)
            {
                features_info[inliers[j]].low_innovation_inlier = true;
            }

            float epsilon = 1-(hyp_support/num_IC_matches);

            //this is what matlab code does; not sure if it is intended though... what is the rationale?
            if (ceil(log(1-p_at_least_one_spurious_free)/log(1-(1-epsilon))) == 0) break;
        }
    }
}

void EKF::updateLIInliers()  //calculate new x_k_k and p_k_k
{
    Mat zlist, hlist, Hlist;

    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if ( ! it->low_innovation_inlier) continue;

        if (zlist.empty())
        {
            it->H.copyTo(Hlist);
            it->z.copyTo(zlist); //
            it->h.copyTo(hlist); //are those two right orientation? should be column vector; we'll see if it goes allright later
        }
        else
        {
            vconcat(Hlist, it->H, Hlist);
            vconcat(hlist, it->h, hlist);
            vconcat(zlist, it->z, zlist);
        }
    }

    if (zlist.empty()) //in case we dont have inliers at all (first frames perhaps? does this happen at all?)
    {
        x_k_k = x_k_km1;
        p_k_k = p_k_km1;
        return;
    }


    Mat R = Mat::eye(zlist.rows , zlist.rows , CV_64F );

    //now do the actual update

    Mat S = Hlist*p_k_km1*Hlist.t() + R;  //gain
    Mat invS;
    invert(S,invS);
    Mat K1 = p_k_km1 * Hlist.t() * invS;    // the filters K is not actually updated here (correct?)

    Mat tmp = K1 * (zlist - hlist); //update state and covariance
    x_k_k = x_k_km1 + tmp.t(); //cause my x_k_k is row instead of column...
    p_k_k = p_k_km1 - K1 * S * K1.t();
    p_k_k = 0.5 * p_k_k + 0.5 * p_k_k.t();
    //commented out in matlab code:  p_k_k = ( speye(size(p_km1_k,1)) - K*H )*p_km1_k;  //why is it there?

    //normalize quaternion (is it necessary to do this every step?)
    double r = x_k_k.at<double>(3), x = x_k_k.at<double>(4), y = x_k_k.at<double>(5), z = x_k_k.at<double>(6);
    double prod = r*r+x*x+y*y+z*z;
    Mat Jnorm = 1/ (prod*sqrt(prod)) * (Mat_<double>(4,4) << x*x+y*y+z*z, -r*x, -r*y, -r*z, -x*r, r*r+y*y+z*z, -x*y, -x*z,
                                        -y*r, -y*x, r*r+x*x+z*z, -y*z, -z*r, -z*x, -z*y, r*r+x*x+y*y);

    Mat quat = x_k_k.colRange(Range(3,7));
    Mat newquat = quat / norm(quat);
    newquat.copyTo(quat);

    int end = x_k_k.cols - 7;
    Mat pkk1 = p_k_k(Rect(3,0,4,3));
    tmp = pkk1 * Jnorm.t();
    tmp.copyTo(pkk1);
    Mat pkk2 = p_k_k(Rect(0,3,3,4));
    tmp = Jnorm * pkk2;
    tmp.copyTo(pkk2);
    Mat pkk3 = p_k_k(Rect(3,3,4,4));
    tmp = Jnorm * pkk3 * Jnorm.t();
    tmp.copyTo(pkk3);
    Mat pkk4 = p_k_k(Rect(3,7,4,end));
    tmp = pkk4 * Jnorm.t();
    tmp.copyTo(pkk4);
    Mat pkk5 = p_k_k(Rect(7,3,end,4));
    tmp = Jnorm * pkk5;
    tmp.copyTo(pkk5); //should be done now..

}

void EKF::rescueHIInliers()
{
    //first the couple from hell: predict_camera_measurements and calculate_derivatives (4th time doing this stuff; put in seperate function)

    int position = 13;
    Mat t_wc = x_k_k.colRange(Range(0, 3));  //t_wc
    Mat quat = x_k_k.colRange(Range(3,7));
    Mat rotMat = quaternion2rotmatrix(quat); //r_wc

    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if ( ! it->individually_compatible || it->low_innovation_inlier ) continue;

        Mat hrl;
        Mat yi;
        Mat distvect;

        if (it->cartesian)
        {
            yi = x_k_k.colRange(position, position + 3);
            Mat r_cw;
            invert(rotMat, r_cw);  // its possible to check return value if necessary; also, can we directly use it in projectPoints?
            distvect = yi.t() - t_wc.t();
            hrl = r_cw * distvect; //this should give point in camera coordinates.. does it?

            position = position + 3;
        }
        else
        {
            yi = x_k_k.colRange(position, position + 3);     //actually, if i remember correctly
            double rho = x_k_k.at<double>(0,position + 5);   //this x_k_km1 stuff, the features..
            double theta = x_k_k.at<double>(0,position + 3); //are the same as in x_k_k (thus no need to copy?)
            double phi = x_k_k.at<double>(0,position + 4);
            Mat mi = (Mat_<double>(3,1) << cos(phi) * sin(theta), -sin(phi), cos(phi) * cos(theta) );
            distvect = (yi.t() - t_wc.t() )*rho + mi;
            hrl = rotMat.t() * distvect;

            position = position + 6;
        }

        Mat tvec = Mat::zeros(1,3,CV_64F), rvec = Mat::zeros(1,3,CV_64F);
        vector<Point2d> projectedLocation;
        vector<Point3d> coordinates;
        coordinates.push_back(Point3d(hrl.at<double>(0),hrl.at<double>(1),hrl.at<double>(2) ));

        projectPoints( coordinates, rvec, tvec, K, distCoef, projectedLocation);

        //projectedLocation[0].x; //+= 320; //because projectPoints assumes (0,0) straight ahead // hrm, not?
        //projectedLocation[0].y;// += 240;         //probably something is going wrong...

        Mat location = (Mat_<double>(2,1) << projectedLocation[0].x , projectedLocation[0].y );


        ///TODO: probably its important that they also are within column / row range; skipped that so far, so add that after all!
        location.copyTo(it->h);

        //also calculate derivative H  ( = dh/dx|predictedstate )   2xn matrix cause h returns x and y coordinates

        Mat Hi;

        double fku = K.at<double>(0,0), fkv = K.at<double>(1,1);
        double hrlx = hrl.at<double>(0), hrly = hrl.at<double>(1), hrlz = hrl.at<double>(2);
        Mat dhu_dhrl = (Mat_<double>(2,3) << fku/hrlz, 0 , -hrlx*fku / (hrlz*hrlz), 0, fkv/hrlz, -hrly*fkv/(hrlz*hrlz) );

        Mat invdhd_dhu = Mat::zeros(2,2,CV_64F), dhd_dhu = Mat::zeros(2,2,CV_64F);
        Point2d loc;
        loc.x = location.at<double>(0);
        loc.y = location.at<double>(1);
        jacob_undistor_fm( loc , invdhd_dhu);
        invert(invdhd_dhu, dhd_dhu); //is this actuallt the same as what we calculated before in function jacob_babla ? dont think so..

        Mat dh_dhrl = dhd_dhu * dhu_dhrl; //add distortion derivative

        Mat dqbar_by_dq = (Mat_<double>(4,4) << 1,0,0,0, 0,-1,0,0, 0,0,-1,0, 0,0,0,-1);
        Mat dRqabydq;
        Mat quatconj = -quat; //is this allright? not changing original etc..?
        quatconj.at<double>(0) = quat.at<double>(0);
        dRq_times_a_by_dq(quatconj, distvect, dRqabydq ); //second argument should be column vector
        Mat dhrl_dqwr = dRqabydq * dqbar_by_dq;
        Mat dh_dqwr = dh_dhrl * dhrl_dqwr;

        if (it->cartesian)  //now parametrization specific stuff
        {

            Mat dhrl_dy;
            invert(rotMat, dhrl_dy); //calculated before but not stored (r_cw)
            Mat dhrl_drw = -dhrl_dy;

            Mat dh_drw = dh_dhrl * dhrl_drw;

            Mat dh_dxv = dh_drw;
            hconcat(dh_dxv, dh_dqwr, dh_dxv);
            hconcat(dh_dxv, Mat::zeros(2,6,CV_64F), dh_dxv);
            Mat dh_dy = dh_dhrl * dhrl_dy;

            Hi = dh_dxv;
            if (position - 3 - 13 > 0)  //amount of zeros to be added before
                hconcat(Hi, Mat::zeros(2, position - 3 - 13, CV_64F), Hi);

            hconcat(Hi, dh_dy, Hi);

            if (p_k_k.size().width - Hi.size().width > 0) //amount of zeros to be added after
                hconcat(Hi, Mat::zeros(2, p_k_k.size().width - Hi.size().width, CV_64F), Hi);
        }
        else
        {
            double lambda = x_k_k.at<double>(position - 1);  //different parametrization here, is that right?
            double phi = x_k_k.at<double>(position - 2);
            double theta = x_k_k.at<double>(position - 3);

            Mat Rrw;
            invert(rotMat, Rrw);
            Mat dhrl_drw = Rrw * -lambda;

            Mat dhrl_dy = lambda * Rrw;
            Mat dmi1 = (Mat_<double>(3,1) << cos(phi)*cos(theta), 0, -cos(phi)*sin(theta) ); //column vector!
            Mat dmi_dthetai = Rrw * dmi1;
            hconcat(dhrl_dy, dmi_dthetai, dhrl_dy);
            Mat dmi2 = (Mat_<double>(3,1) << -sin(phi)*sin(theta), -cos(phi), -sin(phi)*cos(theta) ); //column vector!
            Mat dmi_dphii = Rrw * dmi2;
            hconcat(dhrl_dy, dmi_dphii, dhrl_dy);
            Mat lastpart = Rrw * (yi.t() - t_wc.t() );
            hconcat(dhrl_dy, lastpart, dhrl_dy);
            //dhrl_dy finished

            Mat dh_drw = dh_dhrl * dhrl_drw;

            Mat dh_dxv = dh_drw;
            hconcat(dh_dxv, dh_dqwr, dh_dxv);
            hconcat(dh_dxv, Mat::zeros(2,6,CV_64F), dh_dxv);
            Mat dh_dy = dh_dhrl * dhrl_dy;

            Hi = dh_dxv;
            if (position - 6 - 13 > 0)  //amount of zeros to be added before
                hconcat(Hi, Mat::zeros(2, position - 6 - 13, CV_64F), Hi);

            hconcat(Hi, dh_dy, Hi);

            if (p_k_k.size().width - Hi.size().width > 0) //amount of zeros to be added after
                hconcat(Hi, Mat::zeros(2, p_k_k.size().width - Hi.size().width, CV_64F), Hi);

        }

        Hi.copyTo(it->H);
        //finally, H calculated

        //now check if this feature is in fact a high innovation inlier
        Mat Si = it->H * p_k_k * it->H.t();
        Mat invSi;
        invert(Si, invSi);
        Mat nui = it->z - it->h;
        Mat p = nui.t() * invSi * nui;
        if (p.at<double>(0) < 5.9915)
            it->high_innovation_inlier = true;
        else
            it->high_innovation_inlier = false;
    }

}

void EKF::updateHIInliers()  //this function is identical to the LI inliers, apart from the if ( ! it->high_innovation_inlier) continue;
{
    //lets just make it one function with a parameter to set if it is low or high innovation inliers that we do...
    Mat zlist, hlist, Hlist;

    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if ( ! it->high_innovation_inlier) continue;

        if (zlist.empty())
        {
            it->H.copyTo(Hlist);
            it->z.copyTo(zlist); //
            it->h.copyTo(hlist); //are those two right orientation? should be column vector; we'll see if it goes allright later
        }
        else
        {
            vconcat(Hlist, it->H, Hlist);
            vconcat(hlist, it->h, hlist);
            vconcat(zlist, it->z, zlist);
        }
    }

    if (zlist.empty()) //in case we dont have inliers at all (first frames perhaps? does this happen at all?)
    {
        x_k_k = x_k_km1;
        p_k_k = p_k_km1;
        return;
    }


    Mat R = Mat::eye(zlist.rows , zlist.rows , CV_64F );

    //now do the actual update

    Mat S = Hlist*p_k_km1*Hlist.t() + R;  //gain
    Mat invS;
    invert(S,invS);
    Mat K1 = p_k_km1 * Hlist.t() * invS;    // the filters K is not actually updated here (correct?)

    Mat tmp = K1 * (zlist - hlist); //update state and covariance
    x_k_k = x_k_km1 + tmp.t(); //cause my x_k_k is row instead of column...
    p_k_k = p_k_km1 - K1 * S * K1.t();
    p_k_k = 0.5 * p_k_k + 0.5 * p_k_k.t();
    //commented out in matlab code:  p_k_k = ( speye(size(p_km1_k,1)) - K*H )*p_km1_k;  //why is it there?

    //normalize quaternion (is it necessary to do this every step?)
    double r = x_k_k.at<double>(3), x = x_k_k.at<double>(4), y = x_k_k.at<double>(5), z = x_k_k.at<double>(6);
    double prod = r*r+x*x+y*y+z*z;
    Mat Jnorm = 1/ (prod*sqrt(prod)) * (Mat_<double>(4,4) << x*x+y*y+z*z, -r*x, -r*y, -r*z, -x*r, r*r+y*y+z*z, -x*y, -x*z,
                                        -y*r, -y*x, r*r+x*x+z*z, -y*z, -z*r, -z*x, -z*y, r*r+x*x+y*y);

    Mat quat = x_k_k.colRange(Range(3,7));
    Mat newquat = quat / norm(quat);
    newquat.copyTo(quat);

    int end = x_k_k.cols - 7;
    Mat pkk1 = p_k_k(Rect(3,0,4,3));
    tmp = pkk1 * Jnorm.t();
    tmp.copyTo(pkk1);
    Mat pkk2 = p_k_k(Rect(0,3,3,4));
    tmp = Jnorm * pkk2;
    tmp.copyTo(pkk2);
    Mat pkk3 = p_k_k(Rect(3,3,4,4));
    tmp = Jnorm * pkk3 * Jnorm.t();
    tmp.copyTo(pkk3);
    Mat pkk4 = p_k_k(Rect(3,7,4,end));
    tmp = pkk4 * Jnorm.t();
    tmp.copyTo(pkk4);
    Mat pkk5 = p_k_k(Rect(7,3,end,4));
    tmp = Jnorm * pkk5;
    tmp.copyTo(pkk5); //should be done now..
}

void EKF::visualize(Mat & frameGray, char * fps)
{

    //full disclosure here..
    std::cout << "step " << step << std::endl;
    //std::cout << "xkk " << x_k_k << std::endl;



    Mat outFrame;

    cvtColor(frameGray, outFrame, CV_GRAY2BGR);

    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if (it->z.empty()) continue;
        Point2f p;
        p.x = it->z.at<double>(0);
        p.y = it->z.at<double>(1);
        putText(outFrame, "*", p, FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255,255)); //put FPS text

        p.x = it->h.at<double>(0);
        p.y = it->h.at<double>(1);
        putText(outFrame, "O", p, FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255,255)); //put FPS text


    }

    putText(outFrame, fps, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255,255)); //put FPS text

    imshow("det", outFrame);


}
