#include "EKF.h"
#include "time.h"

using namespace cv;

EKF::EKF() //ctor
{
    min_number_of_features = 15;
    max_number_of_features = 15;

    sigma_a = 0.007;
    sigma_alpha = 0.007;
    sigma_image_noise = 1.0;

    double eps = 1e-15;

    //init cam

    //logitech camera
    double Kdata[] = {7.2208441300095387e+02, 0, 3.1950000000000000e+02, 0, 7.2208441300095387e+02, 2.3950000000000000e+02, 0, 0, 1}; //for logitech webcam
    distCoef = (Mat_<double>(1, 4) << -0.0503375865, 0.1814439, 0, 0, -1.874947); //this is close enough for logitech webcam

    //test sequence: unibrain fire-i
    //distCoef = (Mat_<double>(1, 4) <<  -0.13, -0.06,0,0); //this is close enough for unibrain fire-i (test sequence)
    //double Kdata[] = {2.1735 / 0.0112, 0, 1.7945 / 0.0112,  0, 2.1735 / 0.0112, 1.4433 / 0.0112,  0,0,1}; //for test...

    //other/old stuff
    //double Kdata[] = {640/(2*tan(60*3.14/360)) ,0, 640/2,  0, 640/(2*tan(60*3.14/360)), 480/2,  0,0,1}; //is this OK? esp. focal lengths..
    //distCoef = (Mat_<double>(1, 4) <<  0.06333, 0.0139, 0, 0); //TODO get better distortion coefficients!
    //distCoef = (Mat_<double>(1, 4) << 0.09542387, -0.755768, 0, 0, 0.148786);

    K = Mat(3,3,CV_64F, Kdata).clone();  //fix this better! find out theory and so on...

    //init state

    xkk.setZero(13);
    xkk << 0,0,0, 1,0,0,0, 0,0,0, eps,eps,eps;
    //xkk << 1, 2, 3, 0.570563, 0.570563, -0.051482, 0.588443, 0, 0, 0, eps,eps,eps;
    pkk.setZero(13,13);
    pkk.diagonal() << eps, eps, eps, eps, eps, eps, eps, 0.025*0.025, 0.025*0.025, 0.025*0.025, 0.025*0.025, 0.025*0.025, 0.025*0.025;

    // assume linear velocity for now

    step = 1;

    logfile.open ("/dev/null/log.txt");
    //logfile.open ("log.txt");
    time_t rawtime;
    time(&rawtime);
    logfile << asctime(localtime(&rawtime));

}

EKF::~EKF()
{
    logfile.close();
}

void EKF::deleteFeatures() //This function loops over features and removes unreliable ones from feature_info and xkk and pkk
{
    logfile << "current number of features: "<< features_info.size() << std::endl;

    for(std::vector<feature>::reverse_iterator rit = features_info.rbegin(); rit != features_info.rend(); ++rit )
    {

        if ((rit->timesMeasured > 0.5*rit->timesPredicted) || (rit->timesPredicted < 5)) continue; //to next feature

        int position = rit->position;

        //std::cout << " deleting feature " << it - features_info.begin() << std::endl;
        //std::cout << "predicted: " << it->predicted << " times and measured " << it->measured << " times" << std::endl;
        int fsize;
        if (rit->cartesian)
        {
            fsize = 3;
            logfile << "deleting cartesian feature! nr " << features_info.rend() - rit << std::endl;
        }
        else
        {
            fsize = 6;
            logfile << "deleting inverse depth feature! nr " << features_info.rend() - rit << std::endl;
        }

        //std::cout << xkk << std::endl;
        if ((position + fsize) < xkk.size())
        {
            //this should do the trick
            xkk.segment(position, xkk.size() - position - fsize) = xkk.segment(position + fsize, xkk.size() - position - fsize).eval();

            pkk.block(0, position, pkk.rows(), pkk.cols() - position - fsize) = pkk.rightCols(pkk.cols() - position - fsize).eval();
            pkk.block(position, 0, pkk.rows() - position - fsize, pkk.cols()) = pkk.bottomRows(pkk.rows() - position - fsize).eval();

        }

        pkk.conservativeResize(pkk.rows() - fsize, pkk.cols() - fsize);
        xkk.conservativeResize(xkk.size() - fsize);

        // update position fields of following features
        for(std::vector<feature>::iterator it = rit.base(); it != features_info.end(); ++it)
        {
            it->position -= fsize;
        }

        // - 1 because the base() points one element further (which is why there is no +1 in the lines above)
        features_info.erase( rit.base() - 1);
        //hrm actually this is not very pretty, but seems to work even if rit = features_info.rbegin() (test to see if it actally works as expected)

    }

    logfile << "number of features after deleting: "<< features_info.size() << std::endl;

}

void EKF::addAndUpdateFeatures(Mat & frame) //works fine in first timestep
{
    int measuredFeatures = 0;
    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if (it->low_innovation_inlier || it->high_innovation_inlier)
        {
            measuredFeatures++;
            it->timesMeasured++;
        }

        if ( it->predicted ) it->timesPredicted++;

        it->predicted = false;
        it->measured = false; //the same as individually_compatible
        it->high_innovation_inlier = false;
        it->low_innovation_inlier = false;
        it->keepaway = false; //for adding new features

    }

    convertToCartesian(); //something goes wrong in here i think...

    //predict locations of old features; if in front of camera, make sure we dont initialize a feature close to it (keepaway bool)
    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if (it->updatePrediction(xkk, 1, K, distCoef)) it->keepaway = true;
    }
    ///TODO: maybe add features just after searching matches instead, so that we can use the same predicted locations

    //start adding new features
    int max_attempts = 50;
    int initialized = 0, attempts = 0;

    //note: this will initialize features even if features_info already has too many features in it, iff they are not measured this step
    while ((initialized + measuredFeatures < min_number_of_features) && ( attempts < max_attempts ) && (features_info.size() < max_number_of_features))
    {
        attempts++;

        //extract fast corners and check if it is far enough away from others (predicted from EKF)

        //make a random small box further than 21 pixels away from borders, 60x40 large
        int x = 21 + rand() % (frame.cols - 42 - 60 );
        int y = 21 + rand() % (frame.rows - 42 - 40 );
        Mat searchBox(frame, Rect(x,y,60,40) );
        //extract FAST corners here
        vector<KeyPoint> result;
        FAST(searchBox, result, 40, true);

        Point2f FASTresult;

        if (result.size() == 0) continue; //lets try again


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
        //Size winSize = Size( 5, 5 );
        //Size zeroZone = Size( -1, -1 );
        //TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 20, 0.01 );

        // Calculate the refined corner locations
        //cornerSubPix( frame, resultVector, winSize, zeroZone, criteria );

        FASTresult = resultVector[0];

        //check if feature is far enough away from existing features (minimum distance)
        bool success = true;

        for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
        {
            if (!it->keepaway) continue;
            logfile << " feature " << it - features_info.begin() << " predicted at " << it->he << std::endl;
            if ( (it->he - Eigen::Vector2f(FASTresult.x , FASTresult.y)).norm() < 20 )
            {
                success = false;
                break;
            }
        }

        if (!success) continue; //lets try again

        //add feature to info vector
        Mat patch = frame.colRange( (int) FASTresult.x - 20, (int) FASTresult.x + 21).rowRange(
                                                        (int) FASTresult.y - 20, (int) FASTresult.y + 21) ;
        Eigen::Matrix<float, 6, 13> dydxv;
        Eigen::Matrix<float, 6, 3> dydhd;
        Eigen::Vector3f n;

        feature newFeatureInfo(FASTresult, xkk.head<7>(), patch, step, xkk.size(), dydxv, dydhd, n, K, distCoef);
        //heavy lifting is done in constructor
        //NB: Should the first argument be the distorted point (as in matlab code) or the undistorted point? its probably right

        features_info.push_back(newFeatureInfo);

        //assemble new xkk
        float nx = n(0); float ny = n(1); float nz = n(2);

        xkk.conservativeResize(xkk.size() + 6 );
        xkk.tail<6>() << xkk(0), xkk(1), xkk(2), atan2(nx, nz), atan2(-ny, sqrt(nx*nx + nz*nz) ), 1;
        ///retreive this from feature constructor instead of n, as this is already calculated there

        //assemble new pkk
        Eigen::Matrix3f Pa;
        Pa << sigma_image_noise*sigma_image_noise, 0, 0, 0, sigma_image_noise*sigma_image_noise, 0, 0, 0, 1;

        int psize = pkk.cols();
        pkk.conservativeResize(psize + 6, psize + 6); //this may be slower than initializing new matrix..
        pkk.topRightCorner(psize ,6) = pkk.topLeftCorner( psize ,13) * dydxv.transpose();
        pkk.bottomLeftCorner(6, psize) = dydxv * pkk.topLeftCorner( 13 , psize );
        pkk.bottomRightCorner<6,6>() = dydxv * pkk.topLeftCorner<13,13>() * dydxv.transpose() + dydhd * Pa * dydhd.transpose();

        initialized++;

        //logfile << "added feature at " << FASTresult << " , undistorted "<< undistorted[0] << std::endl;

    }

    logfile << std::endl << "xkk after adding features " << xkk << std::endl ;
    logfile << std::endl << "pkk after adding features " << pkk << std::endl;

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

    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if (it->cartesian) continue; // skip to next feature if this one is already cartesian

        float std_rho = sqrt( pkk(it->position + 5,it->position + 5) );
        float rho = xkk(it->position + 5);
        float std_d = std_rho / (rho*rho);

        float theta = xkk(it->position + 3);
        float phi = xkk(it->position + 4);

        Eigen::Vector3f mi(cos(phi) * sin(theta), -sin(phi), cos(phi)*cos(theta));

        Eigen::Vector3f xc1 = xkk.segment<3>(it->position);
        Eigen::Vector3f xc2 = xkk.head<3>();

        Eigen::Vector3f p = xc1 + mi / rho; //this should give the cartesian equivalent of the inverse parametrization used so far
        //parallax
        Eigen::Vector3f p1 = p - xc1;
        Eigen::Vector3f p2 = p - xc2;

        //logfile << std::endl << "feature " << it - features_info.begin() << " location is " << p << std::endl ;

        float d_c2p = (p - xc2).norm(); //distance camera to point
        //logfile << "distance " << d_c2p << std::endl;

        float cos_alpha = p1.dot(p2) / (p1.norm() * p2.norm() );

        //logfile << "cos alpha " << cos_alpha << std::endl;
        //logfile << "old xkk " << xkk << std::endl;
        //logfile << "old pkk " << pkk << std::endl;

        float linearity_index = 4 * std_d * cos_alpha / d_c2p;

        //std::cout << linearity_index << std::endl;
        //linearity_index = 0.01;

        if (linearity_index > linearity_index_threshold) continue; //skip to next feature if linearity is not below threshold

        logfile << " converting feature " << it - features_info.begin() << " to cartesian!" << std::endl;
        //change x_k_k

        if ((it->position + 6) < xkk.size())
        {
            //this should do the trick
            xkk.segment(it->position + 3, xkk.size() - it->position - 6) = xkk.segment(it->position + 6, xkk.size() - it->position - 6).eval();
        }
        xkk.segment(it->position, 3) = p;
        xkk.conservativeResize(xkk.size() - 3);

        //logfile << "new xkk " << xkk << std::endl;

        //change p_k_k
        Eigen::MatrixXf Jall(pkk.rows() - 3, pkk.cols());
        Jall.setZero();
        Jall.topLeftCorner( it->position , it->position ).setIdentity();
        Jall.bottomRightCorner( pkk.rows() - it->position - 6, pkk.rows() - it->position - 6 ).setIdentity();
        Eigen::Vector3f dmdtheta( cos(phi)*cos(theta), 0, -cos(phi)*sin(theta) );
        Eigen::Vector3f dmdphi( -sin(phi)*sin(theta), -cos(phi), -sin(phi)*cos(theta) );
        Eigen::Vector3f m( cos(phi) * sin(theta), -sin(phi), cos(phi)*cos(theta) );
        Jall.block<3,6>(it->position , it->position) << Eigen::Matrix3f::Identity() , dmdtheta / rho , dmdphi / rho , -m / (rho * rho);

        pkk = Jall * pkk * Jall.transpose(); //now we should have a smaller p_k_k
        //logfile << "new pkk " << pkk << std::endl;

        it->cartesian = true;
        // update position fields of following features
        for(std::vector<feature>::iterator it2 = it + 1; it2 != features_info.end(); ++it2)
        {
            it2->position -= 3;
        }
        //return; //for the moment, only convert one feature per step (sorry!) //uhhh why again?
    }

}

void EKF::ekfPrediction() //here, fill the m1 copies of p_k_k and x_k_k (the predictions)  //works perfectly now
{ ///prolly something wrong here..

    float delta_t = 1;
    //camera motion prediction (constant velocity model for now)
    double theta = xkk.segment<3>(10).norm() * delta_t;
    Eigen::Quaternionf q4;
    if (theta < 0.00001)  //we can do quaternion stuff better with Eigen!
        q4 = Eigen::Quaternionf(1,0,0,0);
    else
        q4 = Eigen::Quaternionf(cos(theta / 2) , xkk(10) * sin(theta / 2) * delta_t / theta,
              xkk(11) * sin(theta / 2) * delta_t / theta, xkk(12) * sin(theta / 2) * delta_t / theta);

    xkkm1 = xkk;
    Eigen::Quaternionf q3(xkk(3) , xkk(4), xkk(5), xkk(6) );
    Eigen::Quaternionf q5(q3*q4); //right order?
    xkkm1.head<7>() << xkk.segment<3>(0) + xkk.segment<3>(7) * delta_t , q5.w(), q5.x(), q5.y(), q5.z(); //the rest actually stays the same

    //and predict pkkm1

    //state transition equation derivatives
    float omod = theta / delta_t;
    Eigen::Matrix<float, 4, 3> res2;
    res2 << dq0_by_domegaA(xkk(10), omod, delta_t) , dq0_by_domegaA(xkk(11), omod, delta_t) ,
    dq0_by_domegaA(xkk(12), omod, delta_t) , dqA_by_domegaA(xkk(10), omod, delta_t) ,
    dqA_by_domegaB(xkk(10), xkk(11), omod, delta_t) , dqA_by_domegaB(xkk(10), xkk(12), omod, delta_t) ,
    dqA_by_domegaB(xkk(11), xkk(10), omod, delta_t) , dqA_by_domegaA(xkk(11), omod, delta_t) ,
    dqA_by_domegaB(xkk(11), xkk(12), omod, delta_t) , dqA_by_domegaB(xkk(12), xkk(10), omod, delta_t) ,
    dqA_by_domegaB(xkk(12), xkk(11), omod, delta_t) , dqA_by_domegaA(xkk(12), omod, delta_t);
    Eigen::Matrix<float, 4, 4> dq3bydq1_2;
    float q1r = xkk(3), q1x = xkk(4), q1y = xkk(5), q1z = xkk(6);
    dq3bydq1_2 << q1r, -q1x, -q1y, -q1z, q1x, q1r, -q1z, q1y, q1y, q1z, q1r, -q1x, q1z, -q1y, q1x, q1r;
    Eigen::Matrix<float, 4, 3> prod2(dq3bydq1_2 * res2);

    Eigen::MatrixXf Fn(13,13);
    Fn.setIdentity();
    float q2r = q4.w(), q2x = q4.x(), q2y = q4.y(), q2z = q4.z();
    Fn.block<4,4>(3,3) << q2r, -q2x, -q2y, -q2z, q2x, q2r, q2z, -q2y, q2y, -q2z, q2r, q2x, q2z, q2y, -q2x, q2r;
    Fn.block<3,3>(0,7).setIdentity();
    Fn.block<4,3>(3,10) << prod2;

    //state noise
    double linear = (sigma_a*delta_t)*(sigma_a*delta_t);
    double angular = (sigma_alpha*delta_t)*(sigma_alpha*delta_t);

    Eigen::Matrix<float, 6, 6> P;
    P.setZero();
    P.diagonal() << linear, linear, linear, angular, angular, angular;

    //calculate Q
    Eigen::MatrixXf Gn(13,6);
    Gn.setZero();
    Gn.topLeftCorner<3,3>() <<  Eigen::Matrix3f::Identity() * delta_t;
    Gn.block<3,3>(7,0).setIdentity();
    Gn.block<4,3>(3,3) << prod2;
    Gn.bottomRightCorner<3,3>().setIdentity();

    Eigen::MatrixXf Qn;
    Qn = Gn * P * Gn.transpose();

    pkkm1 = pkk;  //this is not really the most efficient way probably, as part is copied unnecessarily.. improve?
    pkkm1.topRows(13) = Fn * pkkm1.topRows(13); //eigen assumes aliasing, thats ok here
    pkkm1.leftCols(13) = pkkm1.leftCols(13) * Fn.transpose();
    pkkm1.topLeftCorner(13,13) += Qn;

    logfile << std::endl << "xkkm1 " << xkkm1 << std::endl;
    logfile << std::endl << "pkkm1 "<< pkkm1 << std::endl;

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

void EKF::searchICmatches(Mat & frame)   //calculating derivatives of inversedepth works!
{

    //however, we now use EKF prediction to calculate predicted points instead of actual points;
    //thus, it probably wasnt really necessary while adding features, as we can use last measured location there
    // (only if they were measured in fact though :-( )


    //warp patches according to predicted motion (predict_features_appearance)
///TODO

    //Find correspondences in the search regions using normalized cross-correlation
    double correlation_threshold = 0.8;
    //double chi_095_2 = 5.9915;

    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        logfile << " feature " << it - features_info.begin();

        //predict feature locations and check if it should be visible
        if (! it->updatePredictionAndDerivatives(xkkm1, K, distCoef) ) continue; //go to next feature if this one is not in front of camera
        if ( (it->he(0) < -10) || (it->he(0) > frame.cols + 10) || (it->he(1) < -10) || (it->he(1) > frame.rows + 10) ) continue;
            //go to next feature if this one is not in the frame

        it->predicted = true;

        //now also calculate S
        it->Se = it->He * pkk * it->He.transpose() + it->Re;

        double TS = it->Se.trace();
        double DS = it->Se.determinant();
        double underroot = TS*TS/4 - DS;
        if (underroot < 0) continue; //something is very wrong in that case

        double EigS1 = TS / 2 + sqrt(underroot); //eigenvalues of 2x2 matrix
        double EigS2 = TS / 2 - sqrt(underroot);

        //note, this is higher than in matlab because it depends on amount of pixels
        if ((EigS1 > 200) && (EigS2 > 200) && (step > 40))
        {

            std::cout << " uncertainty too large: " << EigS1 << " " << EigS2 << std::endl;
            continue; //do not search if ellipse is too big
        }

        //everything OK, lets search!

        ///TODO we can do much better here by searching only in 95% region, as in matlab; for now, I simply used openCV template matching
        ///over a rectangle
        /// and of course used warped patch instead of patch stored in beginning

        Mat patch_when_matching = it->patch_when_initialized.rowRange(Range(14,26)).colRange(Range(14,26));
        patch_when_matching.copyTo(it->patch_when_matching); //make matching patch small (to be removed eventually)

        double half_search_region_size_x = ceil(2*sqrt(it->Se(0,0) ) );
        double half_search_region_size_y = ceil(2*sqrt(it->Se(1,1) ) );

        int xmin = it->he(0) - half_search_region_size_x - it->half_patch_size_when_matching;
        int xmax = it->he(0) + half_search_region_size_x + it->half_patch_size_when_matching;
        int ymin = it->he(1) - half_search_region_size_y - it->half_patch_size_when_matching;
        int ymax = it->he(1) + half_search_region_size_y + it->half_patch_size_when_matching;

        if (xmin < 0) xmin = 0; //check if we're not going over border
        if (ymin < 0) ymin = 0;
        if (xmax >= frame.cols) xmax = frame.cols - 1;
        if (ymax >= frame.rows) ymax = frame.rows - 1;

        if ((ymax - ymin < patch_when_matching.rows) || (xmax - xmin < patch_when_matching.cols) ) continue; //search area is not big enough, because feature is near edge

        Mat searchArea = frame(Rect(xmin, ymin, xmax - xmin, ymax - ymin));
        it->lastSearchArea = Rect(xmin, ymin, xmax - xmin, ymax - ymin);

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
            it->measured = true;
            it->ze = Eigen::Vector2f(maxLoc.x + xmin + 6, maxLoc.y + ymin + 6); //6 is half the size of the patch

            //std::cout << "match found at " << z << std::endl;
            logfile << " found at " << it->ze << std::endl;
        }
        else
        {
            logfile << " not found" << std::endl;
        }

    }

}

void EKF::ransacHypotheses()
{
    ///TODO just loop over all IC matches instead of over 100 random IC matches

    double p_at_least_one_spurious_free = 0.99;
    double threshold = sigma_image_noise;
    int hyp_iterations = 100; // max number of iterations
    //int hyp_iterations = 1000;
    int max_hypothesis_support = -1; // will be updated

    int num_IC_matches = 0;
    vector<int> IC_matches;
    for(int i = 0; i < features_info.size(); i++)
    {
        if (features_info[i].measured)
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
        Eigen::Vector2f zi = features_info[randposition].ze;
        Eigen::Vector2f hi = features_info[randposition].he;
        Eigen::MatrixXf Hi = features_info[randposition].He;

        Eigen::Matrix2f S = Hi*pkkm1*Hi.transpose() + features_info[randposition].Re;
        Eigen::MatrixXf K1 = pkkm1 * Hi.transpose() * S.inverse();

        Eigen::VectorXf xi = xkkm1 + K1 * (zi - hi ) ;

        //compute hypothesis support (using commented method in matlab as "compute hypothesis support fast" is heavy on matlab stuff)
        //based on predicted measurements h2 and actual measurements z
        int hyp_support = 0;
        vector <int> inliers;
        for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
        {
            //first, use xi for predicting camera measurements if feature was measured,
            //and store in h2e not to interfere with following 1-point RANSAC hypotheses
            if (!it->measured) continue;
            it->updatePrediction(xi, 2, K, distCoef);

            if ( (it->ze - it->h2e).norm() < threshold )
            {
                hyp_support++;
                inliers.push_back(it - features_info.begin()); //just the index of the feature in the vector
            }
        }

        if (hyp_support <= max_hypothesis_support) continue; //not good enough, try a new hypothesis

        //std::cout << "new max hypothesis" << std::endl;
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
        if (ceil(log(1-p_at_least_one_spurious_free)/log(1-(1-epsilon))) == 0) //it doesnt always do this. (mostly not, same in Matlab)
        {
                /*
                   std::cout << "hypothesis accepted with the following predictions: " << std::endl;
                   for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
                   {
                       std::cout << " feature " << it - features_info.begin() << " predicted at " << it->h2;
                       if (!it->z.empty())
                       {
                           std::cout << " , measured " << it->z;
                           std::cout << (it->low_innovation_inlier ? " LI inlier": " HI inlier") << std::endl;
                       }
                       else
                       {
                           std::cout << " not measured " << std::endl;
                       }

                   }*/

            break;
        }


    }
}

void EKF::updateLIInliers()  //calculate new xkk and pkk // works as intended
{
    int inlierCount = 0;
    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if ( it->low_innovation_inlier) inlierCount++;
    }

    if (inlierCount == 0) //in case we dont have inliers at all (first frames perhaps? does this happen at all?)
    {
        xkk = xkkm1;
        pkk = pkkm1;
        return;
    }

    Eigen::VectorXf hList(inlierCount * 2);
    Eigen::VectorXf zList(inlierCount * 2);
    Eigen::MatrixXf HList(inlierCount * 2, pkkm1.rows() );

    int counter = 0;
    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if ( ! it->low_innovation_inlier) continue;
        HList.block(counter * 2, 0, 2, pkkm1.rows()) = it->He ;
        hList.segment(counter * 2, 2) = it->he;
        zList.segment(counter * 2, 2) = it->ze;
        counter++;
    }

    //now do the actual update
    //gain
    Eigen::MatrixXf S1;
    S1.noalias() = HList * pkkm1 * HList.transpose() + Eigen::MatrixXf::Identity( HList.rows(), HList.rows() );
    //the filters K is not actually updated here (correct?)
    Eigen::MatrixXf K2;
    K2.noalias() = pkkm1 * HList.transpose() * S1.inverse();  //assumes aliasing here, not necessary (and line above too)
    //this inverse is a shortcut, check if its always OK
    //update state and covariance
    xkk.noalias() = xkkm1 + K2 * (zList - hList);
    pkk.noalias() = pkkm1 - K2 * S1 * K2.transpose();
    pkk = 0.5 * pkk;
    pkk += pkk.transpose().eval(); // to solve aliasing issue (which is _not_ assumed here b/c no matmul)
    //commented out in matlab code: p_k_k = ( speye(size(p_km1_k,1)) - K*H )*p_km1_k;  //why is it there?

    //normalize quaternion (is it necessary to do this every step?)
    normalizeQuaternion();

    logfile << "xkk after RANSAC LI update " << xkk << std::endl;
    logfile << "pkk after RANSAC LI update " << pkk << std::endl;

}

void EKF::rescueHIInliers()
{
    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        //calculate he and He for potential HI inliers (features that are measured but are not low innovation)
        if ( (! it->measured) || it->low_innovation_inlier ) continue;
        if ( ! it->updatePredictionAndDerivatives(xkk, K, distCoef) ) continue;
        //this only returns false (and thus continues) if feature is predicted to be not in front of camera; should really not happen

        //now check if this feature is in fact a high innovation inlier
        Eigen::Matrix2f Sie = it->He * pkk * it->He.transpose();
        Eigen::Vector2f nuie(it->ze - it->he);
        float p = nuie.transpose() * Sie.inverse() * nuie;

        if (p < 5.9915) it->high_innovation_inlier = true;
        //else it->high_innovation_inlier = false; //was false already, reset in the beginning of iteration
    }
}

void EKF::updateHIInliers()  //this function is identical to the LI inliers, apart from the if ( ! it->high_innovation_inlier) continue;
{
    //NB: and also apart from that this one works on xkk and pkk instead of xkkm1 and pkkm1 !!!
    //should work as intended!
    //lets just make it one function with a parameter to set if it is low or high innovation inliers that we do...
    int inlierCount = 0;
    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if ( ! it->high_innovation_inlier) continue;
        inlierCount++;
    }

    //in case we dont have high innovation inliers at all
    if (inlierCount == 0) return;   //do nothing!

    Eigen::VectorXf hList(inlierCount * 2);
    Eigen::VectorXf zList(inlierCount * 2);
    Eigen::MatrixXf HList(inlierCount * 2, pkk.rows() );

    int counter = 0;
    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if ( ! it->high_innovation_inlier) continue;

        HList.block(counter * 2, 0, 2, pkk.rows()) = it->He;
        hList.segment(counter * 2, 2) = it->he;
        zList.segment(counter * 2, 2) = it->ze;
        counter++;
    }

    //now do the actual update

    Eigen::MatrixXf S1;
    S1.noalias() = HList * pkk * HList.transpose() + Eigen::MatrixXf::Identity( HList.rows(), HList.rows() );
    Eigen::MatrixXf K2;
    K2.noalias() = pkk * HList.transpose() * S1.inverse();
    //this inverse is a shortcut, check if its always OK
    xkk.noalias() = xkk + K2 * (zList - hList);
    pkk.noalias() = pkk - K2 * S1 * K2.transpose();
    pkk = 0.5 * pkk + 0.5 * pkk.transpose().eval(); // to solve aliasing issue (which is _not_ assumed here b/c no matmul)
    //commented out in matlab code:  p_k_k = ( speye(size(p_km1_k,1)) - K*H )*p_km1_k;  //why is it there?

    normalizeQuaternion();
}

void EKF::normalizeQuaternion()
{
    //normalize quaternion (is it necessary to do this every step?) (twice in every step even?)
    float norm2 = xkk.segment(3,4).norm();
    float r = xkk(3), x = xkk(4), y = xkk(5), z = xkk(6);

    Eigen::Matrix4f JNorm;
    JNorm << x*x+y*y+z*z, -r*x, -r*y, -r*z, -x*r, r*r+y*y+z*z, -x*y, -x*z, -y*r, -y*x, r*r+x*x+z*z, -y*z, -z*r, -z*x, -z*y, r*r+x*x+y*y;
    JNorm /= norm2;
    xkk.segment(3,4) /= norm2;
    pkk.block(3,0,4, pkk.cols()) = JNorm * pkk.block(3,0,4, pkk.cols()); // no aliasing here because of matmul
    pkk.block(0,3, pkk.rows(), 4) = pkk.block(0,3, pkk.rows(), 4) * JNorm.transpose();

}


void EKF::visualize(Mat & frameGray, char * fps)
{

    //full disclosure here..
    std::cout << "step " << step << std::endl;
    //std::cout << "xkk " << x_k_k.colRange(Range(0,13)) << std::endl;

    Point3d pathpoint(xkk(0),xkk(1),xkk(2));
    path.push_back(pathpoint);
    Mat figure1 = Mat::zeros(300,300, CV_8U);
    Mat figure2 = Mat::zeros(300,300, CV_8U);
    Point2d lastPoint1(150,150),lastPoint2(150,150);
    for (int i = 0; i < path.size(); i++)
    {
        Point2d thisPoint1(path[i].x * 150 + 150,path[i].y * 150 + 150);
        line(figure1, lastPoint1, thisPoint1, Scalar(255,255,255,255));
        lastPoint1 = thisPoint1;

        Point2d thisPoint2(path[i].x * 150 + 150,path[i].z * 150 + 150);
        line(figure2, lastPoint2, thisPoint2, Scalar(255,255,255,255));
        lastPoint2 = thisPoint2;
    }
    imshow("top view", figure1);
    imshow("back view", figure2);

    Mat outFrame;

    cvtColor(frameGray, outFrame, CV_GRAY2BGR);

    for(std::vector<feature>::iterator it = features_info.begin(); it != features_info.end(); ++it)
    {
        if (!it->measured) continue;

        Scalar color = Scalar(0,255,0,255);
        if (it->low_innovation_inlier) color = Scalar(0,0,255,255);

        Point2f p;
        p.x = it->ze(0);
        p.y = it->ze(1);
        putText(outFrame, "*", p, FONT_HERSHEY_SIMPLEX, 1, color); //put FPS text

        Point2f p2;
        p2.x = it->he(0);
        p2.y = it->he(1);
        putText(outFrame, "O", p2, FONT_HERSHEY_SIMPLEX, 1, Scalar(255,0,0,255)); //put FPS text

        rectangle(outFrame, it->lastSearchArea, Scalar(255,255,255,255) );
        line(outFrame, p, p2, Scalar(255,255,255,255));

        //std::cout << "feature " << it - features_info.begin() << " predicted " <<

    }

    putText(outFrame, fps, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255,255)); //put FPS text

    imshow("det", outFrame);


}
