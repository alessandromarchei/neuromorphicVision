#include <visionNodeFrames.hpp>

using namespace boost::placeholders;

#define MAX_FPS_FLIR 60.0   //max frames per second of the camera
/*
    CHANNEL MAPPINGS:
        - CHAN 5 : PUBLISH ALTITUDE (USE VISION AS ALTITUDE SOURCE)
        - CHAN 6 : ARM VEHICLE
        - CHAN 7 : MODE (MANUAL, MISSION, STABILIZED)
        - CHAN 8 : READING FRAMES (CAMERA ON)
        - CHAN 11 : RECORD (FROM CAMERA)
*/

#define CHAN_PUB_ALT 4  //5-1 : IF ACTIVE, USE THE OF ESTIMATED ALTITUDE 
#define CHAN_CAM_ON 7   //8-1 : IF ACTIVE, TURN ON THE CAMERA, SO READ FRAMES
#define CHAN_CAM_REC 10 //11-1  : IF ACTIVE, ACTIVE THE RECORDING AND SAVE THAT INTO A .avi file

VisionNodeFrames::VisionNodeFrames() {
    // Load parameters from ROS parameter server
    loadParameters();

    // Initialize ROS subscribers
    imu_sub = nh.subscribe("/mavros/imu/data", 100, &VisionNodeFrames::imuCallback, this);
    velocity_body_sub = nh.subscribe("/mavros/local_position/velocity_body", 100, &VisionNodeFrames::velocityBodyCallback, this);
    velocity_local_sub = nh.subscribe("/mavros/local_position/velocity_local", 100, &VisionNodeFrames::velocityLocalCallback, this);
    local_pos = nh.subscribe("/mavros/local_position/pose", 100, &VisionNodeFrames::localPositionCallback, this);
    gps_fix_sub = nh.subscribe("/mavros/global_position/raw/fix", 100, &VisionNodeFrames::gpsFixCallback, this);  // New subscriber
    gps_vel_sub = nh.subscribe("/mavros/global_position/raw/gps_vel", 100, &VisionNodeFrames::gpsVelCallback, this);  // New subscriber
    airspeed_sub = nh.subscribe("/mavros/vfr_hud", 100, &VisionNodeFrames::airspeedCallback, this);
    rc_sub = nh.subscribe("/mavros/rc/in", 100, &VisionNodeFrames::rcCallback, this);
    lidar_sub = nh.subscribe("/mavros/distance_sensor/hrlv_ez4_pub", 100, &VisionNodeFrames::lidarCallback, this);


    // Initialize ROS publisher on /mavros/vision_pose/pose
    vision_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 100);
    statustext_pub = nh.advertise<mavros_msgs::StatusText>("/mavros/statustext/send", 100);

    // Start the thread for altitude publishing
    altitude_thread = std::thread(&VisionNodeFrames::altitudePublisherThread, this);


    // Send the STATUS TEXT saying the RPI5 is connected correctly, 3 times
    std::string rpi5_setup = "RPI5 CONNECTED";
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        sendStatusText(rpi5_setup, 6); // Send with severity status = 6 (INFO)
    }

    syncSystemTimeWithGPS();

    initializeCamera();
    initializeFeatureDetector();
    initializeOutputFile();

}

void VisionNodeFrames::run() {
    // Start async spinner for callbacks
    ros::AsyncSpinner spinner(std::thread::hardware_concurrency());
    spinner.start();

    ROS_INFO("spinner started\n");

    // Create rate from exposure time (converting microseconds to seconds)
    ros::Rate rate(fps);
    // Main camera loop
    while (ros::ok()) {

        bool currentRecord, currentReadFrames;
        // Lock the mutex before reading the shared variables
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            currentRecord = record;
            currentReadFrames = readFrames;
        }
        // Check if camera should be running (based on RC channels)
        if(currentReadFrames)
        {
            //READ FRAMES HERE, SO GO TO PROCESSING
            if(FASTLKParallel == true)
            {
                processFramesParallel();
            }
            else
            {
                processFrames();
            }
        }

        
    }

    pCam->EndAcquisition();

    pCam->DeInit();

    // Release camera and system resources
    camList.Clear();
    systemCam->ReleaseInstance();

    // Close OpenCV window
    cv::destroyAllWindows();

    //close the recording video pointer
    closeRecorder();
}

void VisionNodeFrames::loadParameters() {    

    // Get FAST parameters
    nh.getParam("/FAST/threshold", fastParams.threshold);
    nh.getParam("/FAST/nonmaxSuppression", fastParams.nonmaxSuppression);
    nh.getParam("/FAST/gradientScoring/enable", fastParams.gradientScoringEnable);
    nh.getParam("/FAST/gradientScoring/desiredFeatures", fastParams.desiredFeatures);

    // Get camera parameters
    nh.getParam("/CAMERA/downSampling/enable", camParams.downsampleEnable);
    nh.getParam("/CAMERA/downSampling/height", camParams.resolutionDownsampled.height);
    nh.getParam("/CAMERA/downSampling/width", camParams.resolutionDownsampled.width);
    nh.getParam("/CAMERA/resolution/width", camParams.resolution.width);
    nh.getParam("/CAMERA/resolution/height", camParams.resolution.height);
    nh.getParam("/CAMERA/exposureTime", camParams.exposureTime);

    nh.getParam("/CAMERA/hfov", camParams.hfov);
    nh.getParam("/CAMERA/vfov", camParams.vfov);
    nh.getParam("/CAMERA/fx", camParams.fx);
    nh.getParam("/CAMERA/fy", camParams.fy);
    nh.getParam("/CAMERA/cx", camParams.cx);
    nh.getParam("/CAMERA/cy", camParams.cy);
    nh.getParam("/CAMERA/pixelSize", camParams.pixelSize);
    nh.getParam("/CAMERA/inclination", camParams.inclination);

    
    deltaTms = camParams.exposureTime / 1e3;        //convert from us to ms
    ROS_INFO("delta T : %f ms \n", deltaTms);


    fps = 1e6 / camParams.exposureTime;             //compute the inverse of the fps
    fps = (fps >= MAX_FPS_FLIR) ? MAX_FPS_FLIR : fps;
    ROS_INFO("fps : %f \n", fps);

    // Get LK parameters
    nh.getParam("/LK/winSize/width", lkParams.winSize.width);
    nh.getParam("/LK/winSize/height", lkParams.winSize.height);
    nh.getParam("/LK/maxLevel", lkParams.maxLevel);
    nh.getParam("/LK/criteria/maxCount", lkParams.criteria.maxCount);
    nh.getParam("/LK/criteria/epsilon", lkParams.criteria.epsilon);
    nh.getParam("/LK/flags", lkParams.flags);
    nh.getParam("/LK/minEigThreshold", lkParams.minEigThreshold);

    nh.getParam("/FASTLKParallel", FASTLKParallel);

    // Get data for outliers rejection
    nh.getParam("/REJECTION_FILTER/magnitudeThresholdPixel", magnitudeThresholdPixel);
    nh.getParam("/REJECTION_FILTER/boundThreshold", boundThreshold);

    // smoothing filter coefficient
    nh.getParam("/SMOOTHINGFILTER/enable", smoothingFilterEnable);
    if(smoothingFilterEnable){
        nh.getParam("/SMOOTHINGFILTER/type", smoothingFilterType);
        if(smoothingFilterType == 0){
            //complementary filter enabled
            nh.getParam("/SMOOTHINGFILTER/complementaryK", complementaryK);
        }
        else if(smoothingFilterType == 1){
            //low pass filter enabled
            nh.getParam("/SMOOTHINGFILTER/lpfK", lpfK);
        }
    }

    nh.getParam("/ALTITUDE/type", altitudeType);        //0 : average, 1 : median
    nh.getParam("/ALTITUDE/saturationValue", saturationValue);  //if something over this is computed, then saturate


    //value to start using the lidar, from 0 m to lidar_max m
    nh.getParam("/lidarMax", lidarMax);


    //frequency to publish altitude
    nh.getParam("/publishAltitudeFrequency", publishAltitudeFrequency);

    //select the source of velocity
    nh.getParam("/velocitySource", velocitySource);     //0 : airspped, 1 : groundspeed

    nh.getParam("/gpsTimeout", gpsTimeoutSeconds);     //0 : airspped, 1 : groundspeed


    //buffer length for the gyro data
    nh.getParam("/gyroBufferLength", gyroBufferLength);     //buffer length for storing the gyro values from the PX4

    gyroBufferLength = (gyroBufferLength <= 0) ? 5 : gyroBufferLength;    //assume a default value of 5


        // Check if downsampling is enabled
    ROS_INFO("downsample : %d\n", camParams.downsampleEnable);
    ROS_INFO("width : %d\n", camParams.resolution.width);
    ROS_INFO("height : %d\n", camParams.resolution.height);
    ROS_INFO("new width : %d\n", camParams.resolutionDownsampled.width);
    ROS_INFO("new height : %d\n", camParams.resolutionDownsampled.height);
    if (camParams.downsampleEnable == true) {
    
        // Calculate the scale factor for downsampling
        double scale_x = static_cast<double>(camParams.resolutionDownsampled.width) / static_cast<double>(camParams.resolution.width);
        double scale_y = static_cast<double>(camParams.resolutionDownsampled.height) / static_cast<double>(camParams.resolution.height);

        ROS_INFO("scale (%f,%f)\n", scale_x, scale_y);
        // Adjust intrinsic parameters
        camParams.fx *= scale_x;
        camParams.fy *= scale_y;
        camParams.cx *= scale_x;
        camParams.cy *= scale_y;
    }

    // Log the updated parameters
    ROS_INFO("Updated Camera Parameters:");
    ROS_INFO("Resolution: %dx%d", camParams.resolutionDownsampled.width, camParams.resolutionDownsampled.height);
    ROS_INFO("fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f", camParams.fx, camParams.fy, camParams.cx, camParams.cy);

    //deblurring parameters
    nh.getParam("/DEBLUR/method", deblurParams.general.method);
    nh.getParam("/DEBLUR/enable", deblurParams.general.enable);
    nh.getParam("/DEBLUR/showDebug", deblurParams.general.showDebug);

    // Laplacian parameters
    nh.getParam("/DEBLUR/laplacian/strength", deblurParams.laplacian.strength);
    nh.getParam("/DEBLUR/laplacian/kernelSize", deblurParams.laplacian.kernelSize);
    nh.getParam("/DEBLUR/laplacian/normalize", deblurParams.laplacian.normalize);

    // Unsharp mask parameters
    nh.getParam("/DEBLUR/unsharp/sigma", deblurParams.unsharp.sigma);
    nh.getParam("/DEBLUR/unsharp/strength", deblurParams.unsharp.strength);
    nh.getParam("/DEBLUR/unsharp/kernelSize", deblurParams.unsharp.kernelSize);

    // Bilateral filter parameters
    nh.getParam("/DEBLUR/bilateral/diameter", deblurParams.bilateral.diameter);
    nh.getParam("/DEBLUR/bilateral/sigmaColor", deblurParams.bilateral.sigmaColor);
    nh.getParam("/DEBLUR/bilateral/sigmaSpace", deblurParams.bilateral.sigmaSpace);
    nh.getParam("/DEBLUR/bilateral/borderType", deblurParams.bilateral.borderType);

    // Richardson-Lucy parameters
    nh.getParam("/DEBLUR/richardsonLucy/maxIterations", deblurParams.richardsonLucy.maxIterations);
    nh.getParam("/DEBLUR/richardsonLucy/kernelSize", deblurParams.richardsonLucy.kernelSize);
    nh.getParam("/DEBLUR/richardsonLucy/sigma", deblurParams.richardsonLucy.sigma);



    return;
}



cv::Mat &VisionNodeFrames::acquireImage()
{

    pImage = pCam->GetNextImage();


    currentTimestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

    //update the ID of the frame each time
    frameID++;
    //ROS_INFO("frame id = %d \n", frameID);
    // Check if the frame is valid
    if (pImage->IsIncomplete()) {
        ROS_ERROR("Image incomplete");
    }


    unsigned char* pData = static_cast<unsigned char*>(pImage->GetData());

    if (!pData) {
        ROS_ERROR("Null image data pointer");
    }


    int width = pImage->GetWidth();
    int height = pImage->GetHeight();
    //ROS_INFO("creating matrix of : %dx%d\n", width, height);

    currImage = cv::Mat(height, width, CV_8UC1, pData).clone(); // Clone to ensure data ownership    //currImage(height, width, CV_8UC1, pData);  // Assuming grayscale format

    //downsample the image in case it is chosen by the parameters
    if(camParams.downsampleEnable)
    {
        //downsample the image, on the same cv::Mat object
        //ROS_INFO("resizing\n");
        auto startResize = std::chrono::high_resolution_clock::now();
        cv::resize(currImage, currImage, camParams.resolutionDownsampled);
        auto stopdResize = std::chrono::high_resolution_clock::now();

        resizeImageTime = std::chrono::duration_cast<std::chrono::microseconds>(stopdResize - startResize).count();
                
    }

    deblurImage(currImage,deblurParams);

    return currImage;

}

void VisionNodeFrames::initializeFeatureDetector() {
    // Initialize the FAST detector
    //create the instance of the fast detector
    fastDetector = FastFeatureDetector::create(fastParams.threshold, fastParams.nonmaxSuppression);

    return;
}


void VisionNodeFrames::initializeCamera() {
    // LIVE MODE

    //instantiate the object called capture of tyoe cameraCapture
    // Initialize the Spinnaker library
    //SystemPtr system = System::GetInstance();
    //CameraList camList = system->GetCameras();
    
    // Check if there are any cameras connected
    if (camList.GetSize() == 0) {
        ROS_ERROR("No camera detected!");
    }

    // Select the first camera
    pCam = camList.GetByIndex(0);

    ROS_INFO("Cameras connected : %d\n", camList.GetSize());
    ROS_INFO("Camera detected : %u\n", pCam->GetUniqueID());
    
    ROS_INFO("Initializing the camera");
    pCam->Init();

    ROS_INFO("Begin Acquisition");
    pCam->BeginAcquisition();

    ROS_INFO("Configuring exposure");

    configureExposure(pCam, camParams.exposureTime);
    return;
}

void VisionNodeFrames::applyCornerDetection(const cv::Mat &image) {
    std::vector<cv::KeyPoint> keypoints;
    safeFeaturesApplied = false;   //prepare for this slice, clear the others
    
    fastDetector->detect(image, keypoints);

    detectedFeatures = keypoints.size();


    // Apply gradient scoring if enabled

    if (fastParams.gradientScoringEnable) {
        keypoints = scoreAndRankKeypointsUsingGradient(keypoints, image, fastParams.desiredFeatures);
    }
    

    filteredDetectedFeatures = keypoints.size();
    cv::KeyPoint::convert(keypoints, prevPoints);

    return;
}


void VisionNodeFrames::applyCornerDetection(const cv::Mat &edgeImage, std::vector<cv::Point2f> &output) {
    std::vector<cv::KeyPoint> keypoints;
    
    fastDetector->detect(edgeImage, keypoints);

    detectedFeatures = keypoints.size();

    // Apply gradient scoring if enabled

    if (fastParams.gradientScoringEnable) {
        keypoints = scoreAndRankKeypointsUsingGradient(keypoints, edgeImage, fastParams.desiredFeatures);
    }

    

    filteredDetectedFeatures = keypoints.size();
    cv::KeyPoint::convert(keypoints, output);

    return;
}


void VisionNodeFrames::processFrames() {


    //acquire image
    cv::Mat currImage = acquireImage();     //get the image from the camera pointer pCam


    //eventually save the image for recording
    if(isRecording)
    {
        saveFrameToVideo();
    }


    //call the opticalFlow function

    calculateOpticalFlow(currImage);


    prevPoints.clear();

    
    applyCornerDetection(currImage);


    prevImage = currImage.clone();

    processOF();

    // Release the image
    pImage->Release();


}


void VisionNodeFrames::processFramesParallel() {

    /*
        THIS FUNCTION IS ANALOGOUS TO THE processEvents FUNCTION
        IT PROCESS THE FAST FEATURE DETECTOR AND THE LK+ALTITUDE IN PARALLEL MANNER

        SINCE THE FAST FEATURE DETECTOR PRODUCES THE prevpoints, used by the LK to compute the currPoints, 
        the PARALLEL THREAD FOR FAST will save the computed features into "nextPrevPoints" that will be then saved into "prevPoints"
        at the thread.join() function, so that LK will use normally the prevPoints at the next iteration
    
    */
    auto startTotal = std::chrono::high_resolution_clock::now();

    //acquire current image
    cv::Mat currImage = acquireImage();

        //eventually save the image for recording
    if(isRecording)
    {
        saveFrameToVideo();
    }
    //START THE PARALLEL THREAD FOR FAST, defined in this way in the header.
    //specify that the function saves the features into nextprevpoints, instead of the default prevpoints as in serial fashion
    //     std::thread FASTThread;   

    //start the thread 
    FASTThread = std::thread(static_cast<void(VisionNodeFrames::*)(const cv::Mat&, std::vector<cv::Point2f>&)>(&VisionNodeFrames::applyCornerDetection), 
                         this, std::ref(currImage), std::ref(nextPrevPoints));
    //fast INPUT = currImage, OUTPUT = nextPrevPoints

    calculateOpticalFlow(currImage);

    //rotate the slices for the next LK iteration
    prevImage = currImage.clone();


    //processOF
    processOF();        //this function invokes the depths and filtered altitudes computation
    

    //wait for the FAST thread to finish
    FASTThread.join();

    //copy the nextPrevPoints into prevPoints
    prevPoints = nextPrevPoints;    //to avoid conflidcts with the other thread


    // Release the image
    pImage->Release();

}


void VisionNodeFrames::calculateOpticalFlow(const cv::Mat &currImage) {
    if (!prevImage.empty() && !prevPoints.empty()) {
        std::vector<cv::Point2f> currPoints;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(prevImage, currImage, prevPoints, currPoints, status, err, lkParams.winSize, lkParams.maxLevel, lkParams.criteria, lkParams.flags, lkParams.minEigThreshold);

        //PRINT CURR POINTS SIZE
        for (size_t i = 0; i < currPoints.size(); i++) {
            if (status[i]) {

                OFVectorFrame ofVector(prevPoints[i], currPoints[i], fps, camParams);
                flowVectors.push_back(ofVector);
            }
        }
        double uDegSec = 0.0;
        double vDegSec = 0.0;
        //now print the average of flowvectors
        for(int i = 0;i < flowVectors.size();i++)
        {
            uDegSec+= flowVectors[i].uDegSec;
            vDegSec+= flowVectors[i].vDegSec;
        }
        uDegSec /= flowVectors.size();
        vDegSec /= flowVectors.size();



        filteredFlowVectors = rejectOutliers(flowVectors, magnitudeThresholdPixel, boundThreshold);

        //save the number of filtered flow vectors
        rejectedVectors = flowVectors.size() - filteredFlowVectors.size();

        //APPLY THE DEROTATION
        avgGyro_rad_cam = getGyroData();        //get gyro data in camera mode from the px4 data (since the CAMERA does not contain a gyro)


        std::vector<float> ratios;
        for (auto& ofVector : filteredFlowVectors) {
            //float norm = ofVector.magnitudePixel;   //added for debugging
            applyDerotation3D(ofVector, avgGyro_rad_cam);
            //float norm_final = ofVector.magnitudePixel;//added for debugging
            //ratios.push_back(norm_final/norm); //added for debugging
        }

        //float mean_ratio = std::accumulate(ratios.begin(), ratios.end(), 0.0f) / ratios.size();
        //ROS_INFO("DEROTATED : %.2f \n", mean_ratio);

    }
}

cv::Vec3f VisionNodeFrames::getGyroData()
{
    // This function gets the stored data from the PX4 GYROSCOPE and converts it into the CAMERA FRAME.
    // An averaging filter is applied to the stored values to make it more precise.

    cv::Vec3f output_FLU(0, 0, 0); // Vector to store the averaged gyro data (initialized to zero)

    // GYRO DATA FROM THE PX4 is stored in the gyroBuffer vector, in rad/second
    // Compute the average of the stored gyro data from the PX4
    for (int i = 0; i < gyroBuffer.size(); i++) {
        output_FLU += gyroBuffer[i];    // Sum every value in the buffer
    }

    // Compute the average now by dividing each component of the vector by the size of the buffer
    if (gyroBuffer.size() > 0) {  // Avoid division by zero if the buffer is empty
        output_FLU[0] /= gyroBuffer.size();
        output_FLU[1] /= gyroBuffer.size();
        output_FLU[2] /= gyroBuffer.size();
    }
    //GYRO DATA IS IN FLU, convert it to FRD
    cv::Vec3f output_FRD(output_FLU[0],-output_FLU[1],-output_FLU[2]);

    // CONVERT IT INTO THE CAMERA REFERENCE SYSTEM
    return bodyToCam(output_FRD, camParams);  // Convert from body frame to camera frame
}


void VisionNodeFrames::applyDerotation3D(OFVectorFrame &ofVector, const cv::Vec3f &avgGyroRadSec) {
    double norm_a = norm(ofVector.AMeter);
    //ROS_INFO("\ndirection vector : (%.2f,%.2f,%.2f)\n", ofVector.directionVector[0], ofVector.directionVector[1], ofVector.directionVector[2]);
    cv::Vec2f Pprime_ms(ofVector.uPixelSec * camParams.pixelSize, ofVector.vPixelSec * camParams.pixelSize);
    cv::Vec3f PpPprime_ms(Pprime_ms[0] / norm_a, Pprime_ms[1] / norm_a, 0);
    cv::Vec3f P = PpPprime_ms - (PpPprime_ms.dot(ofVector.directionVector) * ofVector.directionVector);
    //ROS_INFO("P vector : (%.2f,%.2f,%.2f)\n", P[0], P[1], P[2]);

    cv::Vec3f RotOF = -avgGyroRadSec.cross(ofVector.directionVector);
    //ROS_INFO("RotOF vector : (%.2f,%.2f,%.2f)\n", RotOF[0], RotOF[1], RotOF[2]);
    ofVector.P = P - RotOF;
    return;
}

double VisionNodeFrames::estimateDepth(OFVectorFrame &ofVector, const cv::Vec3f T_cam) {
    double TdotD = T_cam.dot(ofVector.directionVector);
    double depth = norm(T_cam - (TdotD * ofVector.directionVector)) / norm(ofVector.P);
    return depth;
}

double VisionNodeFrames::estimateAltitude(OFVectorFrame &ofVector, double depth) {
    cv::Vec3f directionVector_body = camToBody(ofVector.directionVector, camParams);
    cv::Vec3f directionVector_inertial = bodyToInertial(directionVector_body, cosRoll, sinRoll, cosPitch, sinPitch);
    double cosTheta = directionVector_inertial[2];
    double altitude = depth * cosTheta;
    return altitude;
}

void VisionNodeFrames::processOF()
{

    //THIS FUNCTION PROCESS THE OPTICAL FLOW VECTORS
    //IT USES SPEED and OF vectors in order to compute the depths and airspeed

    std::vector<double> altitudes;
    for (auto& ofVector : filteredFlowVectors) {
        double depth = estimateDepth(ofVector, T_cam);
        double altitude = estimateAltitude(ofVector, depth);
        if (!std::isnan(altitude)) {
            altitudes.push_back(altitude);
        }
    }

//calculate the average or median based on the parameter
    if (altitudes.empty()) {
        avgAltitude = prevFilteredAltitude;
    } else {
        if(altitudeType == 1){
            //calculate the median here
            std::sort(altitudes.begin(), altitudes.end());
            avgAltitude = altitudes[altitudes.size() / 2];
        }
        else{
            //calculate the average here
            avgAltitude = std::accumulate(altitudes.begin(), altitudes.end(), 0.0) / altitudes.size();
        }
    }


    //check if the avgAltitude did not get over 45 meters, otherwise saturate
    if(avgAltitude >= saturationValue)
    {
        avgAltitude = saturationValue;
    }


    if (std::isnan(avgAltitude) || avgAltitude == 0) {
        avgAltitude = prevFilteredAltitude;
    }

    ////("Average altitude : %f", avgAltitude);
    //update the unfilteredAltitude with avgAltitude
    unfilteredAltitude = avgAltitude;

    {
        std::lock_guard<std::mutex> lock(altitude_mutex);
        if (smoothingFilterType == 0) {

            filteredAltitude = complementaryFilter(avgAltitude, prevFilteredAltitude, complementaryK, deltaTms/1000.0);
            } else if (smoothingFilterType == 1) {
                filteredAltitude = LPFilter(avgAltitude, prevFilteredAltitude, lpfK);
            }
    }

    //ROS_INFO("Filtered altitude : %f", filteredAltitude);

    if(lidarCurrentData < lidarMax)
    {
        //use the lidar data because we are too close to the ground
        filteredAltitude = lidarCurrentData;
    }

    if (std::isnan(filteredAltitude)) {
        prevFilteredAltitude = 0;
    } else {
        prevFilteredAltitude = filteredAltitude;
    }


    flowVectors.clear();
    filteredFlowVectors.clear();

}

void VisionNodeFrames::airspeedCallback(const mavros_msgs::VFR_HUD::ConstPtr &msg) {
    
    //this should be the airspeed data processed, not the one used by the sensor
    
    T_airspeed = cv::Vec3f(msg->airspeed, 0, 0); // in m/s
    T_groundspeed = cv::Vec3f(msg->groundspeed, 0, 0); // in m/s

    if(velocitySource == 0)
    {
        //use the airspeed data
        T_cam = bodyToCam(T_airspeed, camParams);
    }
    else
    {
        //use the groundspeed data
        T_cam = bodyToCam(T_groundspeed, camParams);
    }


}

void VisionNodeFrames::velocityBodyCallback(const geometry_msgs::TwistStamped::ConstPtr &msg) {
    //THE RECEIVED VELOCITY is relative to the body frame, FLU (x : forward, y : left, z : up), LIKE IN ROS
    //the body of the drone is in FRD (x : forward, y : right, z : down) (like PX4)
    T_GPS_body_FLU = cv::Vec3f(msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z);
    T_GPS_body_FRD = cv::Vec3f(T_GPS_body_FLU[0], -T_GPS_body_FLU[1], -T_GPS_body_FLU[2]);
    T_GPS_cam_FRD = bodyToCam(T_GPS_body_FRD, camParams);
}

void VisionNodeFrames::velocityLocalCallback(const geometry_msgs::TwistStamped::ConstPtr &msg){
    //THE RECEIVED VELOCITY is relative to the world frame, ENU (x : EAST, y : NORD, z : UP), LIKE IN ROS for world reference frames
    T_GPS_local_ENU = cv::Vec3f(msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z);
    T_GPS_local_NED = cv::Vec3f(T_GPS_local_ENU[1], T_GPS_local_ENU[0], -T_GPS_local_ENU[2]);
    //ROS_INFO("LOCAL VEL NED (vx,vy,vz) : (%f , %f, %f)", T_GPS_local_NED[0], T_GPS_local_NED[1], T_GPS_local_NED[2]);


}

void VisionNodeFrames::localPositionCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
    //the received POSE is represented in ENU, so the Z coordinate becomes more and more positive every time
    // Access position data
    localPosition_ENU = cv::Vec3f(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    localPosition_NED = cv::Vec3f(localPosition_ENU[1], localPosition_ENU[0], -localPosition_ENU[2]);

}


void VisionNodeFrames::imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
{   
    // Create a new gyroscope data vector from the incoming message. UNIT : RAD/SEC, in FLU (FORWARD, LEFT, UP) reference system
    cv::Vec3f incomingGyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    
    // Insert the new data at the beginning of the buffer
    gyroBuffer.insert(gyroBuffer.begin(), incomingGyro);

    // Ensure the buffer length does not exceed the specified limit
    if (gyroBuffer.size() > gyroBufferLength) {
        gyroBuffer.pop_back(); // Remove the oldest element
    }


    // Compute roll and pitch angles
    quat_imu = tf::Quaternion(
        msg->orientation.x,
        msg->orientation.y,
        msg->orientation.z,
        msg->orientation.w);

    // Convert quaternion to roll, pitch, and yaw
    tf::Matrix3x3 m(quat_imu);
    m.getRPY(rollRad, pitchRad, yawRad);

    // Compute sine and cosine for roll and pitch
    cosRoll = std::cos(rollRad);
    sinRoll = std::sin(rollRad);
    cosPitch = std::cos(pitchRad);
    sinPitch = std::sin(pitchRad);

    // Convert roll and pitch to degrees
    rollDeg = rollRad * (180.0 / CV_PI);
    pitchDeg = pitchRad * (180.0 / CV_PI);

    // Extract the timestamp in microseconds
    uint64_t timestamp_imu_us = static_cast<uint64_t>(msg->header.stamp.sec) * 1e6 + msg->header.stamp.nsec / 1e3;

}


VisionNodeFrames::~VisionNodeFrames() {
    // Join the altitude thread before destruction
    if (altitude_thread.joinable()) {
        altitude_thread.join();
    }
}

void VisionNodeFrames::lidarCallback(const sensor_msgs::Range::ConstPtr &msg) {
    // This function should be used to get the lidar data
    // Data received from topic: /mavros/distance_sensor/hrlv_ez4_pub
    // This function should store the lidar data in a variable
    // double lidarData: LIDAR DATA IN METERS

    lidarCurrentData = msg->range;
}

void VisionNodeFrames::sendStatusText(const std::string &message, uint8_t severity) {
    mavros_msgs::StatusText status_text_msg;
    status_text_msg.text = message;
    status_text_msg.severity = severity;
    statustext_pub.publish(status_text_msg);
}


void VisionNodeFrames::altitudePublisherThread() {
    ros::Rate rate(publishAltitudeFrequency);
    while (ros::ok()) {
        
        bool currentreadFrames, currentRecord;

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            currentreadFrames = readFrames;
            currentRecord = record;
        }

        if (publishAltitude && readFrames) {
            geometry_msgs::PoseStamped pose;
            pose.header.stamp = ros::Time::now();
            {
                std::lock_guard<std::mutex> lock(altitude_mutex);
                pose.pose.position.z = filteredAltitude;
            }

            //if (lidarCurrentData < lidarMax) {
            //    pose.pose.position.z = lidarCurrentData;
            //}

            vision_pose_pub.publish(pose);

            //ROS_INFO("published\n");
        }
        rate.sleep();
    }
}



void VisionNodeFrames::syncSystemTimeWithGPS() {
    ROS_INFO("Attempting to synchronize system time with GPS...");

    uint32_t satellite_count = 0;
    bool gps_time_synced = false;

    // Start time
    auto start_time = ros::Time::now();

    while (ros::ok()) {
        // Calculate elapsed time
        auto elapsed_time = ros::Time::now() - start_time;

        // If more than 5 seconds have passed, break the loop
        if (elapsed_time.toSec() > gpsTimeoutSeconds) {
            ROS_WARN("Timeout reached. Proceeding without GPS synchronization.");
            break;
        }

        // Wait for the "/mavros/global_position/raw/satellites" message
        auto msg = ros::topic::waitForMessage<std_msgs::UInt32>("/mavros/global_position/raw/satellites", nh, ros::Duration(1.0));

        if (msg) {
            satellite_count = msg->data;
            if (satellite_count >= 3) {
                ROS_INFO("GPS satellites available: %d. Synchronizing time...", satellite_count);

                // Wait for the time reference message
                auto time_msg = ros::topic::waitForMessage<sensor_msgs::TimeReference>("/mavros/time_reference", nh, ros::Duration(1.0));

                if (time_msg) {
                    double gps_time = time_msg->time_ref.toSec();
                    auto gps_date = ros::Time(gps_time);

                    // Convert the GPS time to a human-readable string in the local timezone
                    std::time_t gps_time_t = gps_date.sec;
                    std::tm* tm_ptr = std::localtime(&gps_time_t);  // Use localtime to get local time

                    std::stringstream time_ss;
                    time_ss << std::put_time(tm_ptr, "%Y-%m-%d %H:%M:%S");

                    // Set the system time
                    std::string command = "sudo date -s \"" + time_ss.str() + "\"";
                    int ret = system(command.c_str());

                    if (ret == 0) {
                        ROS_INFO("System time synchronized with GPS.");
                        gps_time_synced = true;

                        // Send a status text with the synchronized time
                        std::string status_msg = "GPS Time Synchronized: " + time_ss.str();
                        sendStatusText(status_msg, 6);

                        break; // Exit the loop and continue with the rest of the initialization
                    } else {
                        ROS_ERROR("Failed to synchronize system time with GPS.");
                    }
                }
            } else {
                ROS_INFO("Insufficient satellites (%d). Waiting...", satellite_count);
            }
        }

        ros::Duration(1.0).sleep();  // Wait for 1 second before retrying
    }

    if (!gps_time_synced) {
        // If GPS time sync failed, use the current system time
        ros::Time current_time = ros::Time::now();
        ROS_WARN("Proceeding with system time: %ld", current_time.sec);

        // Convert the current system time to a human-readable string in the local timezone
        std::time_t current_time_t = current_time.sec;
        std::tm* tm_ptr = std::localtime(&current_time_t);  // Use localtime to get local time

        std::stringstream current_time_ss;
        current_time_ss << std::put_time(tm_ptr, "%Y-%m-%d %H:%M:%S");

        // Send a status text with the current system time
        std::string status_msg = "Using System Time: " + current_time_ss.str();
        sendStatusText(status_msg, 6);
    }
}


void VisionNodeFrames::gpsFixCallback(const sensor_msgs::NavSatFix::ConstPtr &msg) {
    // Save the latitude, longitude, and altitude from the GPS fix
    latest_gps_latitude = msg->latitude;
    latest_gps_longitude = msg->longitude;
    latest_gps_altitude = msg->altitude;

}

void VisionNodeFrames::gpsVelCallback(const geometry_msgs::TwistStamped::ConstPtr &msg) {
    // Save the linear and angular velocities from the GPS
    latest_gps_vel_x = msg->twist.linear.x;
    latest_gps_vel_y = msg->twist.linear.y;
    latest_gps_vel_z = msg->twist.linear.z;
    latest_gps_ang_vel_x = msg->twist.angular.x;
    latest_gps_ang_vel_y = msg->twist.angular.y;
    latest_gps_ang_vel_z = msg->twist.angular.z;

}


void VisionNodeFrames::configureExposure(CameraPtr pCam, int exposure)
{
    ROS_INFO("*** CONFIGURING EXPOSURE ***");

    try
    {
        if (exposure == 0)
        {
            // Enable automatic exposure
            if (!IsWritable(pCam->ExposureAuto))
            {
                ROS_ERROR("Unable to enable automatic exposure");
                return;
            }

            pCam->ExposureAuto.SetValue(ExposureAuto_Continuous);
            ROS_INFO("Automatic exposure enabled");
        }
        else
        {
            // Disable automatic exposure
            if (!IsWritable(pCam->ExposureAuto))
            {
                ROS_ERROR("Unable to disable automatic exposure");
                return;
            }

            pCam->ExposureAuto.SetValue(ExposureAuto_Off);
            ROS_INFO("Automatic exposure disabled");

            // Set manual exposure time
            if (!IsReadable(pCam->ExposureTime) || !IsWritable(pCam->ExposureTime))
            {
                ROS_ERROR("Unable to set exposure time");
                return;
            }

            // Ensure desired exposure time does not exceed the maximum
            const double exposureTimeMax = pCam->ExposureTime.GetMax();
            double exposureTimeToSet = exposure;

            if (exposureTimeToSet > exposureTimeMax)
            {
                exposureTimeToSet = exposureTimeMax;
            }

            pCam->ExposureTime.SetValue(exposureTimeToSet);

            std::cout << std::fixed << "Exposure time set to " << exposureTimeToSet << " us..." << std::endl << std::endl;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }

    return;
}




void VisionNodeFrames::rcCallback(const mavros_msgs::RCIn::ConstPtr &msg) {
    std::vector<unsigned short int> channels = msg->channels;

    // Check channel 5 -> publishAltitude. Only if it forward, so ON if CHANNEL < 1000
    publishAltitude = (channels[CHAN_PUB_ALT] < 1000);

    static bool wasCamRecordingOn = false;
    static bool wasReadFramesOn = false;

    if (channels[CHAN_CAM_ON] < 1000) {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            readFrames = true;
            record = true;
        }

        if (!wasReadFramesOn) {
            ROS_INFO("START CAM AND RECORDING\n");
            boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
            std::string date_str = boost::posix_time::to_iso_string(now);
            std::string date_folder = date_str.substr(6, 2) + date_str.substr(4, 2) + date_str.substr(2, 2);

            std::string base_path = "/home/sharedData/test" + date_folder;
            std::string rec_dir = base_path + "/recordings";

            if (!std::filesystem::exists(rec_dir)) std::filesystem::create_directories(rec_dir);

            std::string timestamp_str = boost::posix_time::to_iso_string(now);
            std::string rec_filename = rec_dir + "/rec" + timestamp_str + ".avi";

            recFilePath = rec_filename;

            initializeRecordingFile(recFilePath);

            ROS_INFO("CAM AND RECORDER INITIALIZED\n");
            std::string files_starts = "Saving in " + date_folder + "\n";
            sendStatusText(files_starts, 6);
        }

        wasReadFramesOn = true;
        wasCamRecordingOn = true;
    } else {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            readFrames = false;
            record = false;
        }

        if (wasReadFramesOn) {
            ROS_INFO("CAM AND RECORDING OFF\n");


            {
                std::lock_guard<std::mutex> lock(recorder_mutex);
                closeRecorder();
            }

        }

        wasReadFramesOn = false;
        wasCamRecordingOn = false;
    }
}


void VisionNodeFrames::saveFrameToVideo()
{
    // Check for recording configuration outside this function
    if (isRecording && !currImage.empty() && recorder.isOpened()) {
        // Write the frame to the video
        //ROS_INFO("image saved\n");
        recorder.write(currImage);
    }
}


void VisionNodeFrames::initializeRecordingFile(const std::string& filePath) {
    std::lock_guard<std::mutex> lock(recorderMutex); // Ensure thread safety

    if (recorder.isOpened()) {
        ROS_WARN("Recorder already initialized. Closing the previous one.");
        recorder.release();
    }

    if (filePath.empty()) {
        ROS_ERROR("Invalid path for recording file");
        return;
    }

    cv::Size imageSize;
    if(camParams.downsampleEnable)
    {
        imageSize = camParams.resolutionDownsampled;
    }
    else
    {
        imageSize = camParams.resolution;
    }
    
    int fpsRecording = static_cast<int>(fps);

    recorder.open(filePath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fpsRecording, imageSize,false);
    if (!recorder.isOpened()) {
        ROS_ERROR("Failed to initialize video recorder at path: %s", filePath.c_str());
        return ;
    }

    ROS_INFO("Initialized recording to %s", filePath.c_str());
    isRecording = true;
    return ;
}

void VisionNodeFrames::closeRecorder() {
    std::lock_guard<std::mutex> lock(recorderMutex); // Ensure thread safety

    if (recorder.isOpened()) {
        ROS_INFO("Closing the recorder.");
        recorder.release();
        isRecording = false;
    } else {
        ROS_WARN("Recorder was not opened.");
    }
}
