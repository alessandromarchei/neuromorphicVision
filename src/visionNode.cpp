#include <visionNode.hpp>

using namespace boost::placeholders;


/*
    CHANNEL MAPPINGS:
        - CHAN 5 : PUBLISH ALTITUDE (USE VISION AS ALTITUDE SOURCE)
        - CHAN 6 : ARM VEHICLE
        - CHAN 7 : MODE (MANUAL, MISSION, STABILIZED)
        - CHAN 8 : READING EVENTS (CAMERA ON)
        - CHAN 9 : EFPS
        - CHAN 10 : SENSITIVITY
        - CHAN 11 : RECORD (FROM CAMERA)
*/

#define CHAN_PUB_ALT 4  //5-1
#define CHAN_CAM_ON 7   //8-1
#define CHAN_CAM_REC 10 //11-1
#define CHAN_EFPS 8     //9-1
#define CHAN_SENSITIVITY 9  //10-1

/*
    CHAN_EFPS & CHAN_SENSITIVITY are analog channels, so they are not binary
    MIN VALUE = 982, MAX VALUE = 2006

    SENSITIVITY RANGES
    1 : 982 - 1185      -> VERY LOW
    2 : 1186 - 1390     -> LOW
    3 : 1391 - 1594     -> MEDIUM
    4 : 1595 - 1798     -> HIGH
    5 : 1799 - 2006     -> VERY HIGH


    EFPS ranges (from 1 to 5)
    - EFPS_CONSTANT_100 : 982 - 1185        //FIX GLOBAL SHUTTER @ 100 FPS
    - EFPS_CONSTANT_500 : 1186 - 1390
    - EFPS_CONSTANT_1000 : 1391 - 1594      //HIGHEST EVENT RATE
    - EFPS_VARIABLE_2000 : 1595 - 1798      //COULD PRODUCE LOWER EVENT RATES
    - EFPS_VARIABLE_5000 : 1799 - 2006

*/

VisionNode::VisionNode() : slicer("events") {
    // Load parameters from ROS parameter server
    loadParameters();

    // Initialize ROS subscribers
    imu_sub = nh.subscribe("/mavros/imu/data", 100, &VisionNode::imuCallback, this);
    velocity_body_sub = nh.subscribe("/mavros/local_position/velocity_body", 100, &VisionNode::velocityBodyCallback, this);
    velocity_local_sub = nh.subscribe("/mavros/local_position/velocity_local", 100, &VisionNode::velocityLocalCallback, this);
    local_pos = nh.subscribe("/mavros/local_position/pose", 100, &VisionNode::localPositionCallback, this);
    gps_fix_sub = nh.subscribe("/mavros/global_position/raw/fix", 100, &VisionNode::gpsFixCallback, this);  // New subscriber
    gps_vel_sub = nh.subscribe("/mavros/global_position/raw/gps_vel", 100, &VisionNode::gpsVelCallback, this);  // New subscriber
    airspeed_sub = nh.subscribe("/mavros/vfr_hud", 100, &VisionNode::airspeedCallback, this);
    rc_sub = nh.subscribe("/mavros/rc/in", 100, &VisionNode::rcCallback, this);
    lidar_sub = nh.subscribe("/mavros/distance_sensor/hrlv_ez4_pub", 100, &VisionNode::lidarCallback, this);


    // Initialize ROS publisher on /mavros/vision_pose/pose
    vision_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 100);
    statustext_pub = nh.advertise<mavros_msgs::StatusText>("/mavros/statustext/send", 100);

    // Start the thread for altitude publishing
    altitude_thread = std::thread(&VisionNode::altitudePublisherThread, this);


    syncSystemTimeWithGPS();


    initializeAccumulator();
    initializeFeatureDetector();
    initializeCamera();
    initializeSlicer();



        // Main loop
    // Main loop
    while (ros::ok() && capture->handleNext(handler)) {
        ros::spinOnce();

        bool currentRecord, currentReadEvents;

        // Lock the mutex before reading the shared variables
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            currentRecord = record;
            currentReadEvents = readEvents;
        }

        // Check if recording is enabled
        if (currentRecord && currentReadEvents) {
            auto eventsOpt = capture->getNextEventBatch();
            auto imuOpt = capture->getNextImuBatch();

            if (eventsOpt.has_value() && imuOpt.has_value()) {
                saveEvents(eventsOpt.value(), imuOpt.value());
            } else if (eventsOpt.has_value()) {
                dv::cvector<dv::IMU> emptyImu;  // Create an empty IMU vector
                saveEvents(eventsOpt.value(), emptyImu);
            } else if (imuOpt.has_value()) {
                dv::EventStore emptyEvents;  // Create an empty EventStore
                saveEvents(emptyEvents, imuOpt.value());
            }
        }
    }
}

void VisionNode::run() {
    ros::AsyncSpinner spinner(std::thread::hardware_concurrency());
    spinner.start();
    ros::waitForShutdown();
}

void VisionNode::loadParameters() {
    // GET THE INPUT PARAMETERS, Live or from a file (.aedat4)

    // Get FAST parameters
    nh.getParam("/FAST/threshold", fastParams.threshold);
    nh.getParam("/FAST/nonmaxSuppression", fastParams.nonmaxSuppression);
    nh.getParam("/FAST/gradientScoring/enable", fastParams.gradientScoringEnable);
    nh.getParam("/FAST/gradientScoring/desiredFeatures", fastParams.desiredFeatures);

    // Get accumulator parameters
    nh.getParam("/ACCUMULATOR/neutralPotential", accParams.neutralPotential);
    nh.getParam("/ACCUMULATOR/eventContribution", accParams.eventContribution);
    nh.getParam("/ACCUMULATOR/decay", accParams.decay);
    nh.getParam("/ACCUMULATOR/ignorePolarity", accParams.ignorePolarity);

    // Get slicing parameters
    nh.getParam("/SLICER/fps", fps);
    dtMicroseconds = 1e6 / fps;

    // Get camera parameters
    nh.getParam("/CAMERA/efps", camParams.efps);
    nh.getParam("/CAMERA/sensitivity", camParams.sensitivity);
    nh.getParam("/CAMERA/width", camParams.resolution.width);
    nh.getParam("/CAMERA/height", camParams.resolution.height);
    nh.getParam("/CAMERA/hfov", camParams.hfov);
    nh.getParam("/CAMERA/vfov", camParams.vfov);
    nh.getParam("/CAMERA/fx", camParams.fx);
    nh.getParam("/CAMERA/fy", camParams.fy);
    nh.getParam("/CAMERA/cx", camParams.cx);
    nh.getParam("/CAMERA/cy", camParams.cy);
    nh.getParam("/CAMERA/pixelSize", camParams.pixelSize);
    nh.getParam("/CAMERA/gyroFreq", camParams.gyroFreq);
    nh.getParam("/CAMERA/inclination", camParams.inclination);

    // Get LK parameters
    nh.getParam("/LK/winSize/width", lkParams.winSize.width);
    nh.getParam("/LK/winSize/height", lkParams.winSize.height);
    nh.getParam("/LK/maxLevel", lkParams.maxLevel);
    nh.getParam("/LK/criteria/maxCount", lkParams.criteria.maxCount);
    nh.getParam("/LK/criteria/epsilon", lkParams.criteria.epsilon);
    nh.getParam("/LK/flags", lkParams.flags);
    nh.getParam("/LK/minEigThreshold", lkParams.minEigThreshold);


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
    // load flag for executing FAST and LK in parallel.
    nh.getParam("/FASTLKParallel", FASTLKParallel);

    //frequency to publish altitude
    nh.getParam("/publishAltitudeFrequency", publishAltitudeFrequency);

    //select the source of velocity
    nh.getParam("/velocitySource", velocitySource);     //0 : airspped, 1 : groundspeed

    nh.getParam("/gpsTimeout", gpsTimeoutSeconds);     //0 : airspped, 1 : groundspeed

    return;
}



void VisionNode::initializeFeatureDetector() {
    // Initialize the FAST detector
    //create the instance of the fast detector
    fastDetector = FastFeatureDetector::create(fastParams.threshold, fastParams.nonmaxSuppression);

    return;
}


void VisionNode::initializeAccumulator() {
    accumulator = std::make_unique<dv::EdgeMapAccumulator>(camParams.resolution, accParams.eventContribution,
                                                           accParams.ignorePolarity, accParams.neutralPotential, accParams.decay);
    ROS_INFO("Accumulator initialized.");

    return;
}


void VisionNode::initializeCamera() {
    // LIVE MODE



    dv::io::CameraCapture::DVXeFPS efps = intToDVXeFPS(camParams.efps);
    currentEFPS = camParams.efps;
    dv::io::CameraCapture::BiasSensitivity sensitivity = intToBiasSensitivity(camParams.sensitivity);
    currentSensitivity = camParams.sensitivity;

    //instantiate the object called capture of tyoe cameraCapture

    if(!capture)
    {
        capture = std::make_unique<dv::io::CameraCapture>();
        //now the timestamp of the camera is synchronized with the RPI5, that was previously synchronized with the GPS
    }
    ROS_INFO("Camera [%s] has been opened!", capture->getCameraName().c_str());

    capture->setDVXplorerEFPS(efps);
    capture->setDVSBiasSensitivity(sensitivity);
    ROS_INFO("Camera parameters set.");

    return;
}

void VisionNode::initializeSlicer() {
    handler.mEventHandler = [this](const auto &events) {
        slicer.accept("events", events);
    };

    slicer.addStream<dv::cvector<dv::IMU>>("imu");
    handler.mImuHandler = [this](auto &imu) {
        slicer.accept("imu", imu);
    };

    slicer.doEveryTimeInterval(std::chrono::milliseconds(1000 / fps), [this](const auto &data) {

        if(readEvents && !record)
        {
            
            events = data.template get<dv::EventStore>("events");
            imu_cam = data.template get<dv::cvector<dv::IMU>>("imu");

            currentTimestamp = dv::packets::getPacketTimestamp<dv::packets::Timestamp::END>(events);

            if(FASTLKParallel){
                processEventsParallel(events, imu_cam);
            }
            else{
                processEvents(events, imu_cam);
            }


            //eventually apply the slicing window if enabled
            applyAdaptiveSlicing();
        }

        else
        {
            //do nothing
        }
        

    });

    //ROS_INFO("Slicer initialized.");
}

void VisionNode::applyCornerDetection(const cv::Mat &edgeImage) {
    std::vector<cv::KeyPoint> keypoints;
    safeFeaturesApplied = false;   //prepare for this slice, clear the others
    
    if(accParams.neutralPotential == 0.0)
    {
        fastDetector->detect(edgeImage, keypoints, edgeImage);
    }
    else
    {
        fastDetector->detect(edgeImage, keypoints);
    }

    detectedFeatures = keypoints.size();


    // Apply gradient scoring if enabled

    if (fastParams.gradientScoringEnable) {
        keypoints = scoreAndRankKeypointsUsingGradient(keypoints, edgeImage, fastParams.desiredFeatures);
    }

    filteredDetectedFeatures = keypoints.size();
    cv::KeyPoint::convert(keypoints, prevPoints);

    return;
}


void VisionNode::processEvents(const dv::EventStore &events, dv::cvector<dv::IMU> &imu) {

    accumulator->accept(events);
    cv::Mat currEdgeImage = accumulator->generateFrame().image;
    //show the image

    calculateOpticalFlow(currEdgeImage);

    prevPoints.clear();

    applyCornerDetection(currEdgeImage);

    prevEdgeImage = currEdgeImage.clone();


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

    //ROS_INFO("Average altitude : %f", avgAltitude);
    //update the unfilteredAltitude with avgAltitude
    unfilteredAltitude = avgAltitude;

    {
        std::lock_guard<std::mutex> lock(altitude_mutex);
        if (smoothingFilterType == 0) {
        float deltaT;
        if(downsample == true)
        {
            deltaT = 2.0 / static_cast<float>(fps);
        }
        else
        {
            deltaT = 1.0 / static_cast<float>(fps);
        }

        filteredAltitude = complementaryFilter(avgAltitude, prevFilteredAltitude, complementaryK, deltaT);
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


void VisionNodePlayback::processEventsParallel(const dv::EventStore &events, dv::cvector<dv::IMU> &imu) {

    /*
        THIS FUNCTION IS ANALOGOUS TO THE processEvents FUNCTION
        IT PROCESS THE FAST FEATURE DETECTOR AND THE LK+ALTITUDE IN PARALLEL MANNER

        SINCE THE FAST FEATURE DETECTOR PRODUCES THE prevpoints, used by the LK to compute the currPoints, 
        the PARALLEL THREAD FOR FAST will save the computed features into "nextPrevPoints" that will be then saved into "prevPoints"
        at the thread.join() function, so that LK will use normally the prevPoints at the next iteration
    
    */

    accumulator->accept(events);
    cv::Mat currEdgeImage = accumulator->generateFrame().image;


    //START THE PARALLEL THREAD FOR FAST, defined in this way in the header.
    //specify that the function saves the features into nextprevpoints, instead of the default prevpoints as in serial fashion
    //     std::thread FASTThread;   

    //start the thread 
    FASTThread = std::thread(static_cast<void(VisionNodePlayback::*)(const cv::Mat&, std::vector<cv::Point2f>&)>(&VisionNodePlayback::applyCornerDetection), 
                         this, std::ref(currEdgeImage), std::ref(nextPrevPoints));
    //fast INPUT = currEdgeImage, OUTPUT = nextPrevPoints

    calculateOpticalFlow(currEdgeImage);

    //rotate the slices for the next LK iteration
    prevEdgeImage = currEdgeImage.clone();

    cv::Vec3f T_airspeed(currentSensorData.airspeed, 0, 0);
    cv::Vec3f T_airspeed_cam_sensordata = bodyToCam(T_airspeed, camParams);

    std::vector<double> altitudes;
    for (auto& ofVector : filteredFlowVectors) {
        double depth = estimateDepth(ofVector, T_airspeed_cam_sensordata);
        double altitude = estimateAltitude(ofVector, depth, std::cos(currentSensorData.roll_angle * CV_PI / 180.0), std::sin(currentSensorData.roll_angle * CV_PI / 180.0), std::cos(currentSensorData.pitch_angle * CV_PI / 180.0), std::sin(currentSensorData.pitch_angle * CV_PI / 180.0));
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

    if (std::isnan(avgAltitude) || avgAltitude == 0) {
        avgAltitude = prevFilteredAltitude;
    }

    if (smoothingFilterType == 0) {
        float deltaT;
        deltaT = 1.0 / static_cast<float>(fps);

        filteredAltitude = complementaryFilter(avgAltitude, prevFilteredAltitude, complementaryK, deltaT);
    } else if (smoothingFilterType == 1) {
        filteredAltitude = LPFilter(avgAltitude, prevFilteredAltitude, lpfK);
    }

    if (std::isnan(filteredAltitude)) {
        prevFilteredAltitude = 0;
    } else {
        prevFilteredAltitude = filteredAltitude;
    }
    

    //wait for the FAST thread to finish
    FASTThread.join();

    //copy the nextPrevPoints into prevPoints
    prevPoints = nextPrevPoints;    //to avoid conflidcts with the other thread

}

void VisionNode::calculateOpticalFlow(const cv::Mat &currEdgeImage) {
    if (!prevEdgeImage.empty() && !prevPoints.empty()) {
        std::vector<cv::Point2f> currPoints;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(prevEdgeImage, currEdgeImage, prevPoints, currPoints, status, err, lkParams.winSize, lkParams.maxLevel, lkParams.criteria, lkParams.flags, lkParams.minEigThreshold);

        //PRINT CURR POINTS SIZE
        for (size_t i = 0; i < currPoints.size(); i++) {
            if (status[i]) {

                OFVectorEvents ofVector(prevPoints[i], currPoints[i], fps, camParams);
                flowVectors.push_back(ofVector);
            }
        }

        filteredFlowVectors = rejectOutliers(flowVectors, magnitudeThresholdPixel, boundThreshold);

        //save the number of filtered flow vectors
        rejectedVectors = flowVectors.size() - filteredFlowVectors.size();


        //PRINT NUMBER OF FLOW VECTORS
        avgGyro = avgGyroDataRadSec(imu_cam);

        for (auto& ofVector : filteredFlowVectors) {
            applyDerotation3D(ofVector, avgGyro);
        }
    }
}




cv::Vec3f VisionNode::avgGyroDataRadSec(dv::cvector<dv::IMU>& imuBatch) {
    cv::Vec3f avgGyro(0, 0, 0);

    for (const auto& imu : imuBatch) {
        cv::Vec3f gyro(-imu.gyroscopeX * (CV_PI / 180.0),
                       imu.gyroscopeY * (CV_PI / 180.0),
                       -imu.gyroscopeZ * (CV_PI / 180.0));

        avgGyro += gyro;
    }

    avgGyro /= static_cast<float>(imuBatch.size());

    return avgGyro;
}


void VisionNode::applyDerotation3D(OFVectorEvents &ofVector, const cv::Vec3f &avgGyroRadSec) {
    double norm_a = norm(ofVector.AMeter);
    cv::Vec2f Pprime_ms(ofVector.uPixelSec * camParams.pixelSize, ofVector.vPixelSec * camParams.pixelSize);
    cv::Vec3f PpPprime_ms(Pprime_ms[0] / norm_a, Pprime_ms[1] / norm_a, 0);
    cv::Vec3f P = PpPprime_ms - (PpPprime_ms.dot(ofVector.directionVector) * ofVector.directionVector);
    cv::Vec3f RotOF = -avgGyroRadSec.cross(ofVector.directionVector);
    ofVector.P = P - RotOF;
    return;
}

double VisionNode::estimateDepth(OFVectorEvents &ofVector, const cv::Vec3f T_cam) {
    double TdotD = T_cam.dot(ofVector.directionVector);
    double depth = norm(T_cam - (TdotD * ofVector.directionVector)) / norm(ofVector.P);
    return depth;
}

double VisionNode::estimateAltitude(OFVectorEvents &ofVector, double depth) {
    cv::Vec3f directionVector_body = camToBody(ofVector.directionVector, camParams);
    cv::Vec3f directionVector_inertial = bodyToInertial(directionVector_body, cosRoll, sinRoll, cosPitch, sinPitch);
    double cosTheta = directionVector_inertial[2];
    double altitude = depth * cosTheta;
    return altitude;
}

void VisionNode::airspeedCallback(const mavros_msgs::VFR_HUD::ConstPtr &msg) {
    
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

void VisionNode::velocityBodyCallback(const geometry_msgs::TwistStamped::ConstPtr &msg) {
    //THE RECEIVED VELOCITY is relative to the body frame, FLU (x : forward, y : left, z : up), LIKE IN ROS
    //the body of the drone is in FRD (x : forward, y : right, z : down) (like PX4)
    T_GPS_body_FLU = cv::Vec3f(msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z);
    T_GPS_body_FRD = cv::Vec3f(T_GPS_body_FLU[0], -T_GPS_body_FLU[1], -T_GPS_body_FLU[2]);
    T_GPS_cam_FRD = bodyToCam(T_GPS_body_FRD, camParams);
}

void VisionNode::velocityLocalCallback(const geometry_msgs::TwistStamped::ConstPtr &msg){
    //THE RECEIVED VELOCITY is relative to the world frame, ENU (x : EAST, y : NORD, z : UP), LIKE IN ROS for world reference frames
    T_GPS_local_ENU = cv::Vec3f(msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z);
    T_GPS_local_NED = cv::Vec3f(T_GPS_local_ENU[1], T_GPS_local_ENU[0], -T_GPS_local_ENU[2]);
    //ROS_INFO("LOCAL VEL NED (vx,vy,vz) : (%f , %f, %f)", T_GPS_local_NED[0], T_GPS_local_NED[1], T_GPS_local_NED[2]);

}

void VisionNode::localPositionCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
    //the received POSE is represented in ENU, so the Z coordinate becomes more and more positive every time
    // Access position data
    localPosition_ENU = cv::Vec3f(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    localPosition_NED = cv::Vec3f(localPosition_ENU[1], localPosition_ENU[0], -localPosition_ENU[2]);

}


void VisionNode::imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
{
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

}


void VisionNode::rcCallback(const mavros_msgs::RCIn::ConstPtr &msg) {
    std::vector<unsigned short int> channels = msg->channels;

    // Check channel 5 -> publishAltitude. Only if it forward, so ON if CHANNEL < 1000
    if (channels[CHAN_PUB_ALT] < 1000) {
        publishAltitude = true;
    } else {
        publishAltitude = false;
    }


    //check sensitivity @ channel 9
    //#define CHAN_SENSITIVITY 9  //10-1
    //map the values of the channel
    currentSensitivity = channels[CHAN_SENSITIVITY];   //a value from 982 to 
    
    //map current sensitivity to the values of the camera
    currentSensitivity = mapRCtoSensitivity(currentSensitivity);

    //now change the sensitivity in case it is different from the previous one
    if(currentSensitivity != previousSensitivity)
    {
        //change the sensitivity
        capture->setDVSBiasSensitivity(intToBiasSensitivity(currentSensitivity));
        std::string sens = "sensitivity = " + to_string(currentSensitivity);
        //ROS_INFO("SENSITIVITY CHANGED TO %d : ", currentSensitivity);
        sendStatusText(sens,6);
        previousSensitivity = currentSensitivity;
    }


    //check EFPS @ channel 9
    currentEFPS = channels[CHAN_EFPS];   //a value from 982 to 2006

    //map current efps to the values of the camera
    currentEFPS = mapRCtoEFPS(currentEFPS);

    if(currentEFPS != previousEFPS)
    {
        //change the efps
        capture->setDVXplorerEFPS(intToDVXeFPS(currentEFPS));
        std::string efpsstring = "efps = " + to_string(currentEFPS);
        //ROS_INFO("SENSITIVITY CHANGED TO %d : ", currentSensitivity);
        sendStatusText(efpsstring,6);
        //ROS_INFO("EFPS CHANGED TO %d : ", currentEFPS);

        previousEFPS = currentEFPS;
    }



    static bool wasCamRecordingOn = false;
    if (channels[CHAN_CAM_REC] < 1000) {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            record = true;
        }

        if (!wasCamRecordingOn) {
            ROS_INFO("START RECORDING\n");
            boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
            std::string date_str = boost::posix_time::to_iso_string(now);
            std::string date_folder = date_str.substr(6, 2) + date_str.substr(4, 2) + date_str.substr(2, 2);

            std::string base_path = "/home/sharedData/test" + date_folder;
            std::string rec_dir = base_path + "/recordings";

            if (!fs::exists(rec_dir)) fs::create_directories(rec_dir);

            std::string timestamp_str = boost::posix_time::to_iso_string(now);
            std::string rec_filename = rec_dir + "/rec" + timestamp_str + "e" + to_string(currentEFPS) + "s" + to_string(currentSensitivity) + ".aedat4";

            recFilePath = rec_filename;



            initializeRecordingFile();
            std::string files_starts = "REC Saving in " + date_folder + "\n";
            sendStatusText(files_starts, 6);
        }

        wasCamRecordingOn = true;
    } else {
        if (wasCamRecordingOn) {
            ROS_INFO("RECORDING OFF\n");

            {
                std::lock_guard<std::mutex> lock(recorder_mutex);
                if (recorder != nullptr) {
                    recorder.reset();
                }
            }


            std::string files_end = "RECORDING DONE \n";
            sendStatusText(files_end, 6);
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            record = false;
        }

        wasCamRecordingOn = false;
    }

    static bool wasReadEventsOn = false;
    if (channels[CHAN_CAM_ON] < 1000) {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            readEvents = true;
        }

        if (!wasReadEventsOn) {
            ROS_INFO("START CAM\n");
        }

        wasReadEventsOn = true;
    } else {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            readEvents = false;
        }

        if (wasReadEventsOn) {
            ROS_INFO("CAM OFF\n");


        }

        wasReadEventsOn = false;
    }
}


VisionNode::~VisionNode() {
    // Join the altitude thread before destruction
    if (altitude_thread.joinable()) {
        altitude_thread.join();
    }
}

void VisionNode::lidarCallback(const sensor_msgs::Range::ConstPtr &msg) {
    // This function should be used to get the lidar data
    // Data received from topic: /mavros/distance_sensor/hrlv_ez4_pub
    // This function should store the lidar data in a variable
    // double lidarData: LIDAR DATA IN METERS

    lidarCurrentData = msg->range;
}

void VisionNode::sendStatusText(const std::string &message, uint8_t severity) {
    mavros_msgs::StatusText status_text_msg;
    status_text_msg.text = message;
    status_text_msg.severity = severity;
    statustext_pub.publish(status_text_msg);
}


void VisionNode::altitudePublisherThread() {
    ros::Rate rate(publishAltitudeFrequency);
    while (ros::ok()) {
        bool currentReadEvents, currentRecord;

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            currentReadEvents = readEvents;
            currentRecord = record;
        }

        if (publishAltitude && currentReadEvents && !currentRecord) {
            geometry_msgs::PoseStamped pose;
            pose.header.stamp = ros::Time::now();
            {
                std::lock_guard<std::mutex> lock(altitude_mutex);
                pose.pose.position.z = filteredAltitude;
            }

            vision_pose_pub.publish(pose);

        }
        rate.sleep();
    }
}



void VisionNode::syncSystemTimeWithGPS() {
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


void VisionNode::gpsFixCallback(const sensor_msgs::NavSatFix::ConstPtr &msg) {
    // Save the latitude, longitude, and altitude from the GPS fix
    latest_gps_latitude = msg->latitude;
    latest_gps_longitude = msg->longitude;
    latest_gps_altitude = msg->altitude;

}

void VisionNode::gpsVelCallback(const geometry_msgs::TwistStamped::ConstPtr &msg) {
    // Save the linear and angular velocities from the GPS
    latest_gps_vel_x = msg->twist.linear.x;
    latest_gps_vel_y = msg->twist.linear.y;
    latest_gps_vel_z = msg->twist.linear.z;
    latest_gps_ang_vel_x = msg->twist.angular.x;
    latest_gps_ang_vel_y = msg->twist.angular.y;
    latest_gps_ang_vel_z = msg->twist.angular.z;


}



void VisionNode::initializeRecordingFile() {
    {
        std::lock_guard<std::mutex> lock(recorder_mutex);

        if (!recorder) {  // Only create a new instance if there isn't one already
            // Use capture.get() to pass a raw pointer to the constructor
            recorder = std::make_unique<dv::io::MonoCameraWriter>(recFilePath, *capture);
        }
    }
}

void VisionNode::saveEvents(const dv::EventStore &events, const dv::cvector<dv::IMU> &imu_cam) {
    if (!events.isEmpty()) {

        // Write events and get the number of bytes written
        {
            std::lock_guard<std::mutex> lock(recorder_mutex);

            written_bytes = recorder->writeEvents(events, safeEventsWrite);

        }


        // Check if imu_cam is not empty and write IMU data stream
        if (!imu_cam.empty()) {

            {
                std::lock_guard<std::mutex> lock(recorder_mutex);
            
                recorder->writeImuPacket(imu_cam);
            }
        }
    } else {
        // do nothing
    }
}




void VisionNodePlayback::applyAdaptiveSlicing()
{   
    //only apply if it implemented
    if(adaptiveSlicingEnable)
    {   
        bool updateTimingWindow = false;
        if (!filteredFlowVectors.empty()) {
            // Calculate the average length of the optical flow vectors

            double magnitude = 0.0;
            for (const auto& vec : filteredFlowVectors) {
                // in pixels
                magnitude += std::sqrt(vec.deltaX * vec.deltaX + vec.deltaY * vec.deltaY);
            }
            magnitude /= static_cast<double>(filteredFlowVectors.size());

            // Calculate PID error terms
            //if error is positive, it means the timing window is too short, and we need to increase it
            //if error is negative, it means the timing window is too long, and we need to decrease it, so the optical flow vectors will be shorter
            double error = OFPixelSetpoint - magnitude;    
            integralError += error;
            double derivative = error - previousError;

            // PID output
            PIDoutput = P_adaptiveSlicing * error + I_adaptiveSlicing * integralError + D_adaptiveSlicing * derivative;


            if (std::abs(PIDoutput) > thresholdPIDEvents && PIDoutput > 0 && adaptiveTimeWindow < maxTimingWindow)
            {
                adaptiveTimeWindow += adaptiveTimingWindowStep;
                adaptiveTimeWindow = std::min(adaptiveTimeWindow, maxTimingWindow); // Clamp to max
                integralError = 0.0;
                PIDoutput = 0.0;
                updateTimingWindow = true;
                //assign new value to the slicer
                
            }
            else if(std::abs(PIDoutput) > thresholdPIDEvents && PIDoutput < 0 && adaptiveTimeWindow > minTimingWindow)
            {
                //ROS_INFO("decrease");
                adaptiveTimeWindow -= adaptiveTimingWindowStep;
                adaptiveTimeWindow = std::max(adaptiveTimeWindow, minTimingWindow); // Clamp to min
                integralError = 0.0;
                PIDoutput = 0.0;
                updateTimingWindow = true;

            }
            else if(std::abs(PIDoutput) > thresholdPIDEvents)
            {
                //reached the saturation values, so instead of accumulating the error, reset it
                integralError = 0.0;
                PIDoutput = 0.0;

                updateTimingWindow = false;
            }

            //ROS_INFO("updateTimingWindow = %d", updateTimingWindow);
            if(updateTimingWindow)
            {
                // Update the slicer timing window
                //ROS_INFO("updating slicer");
                if(slicer->hasJob(0))
                {
                    slicer->modifyTimeInterval(0, std::chrono::milliseconds(static_cast<int>(adaptiveTimeWindow)));
                    //std::cout << "Slicer job 0 timing window updated to " << adaptiveTimeWindow << " us" << std::endl;
                }
                else if(slicer->hasJob(1))
                {
                    slicer->modifyTimeInterval(1, std::chrono::milliseconds(static_cast<int>(adaptiveTimeWindow)));
                    //std::cout << "Slicer job 1 timing window updated to " << adaptiveTimeWindow << " us" << std::endl;
                }
                else
                {
                    //std::cout << "No job found to update the timing window" << std::endl;
                
                    // Recalculate FPS
                    
                }
                fps = 1e3 / static_cast<float>(adaptiveTimeWindow);
            }
           

            // Update the previous error
            previousError = error;

            
        }

    }

    return;
}