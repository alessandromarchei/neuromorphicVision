#include <visionNode.hpp>

using namespace boost::placeholders;

VisionNode::VisionNode() : slicer("events") {
    // Load parameters from ROS parameter server
    loadParameters();

    // Initialize ROS subscribers
    imu_sub = nh.subscribe("/mavros/imu/data", 100, &VisionNode::imuCallback, this);
    velocity_body_sub = nh.subscribe("/mavros/local_position/velocity_body", 100, &VisionNode::velocityBodyCallback, this);
    airspeed_sub = nh.subscribe("/mavros/vfr_hud", 100, &VisionNode::airspeedCallback, this);
    rc_sub = nh.subscribe("/mavros/rc/in", 100, &VisionNode::rcCallback, this);
    lidar_sub = nh.subscribe("/mavros/distance_sensor/hrlv_ez4_pub", 100, &VisionNode::lidarCallback, this);


    // Initialize ROS publisher on /mavros/vision_pose/pose
    vision_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 100);
    statustext_pub = nh.advertise<mavros_msgs::StatusText>("/mavros/statustext/send", 100);


    initializeCamera();
    initializeAccumulator();
    initializeFeatureDetector();
    initializeSlicer();

    // Start the thread for altitude publishing
    altitude_thread = std::thread(&VisionNode::altitudePublisherThread, this);


    // Send the STATUS TEXT saying the RPI5 is connected correctly, 3 times
    std::string rpi5_setup = "RPI5 CONNECTED";
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        sendStatusText(rpi5_setup, 6); // Send with severity status = 6 (INFO)
    }


        // Main loop
    while (ros::ok() && capture.handleNext(handler)) {
        ros::spinOnce();
    }
}

void VisionNode::run() {
    ros::AsyncSpinner spinner(std::thread::hardware_concurrency());
    spinner.start();
    ros::waitForShutdown();
}

void VisionNode::loadParameters() {
    // GET THE INPUT PARAMETERS, Live or from a file (.aedat4)
    
    // Load timing file path
    nh.getParam("/timingFileEnable", timingFileEnable);
    //then write the timing file name with the current timestamp

    //load the altitude file path
    nh.getParam("/altitudeFileEnable", altitudeFileEnable);


    // Get FAST parameters
    nh.getParam("/FAST/threshold", fastParams.threshold);
    nh.getParam("/FAST/nonmaxSuppression", fastParams.nonmaxSuppression);
    nh.getParam("/FAST/randomSampleFilter/enable", fastParams.randomSampleFilterEnable);
    nh.getParam("/FAST/randomSampleFilter/ratio", fastParams.randomSampleFilterRatio);
    nh.getParam("/FAST/gradientScoring/enable", fastParams.gradientScoringEnable);
    nh.getParam("/FAST/gradientScoring/desiredFeatures", fastParams.desiredFeatures);
        //safeFeatures : use the SOBEL FILTER in case of no features detected
    nh.getParam("/FAST/safeFeatures", fastParams.safeFeatures);

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

    // Load downsample parameter
    nh.getParam("/downsample", downsample);


    //value to start using the lidar, from 0 m to lidar_max m
    nh.getParam("/lidarMax", lidarMax);


    //frequency to publish altitude
    nh.getParam("/publishAltitudeFrequency", publishAltitudeFrequency);

    //select the source of velocity
    nh.getParam("/velocitySource", velocitySource);     //0 : airspped, 1 : groundspeed

    return;
}

void VisionNode::initializeOutputFile() {
    // Open timing file
    if(timingFileEnable)
    {
        timingFileStream.open(timingFilePath, std::ios::out);
        if (timingFileStream.is_open()) {
            ROS_INFO("Timing file opened, with name : %s", timingFilePath.c_str());
            //writing headers
            timingFileStream << "Timestamp,Accumulate,OpticalFlow,FeatureDetection,TotalProcessingTime,detectedFeatures,filteredDetectedFeatures,safetyFeaturesApplied,rejectedVectors,MEPS\n";
        }
    }
    
    //initialize altitude file
    if(altitudeFileEnable){
        altitudeFileStream.open(altitudeFilePath, std::ios::out);
        if(altitudeFileStream.is_open()){
            ROS_INFO("Altitude file opened, with name : %s", altitudeFilePath.c_str());
            altitudeFileStream << "Timestamp,estimatedAltitude,unfilteredAltitude,lidarData,Vx_body_FRD,Vy_body_FRD,Vz_body_FRD,Airspeed,Groundspeed,RollAngle,PitchAngle\n";
        }
    }   

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

    //LIVE MODE
    dv::io::CameraCapture::DVXeFPS efps = intToDVXeFPS(camParams.efps);
    dv::io::CameraCapture::BiasSensitivity sensitivity = intToBiasSensitivity(camParams.sensitivity);

    ROS_INFO("Camera [%s] has been opened!", capture.getCameraName().c_str());

    capture.setDVXplorerEFPS(efps);
    capture.setDVSBiasSensitivity(sensitivity);
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

        if(readEvents)
        {
            //read events only if the button is pressed
            //ROS_INFO("slicing");
            auto startSlice = std::chrono::high_resolution_clock::now();
            
            auto events = data.template get<dv::EventStore>("events");
            imu_cam = data.template get<dv::cvector<dv::IMU>>("imu");

            //save the MEPS value
            try {
                MEPS = events.size() * fps / 1e6;
            } catch (const std::exception &e) {
                //IN CASE OF ERROR, SET MEPS TO 0
                MEPS = 0;
            }

            prevTimestamp = currentTimestamp;
            try {
                currentTimestamp = dv::packets::getPacketTimestamp<dv::packets::Timestamp::START>(events);
            } catch (const std::exception &e) {
                //use the old timestamp instead
                currentTimestamp = prevTimestamp + 1e6 / fps;
            }

            //processEvents computes the optical flow with LK and concurrently the feature detection 
            processEvents(events, imu_cam);
            //accumulator->accept(events);
            //cv::Mat currEdgeImage = accumulator->generateFrame().image;
            //cv::imshow("events",currEdgeImage);
            //cv::waitKey(1);

            auto endSlice = std::chrono::high_resolution_clock::now();
            double sliceDuration = std::chrono::duration_cast<std::chrono::microseconds>(endSlice - startSlice).count();

            if (timingFileStream.is_open()) {
                timingFileStream << "," << sliceDuration << "," << detectedFeatures << "," << filteredDetectedFeatures << "," << safeFeaturesApplied << "," << rejectedVectors << "," << MEPS <<"\n";
            }
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

    // Randomly sample keypoints
    if (fastParams.randomSampleFilterEnable) {

        keypoints = randomlySampleKeypoints(keypoints, fastParams.desiredFeatures, fastParams.randomSampleFilterRatio);
    }

    // Apply gradient scoring if enabled

    if (fastParams.gradientScoringEnable) {
        keypoints = scoreAndRankKeypointsUsingGradient(keypoints, edgeImage, fastParams.desiredFeatures);
    }

    //CHECK IF THE NUMBER OF DETECTED FEATURES IS < 0.5 * DESIRED FEATURES, and so check if the safeFeatures is enabled
    if(!fastParams.randomSampleFilterEnable && !fastParams.gradientScoringEnable)
    {
        if(fastParams.safeFeatures && detectedFeatures < 0.5*fastParams.desiredFeatures)
        {
            //set the flag up for the safeFeatures = ON
            safeFeaturesApplied = true;

            //select a lower threshold for the fast detector, 50 to make sure features are detected
            fastDetector->setThreshold(50); //threshold = 50

            keypoints.clear();
            //apply again the fast detection with the new threshold
            fastDetector->detect(edgeImage, keypoints);

            //now apply the randomsamplefilter and the gradient scoring
            // Randomly sample keypoints with a ratio of 0.5
            keypoints = randomlySampleKeypoints(keypoints, fastParams.desiredFeatures, 0.5);

            // Apply gradient scoring
            keypoints = scoreAndRankKeypointsUsingGradient(keypoints, edgeImage, fastParams.desiredFeatures);

            //restore the previous threshold
            fastDetector->setThreshold(fastParams.threshold);
        }
        // If neither random sampling nor gradient scoring is applied, perform random sampling to get the desired number of features
        else {
            //apply the random sample filter normally
            keypoints = randomlySampleKeypoints(keypoints, fastParams.desiredFeatures, 0.0);
        }
    }
    

    filteredDetectedFeatures = keypoints.size();
    cv::KeyPoint::convert(keypoints, prevPoints);

    return;
}


void VisionNode::processEvents(const dv::EventStore &events, dv::cvector<dv::IMU> &imu) {
    auto startTotal = std::chrono::high_resolution_clock::now();

    auto startAccumulate = std::chrono::high_resolution_clock::now();
    accumulator->accept(events);
    cv::Mat currEdgeImage = accumulator->generateFrame().image;
    //show the image

    auto endAccumulate = std::chrono::high_resolution_clock::now();
    double accumulateTime = std::chrono::duration_cast<std::chrono::microseconds>(endAccumulate - startAccumulate).count();

    auto startOpticalFlow = std::chrono::high_resolution_clock::now();
    if(downsample == false || (downsample == true && firstSlice == false)){
        //perform the optical flow only in alternated manner
        calculateOpticalFlow(currEdgeImage);
    }

    auto endOpticalFlow = std::chrono::high_resolution_clock::now();
    double opticalFlowTime = std::chrono::duration_cast<std::chrono::microseconds>(endOpticalFlow - startOpticalFlow).count();

    prevPoints.clear();

    auto startFeatureDetection = std::chrono::high_resolution_clock::now();
    if(downsample == false || (downsample == true && firstSlice == true)){
        //always perform corner detection
        applyCornerDetection(currEdgeImage);
    }

    auto endFeatureDetection = std::chrono::high_resolution_clock::now();
    double featureDetectionTime = std::chrono::duration_cast<std::chrono::microseconds>(endFeatureDetection - startFeatureDetection).count();

    prevEdgeImage = currEdgeImage.clone();

    //print the number of detected features
    //ROS_INFO("Number of detected features : %d", prevPoints.size());

    //only perform calculations if downsample is not active or if it is not the frist slice
    if(downsample == false || (downsample == true && firstSlice == false))
    {   

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

        //eventually write the altitudes into the file
        if (altitudeFileStream.is_open()) {
            altitudeFileStream << currentTimestamp << "," << filteredAltitude << "," << unfilteredAltitude << "," << lidarCurrentData <<"," << T_GPS_body_FRD[0] << "," << T_GPS_body_FRD[1] << "," << T_GPS_body_FRD[2] << "," << norm(T_airspeed) << "," << norm(T_groundspeed) << "," << rollDeg << "," << pitchDeg << "\n";
        }
    }
    
    //publish altitude on the /mavros/vision_pose/pose topic, with only z value, positive because it is respected to ENU frame
    //publishAltitudeData(filteredAltitude);

    auto endTotal = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTotal - startTotal).count();

    //always write timings
    if (timingFileStream.is_open()) {
        timingFileStream << currentTimestamp << "," << accumulateTime << "," << opticalFlowTime << "," << featureDetectionTime;
    }


    firstSlice = !firstSlice;
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

                OFVector ofVector(prevPoints[i], currPoints[i], fps, camParams);
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


void VisionNode::applyDerotation3D(OFVector &ofVector, const cv::Vec3f &avgGyroRadSec) {
    double norm_a = norm(ofVector.AMeter);
    cv::Vec2f Pprime_ms(ofVector.uPixelSec * camParams.pixelSize, ofVector.vPixelSec * camParams.pixelSize);
    cv::Vec3f PpPprime_ms(Pprime_ms[0] / norm_a, Pprime_ms[1] / norm_a, 0);
    cv::Vec3f P = PpPprime_ms - (PpPprime_ms.dot(ofVector.directionVector) * ofVector.directionVector);
    cv::Vec3f RotOF = -avgGyroRadSec.cross(ofVector.directionVector);
    ofVector.P = P - RotOF;
    return;
}

double VisionNode::estimateDepth(OFVector &ofVector, const cv::Vec3f T_cam) {
    double TdotD = T_cam.dot(ofVector.directionVector);
    double depth = norm(T_cam - (TdotD * ofVector.directionVector)) / norm(ofVector.P);
    return depth;
}

double VisionNode::estimateAltitude(OFVector &ofVector, double depth) {
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


void VisionNode::imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
{
    //THIS FUNCTION SERVES AS COMPUTING THE ROLL AND PITCH ANGLES

    
    //data received from topic : /mavros/imu/data
    tf::Quaternion q(
        msg->orientation.x,
        msg->orientation.y,
        msg->orientation.z,
        msg->orientation.w);

    //save values into the roll, pitch, yaw variables, in radiants

    tf::Matrix3x3 m(q);
    m.getRPY(rollRad, pitchRad, yawRad);

    //already compute the sin and cosin for fast computation during the rotation matrix computation
    cosRoll = std::cos(rollRad);
    sinRoll = std::sin(rollRad);
    cosPitch = std::cos(pitchRad);
    sinPitch = std::sin(pitchRad);

    //convert the roll and pitch angles in degrees for writing into the file
    rollDeg = rollRad * (180.0 / CV_PI);
    pitchDeg = pitchRad * (180.0 / CV_PI);
    //ROS_INFO("Roll : %f, Pitch : %f", rollDeg, pitchDeg);


    //x axis = ROLL     : X = forward
    //y axis = PITCH    : Y = right
    //z axis = YAW      : Z = downwards
}


void VisionNode::rcCallback(const mavros_msgs::RCIn::ConstPtr &msg) {
    // This function should be used to get the RC data
    // Data received from topic: /mavros/rc/in
    // This function samples the RC channels and then modifies the variables
    // bool readEvents: READ EVENTS FROM THE CAMERA
    // bool publishAltitude: PUBLISH ALTITUDE DATA onto the topic

    std::vector<unsigned short int> channels = msg->channels;

    // Check channel 5 -> publishAltitude. Only if it forward, so ON if CHANNEL < 1000
    if (channels[4] < 1000) {
        publishAltitude = true;
    } else {
        publishAltitude = false;
    }

    // Check channel 8 -> readEvents. Only if it forward, so ON if CHANNEL < 1000
    static bool wasReadEventsOn = false; // To detect rising and falling edges
    if (channels[7] < 1000) {
        readEvents = true;

        if (!wasReadEventsOn) { // Rising edge detected
            ROS_INFO("START CAM\n");
            // Get current time
            //ROS_INFO("rising edge detected");
            boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
            std::string date_str = boost::posix_time::to_iso_string(now);
            std::string date_folder = date_str.substr(6, 2) + date_str.substr(4, 2) + date_str.substr(2, 2);

            // Create directory paths
            std::string base_path = "/home/sharedData/test" + date_folder;
            std::string altitude_dir = base_path + "/altitude";
            std::string time_dir = base_path + "/time";

            // Create directories if they do not exist
            if (!fs::exists(altitude_dir)) fs::create_directories(altitude_dir);
            if (!fs::exists(time_dir)) fs::create_directories(time_dir);

            // Create file names
            std::string timestamp_str = boost::posix_time::to_iso_string(now);
            std::string altitude_filename = altitude_dir + "/altitude_" + timestamp_str + ".csv";
            std::string time_filename = time_dir + "/timing_" + timestamp_str + ".csv";

            // Store the file paths for future use
            altitudeFilePath = altitude_filename;
            timingFilePath = time_filename;

            // Initialize files
            //ROS_INFO("Initializing output files");
            initializeOutputFile();


            //SEND STATUS TEXT
            std::string files_starts = "Saving in "+ date_folder + "\n";
            sendStatusText(files_starts, 6);

        }

        wasReadEventsOn = true;
    } else {
        readEvents = false;

        if (wasReadEventsOn) { // Falling edge detected
            ROS_INFO("CAM OFF\n");

            // Close the files
            if (timingFileStream.is_open()) {
                //ROS_INFO("Closing timing file named %s", timingFilePath.c_str());
                timingFileStream.close();
            }
            if (altitudeFileStream.is_open()) {
                //ROS_INFO("Closing altitude file named %s", altitudeFilePath.c_str());
                altitudeFileStream.close();
            }

            //SEND STATUS TEXT
            std::string files_end = "SAVING DONE \n";
            sendStatusText(files_end, 6);
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
        if (publishAltitude) {
            geometry_msgs::PoseStamped pose;
            pose.header.stamp = ros::Time::now();
            {
                std::lock_guard<std::mutex> lock(altitude_mutex);
                pose.pose.position.z = filteredAltitude;
            }

            //check if the altitude is within the lidar range, and in that case use it
            if(lidarCurrentData < lidarMax)
            {
                //use the lidar data because we are too close to the ground
                pose.pose.position.z = lidarCurrentData;
            }

            //ROS_INFO("Publishing altitude data: %f", pose.pose.position.z);
            vision_pose_pub.publish(pose);
        }
        rate.sleep();
    }
}



//void VisionNode::publishAltitudeData(double altitude) {
//    if (publishAltitude) {
//        geometry_msgs::PoseStamped pose;
//        pose.header.stamp = ros::Time::now();
//        {
//            std::lock_guard<std::mutex> lock(altitude_mutex);
//            pose.pose.position.z = altitude;
//        }
//
//        //check if the altitude is within the lidar range, and in that case use it
//        if(lidarCurrentData < lidarMax)
//        {
//            //use the lidar data because we are too close to the ground
//            pose.pose.position.z = lidarCurrentData;
//        }
//        else
//        {
//            //actually use the value passed as paramter, so the VISION ESTIMATED altitude data
//            pose.pose.position.z = altitude;
//        }
//        
//        //ROS_INFO("Publishing altitude data: %f", pose.pose.position.z);
//        vision_pose_pub.publish(pose);
//    }
//}
//