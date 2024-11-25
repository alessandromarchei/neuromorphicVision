#include <visionNodePlayback.hpp>

using namespace boost::placeholders;

VisionNodePlayback::VisionNodePlayback() : slicer("events") {
    // Load parameters from ROS parameter server
    loadParameters();

    // Initialize ROS publishers
    altitude_setpoint_pub = nh.advertise<std_msgs::Float64>("altitude_setpoint", static_cast<int>(2 * (fastParams.desiredFeatures)));
    imu_sub = nh.subscribe("/mavros/imu/data", 100, &VisionNodePlayback::imuCallback, this);
    velocity_sub = nh.subscribe("/mavros/local_position/velocity_local", 100, &VisionNodePlayback::velocityCallback, this);
    airspeed_sub = nh.subscribe("/mavros/vfr_hud", 100, &VisionNodePlayback::airspeedCallback, this);

    initializePlayback();
    initializeOutputFile();
    initializeSensorData();
    initializeAccumulator();
    initializeFeatureDetector();
    initializeSlicer();


    // Main loop
    while (ros::ok()) {
        if (!playback->handleNext(handler)) {
            // Playback finished
            ROS_INFO("Playback finished. Shutting down...");
            ros::shutdown();
            // Kill the ROS master
            std::system("killall -9 rosmaster");
            break;
        }
        ros::spinOnce();
    }

    // Close timing file
    if (timingFileStream.is_open()) {
        timingFileStream.close();
    }

    if(altitudeFileStream.is_open()){
        altitudeFileStream.close();
    }
}

void VisionNodePlayback::run() {
    ros::AsyncSpinner spinner(std::thread::hardware_concurrency());
    spinner.start();
    ros::waitForShutdown();
}

void VisionNodePlayback::loadParameters() {
    // GET THE INPUT PARAMETERS, Live or from a file (.aedat4)
    // PLAYBACK MODE
    nh.getParam("/PLAYBACK/filepath", aedat4Path);
    nh.getParam("/sensorFile", sensorFile);
    
    // Load timing file path
    nh.getParam("/timingFileEnable", timingFileEnable);
    if(timingFileEnable) nh.getParam("/timingFilePath", timingFilePath);

    //load the altitude file path
    nh.getParam("/altitudeFileEnable", altitudeFileEnable);
    if(altitudeFileEnable) nh.getParam("/altitudeFilePath", altitudeFilePath);


    //load data for synchronization delay
    nh.getParam("/syncDelay", syncDelay_s);
    nh.getParam("/startSliceIndex", startSliceIndex);
    nh.getParam("/endSliceIndex", endSliceIndex);

    // Get FAST parameters
    nh.getParam("/FAST/threshold", fastParams.threshold);
    nh.getParam("/FAST/nonmaxSuppression", fastParams.nonmaxSuppression);
    nh.getParam("/FAST/randomSampleFilter/enable", fastParams.randomSampleFilterEnable);
    nh.getParam("/FAST/randomSampleFilter/ratio", fastParams.randomSampleFilterRatio);
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

    // Load downsample parameter
    nh.getParam("/downsample", downsample);



    syncDelay_us = syncDelay_s * 1e6;


    return;
}

void VisionNodePlayback::initializeOutputFile() {
    // Open timing file
    if(timingFileEnable)
    {
        timingFileStream.open(timingFilePath, std::ios::out);
        if (timingFileStream.is_open()) {
            //writing headers
            timingFileStream << "Timestamp,Accumulate,OpticalFlow,FeatureDetection,TotalSlice\n";
        }
    }
    
    //initialize altitude file
    if(altitudeFileEnable){
        altitudeFileStream.open(altitudeFilePath, std::ios::out);
        if(altitudeFileStream.is_open()){
            altitudeFileStream << "Timestamp,DistanceGround,0,FilteredAvgAltitude,Vx,Vy,Vz,Airspeed,RollAngle,PitchAngle\n";
        }
    }   

    return;
}


void VisionNodePlayback::initializeFeatureDetector() {
    // Initialize the FAST detector
    fastDetector = FastFeatureDetector::create(fastParams.threshold, fastParams.nonmaxSuppression);

    return;
}

void VisionNodePlayback::initializeAccumulator() {
    accumulator = std::make_unique<dv::EdgeMapAccumulator>(camParams.resolution, accParams.eventContribution,
                                                           accParams.ignorePolarity, accParams.neutralPotential, accParams.decay);
    ROS_INFO("Accumulator initialized.");

    return;
}


void VisionNodePlayback::initializePlayback() {
    ROS_INFO("Opening file %s", aedat4Path.c_str());
    playback = std::make_unique<dv::io::MonoCameraRecording>(aedat4Path);
    return;
}

void VisionNodePlayback::initializeSensorData() {

    ROS_INFO("Opening sensor data file %s", sensorFile.c_str());

    //load data into the vector
    sensorData = parseSensorData(sensorFile);

    return;
}

void VisionNodePlayback::initializeSlicer() {
    handler.mEventHandler = [this](const auto &events) {
        slicer.accept("events", events);
    };

    slicer.addStream<dv::cvector<dv::IMU>>("imu");
    handler.mImuHandler = [this](auto &imu) {
        slicer.accept("imu", imu);
    };

    slicer.doEveryTimeInterval(std::chrono::milliseconds(1000 / fps), [this](const auto &data) {
        if (sliceIndex > startSliceIndex && sliceIndex < endSliceIndex)
        {                    
            auto startSlice = std::chrono::high_resolution_clock::now();

            auto events = data.template get<dv::EventStore>("events");
            imu_cam = data.template get<dv::cvector<dv::IMU>>("imu");

            //get data from the file 
            try {
                currentTimestamp = dv::packets::getPacketTimestamp<dv::packets::Timestamp::START>(events);
            } catch (const std::exception &e) {
                //use the old timestamp instead
                currentTimestamp = prevTimestamp + 1e6 / fps;
            }

            // Get closest sensor data
            currentSensorData = getClosestSensorData(sensorData, currentTimestamp + syncDelay_us);


            processEvents(events, imu_cam);

            auto endSlice = std::chrono::high_resolution_clock::now();
            double sliceDuration = std::chrono::duration_cast<std::chrono::microseconds>(endSlice - startSlice).count();

            if (timingFileStream.is_open()) {
                timingFileStream << "," << sliceDuration << "\n";
            }
        }
        sliceIndex++;
        
    });
}

void VisionNodePlayback::applyCornerDetection(const cv::Mat &edgeImage) {
    std::vector<cv::KeyPoint> keypoints;

    fastDetector->detect(edgeImage, keypoints, edgeImage);

    // Randomly sample keypoints
    if (fastParams.randomSampleFilterEnable) {
        keypoints = randomlySampleKeypoints(keypoints, fastParams.desiredFeatures, fastParams.randomSampleFilterRatio);
    }

    // Apply gradient scoring if enabled
    if (fastParams.gradientScoringEnable) {
        keypoints = scoreAndRankKeypointsUsingGradient(keypoints, edgeImage, fastParams.desiredFeatures);
    }

    // If neither random sampling nor gradient scoring is applied, perform random sampling to get the desired number of features
    if (!fastParams.randomSampleFilterEnable && !fastParams.gradientScoringEnable) {
        keypoints = randomlySampleKeypoints(keypoints, fastParams.desiredFeatures, 0.0);
    }

    cv::KeyPoint::convert(keypoints, prevPoints);

    return;
}

void VisionNodePlayback::processEvents(const dv::EventStore &events, dv::cvector<dv::IMU> &imu) {
    auto startTotal = std::chrono::high_resolution_clock::now();

    auto startAccumulate = std::chrono::high_resolution_clock::now();
    accumulator->accept(events);
    cv::Mat currEdgeImage = accumulator->generateFrame().image;

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


    //only perform calculations if downsample is not active or if it is not the frist slice
    if(downsample == false || (downsample == true && firstSlice == false))
    {   
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

        if (std::isnan(filteredAltitude)) {
            prevFilteredAltitude = 0;
        } else {
            prevFilteredAltitude = filteredAltitude;
        }


        flowVectors.clear();
        filteredFlowVectors.clear();

        //eventually write the altitudes into the file
        if (altitudeFileStream.is_open()) {
            altitudeFileStream << currentTimestamp << "," << currentSensorData.distance_ground << "," << 0 << "," << filteredAltitude << "," << currentSensorData.vx << "," << currentSensorData.vy << "," << currentSensorData.vz << "," << currentSensorData.airspeed << "," << currentSensorData.roll_angle << "," << currentSensorData.pitch_angle << "\n";
        }
    }
    

    auto endTotal = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTotal - startTotal).count();

    //always write timings
    if (timingFileStream.is_open()) {
        timingFileStream << currentTimestamp << "," << accumulateTime << "," << opticalFlowTime << "," << featureDetectionTime;
    }


    firstSlice = !firstSlice;
}


void VisionNodePlayback::calculateOpticalFlow(const cv::Mat &currEdgeImage) {
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
        //PRINT NUMBER OF FLOW VECTORS
        avgGyro = avgGyroDataRadSec(imu_cam);

        for (auto& ofVector : filteredFlowVectors) {
            applyDerotation3D(ofVector, avgGyro);
        }
    }
}

cv::Vec3f VisionNodePlayback::avgGyroDataRadSec(dv::cvector<dv::IMU>& imuBatch) {
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

void VisionNodePlayback::applyDerotation3D(OFVector &ofVector, const cv::Vec3f &avgGyroRadSec) {
    double norm_a = norm(ofVector.AMeter);
    cv::Vec2f Pprime_ms(ofVector.uPixelSec * camParams.pixelSize, ofVector.vPixelSec * camParams.pixelSize);
    cv::Vec3f PpPprime_ms(Pprime_ms[0] / norm_a, Pprime_ms[1] / norm_a, 0);
    cv::Vec3f P = PpPprime_ms - (PpPprime_ms.dot(ofVector.directionVector) * ofVector.directionVector);
    cv::Vec3f RotOF = -avgGyroRadSec.cross(ofVector.directionVector);
    ofVector.P = P - RotOF;
    return;
}

double VisionNodePlayback::estimateDepth(OFVector &ofVector, const cv::Vec3f T_cam) {
    double TdotD = T_cam.dot(ofVector.directionVector);
    double depth = norm(T_cam - (TdotD * ofVector.directionVector)) / norm(ofVector.P);
    return depth;
}

double VisionNodePlayback::estimateAltitude(OFVector &ofVector, double depth, double rollCos, double rollSin, double pitchCos, double pitchSin) {
    cv::Vec3f directionVector_body = camToBody(ofVector.directionVector, camParams);
    cv::Vec3f directionVector_inertial = bodyToInertial(directionVector_body, rollCos, rollSin, pitchCos, pitchSin);
    double cosTheta = directionVector_inertial[2];
    double altitude = depth * cosTheta;
    return altitude;
}

void VisionNodePlayback::airspeedCallback(const mavros_msgs::VFR_HUD::ConstPtr &msg) {
    T_airspeed = cv::Vec3f(msg->airspeed, 0, 0); // in m/s
    T_airspeed_cam = bodyToCam(T_airspeed, camParams);
}

void VisionNodePlayback::velocityCallback(const geometry_msgs::TwistStamped::ConstPtr &msg) {
    T_GPS_cam = bodyToCam(cv::Vec3f(msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z), camParams);
}

void VisionNodePlayback::imuCallback(const sensor_msgs::Imu::ConstPtr &msg) {
    tf::Quaternion q(
        msg->orientation.x,
        msg->orientation.y,
        msg->orientation.z,
        msg->orientation.w);

    tf::Matrix3x3 m(q);
    m.getRPY(rollRad, pitchRad, yawRad);

    cosRoll = std::cos(rollRad);
    sinRoll = std::sin(rollRad);
    cosPitch = std::cos(pitchRad);
    sinPitch = std::sin(pitchRad);
}