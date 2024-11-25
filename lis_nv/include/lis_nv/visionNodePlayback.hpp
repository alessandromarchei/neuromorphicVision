#ifndef VISION_NODE_PLAYBACK_HPP
#define VISION_NODE_PLAYBACK_HPP

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64.h>
#include <dv-processing/core/core.hpp>
#include <dv-processing/processing.hpp>
#include <dv-processing/imu/rotation-integrator.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/kinematics/motion_compensator.hpp>
#include <dv-processing/core/multi_stream_slicer.hpp>
#include <dv-processing/features/feature_detector.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <vector>
#include <string>
#include <lis_nv/SparseOFVector.h>  // Include the custom message header
#include <lis_nv/SparseOFVectorArray.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <defs.hpp>
#include <functions.hpp>
#include <thread>
#include <mutex>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/VFR_HUD.h>
#include <tf/tf.h>

using namespace cv;
using namespace std;

class VisionNodePlayback {
public:
    VisionNodePlayback();
    void run();

private:
    ros::NodeHandle nh;
    ros::Publisher altitude_setpoint_pub;
    //ros::Publisher imu_cam_pub;       //not used now because the derotation is applied in the vision node

    ros::Subscriber imu_sub;
    ros::Subscriber velocity_sub;
    ros::Publisher attitude_pub;
    ros::Subscriber airspeed_sub;


    int fps, delayVisualization;
    int cornerDetectionAlgorithm;

    CAMParameters camParams;
    
    AccumulatorParameters accParams;
    FASTParameters fastParams;
    LKParams lkParams;

    int64_t slice_duration_ms;
    int64_t dtMicroseconds;
    int64_t startTime, endTime;

    //rejection filter
    double magnitudeThresholdPixel;
    double boundThreshold;

    //inertial data
    cv::Vec3f T_GPS_cam;            //translation vector in m/s from GPS
    cv::Vec3f T_airspeed;        //translation vector in m/s
    cv::Vec3f T_airspeed_cam;    //translation vector in camera frame, taken from the airspeed sensor

    cv::Vec3f avgGyro;          //average gyro data in rad/s


    double rollRad = 0.0, pitchRad = 0.0, yawRad = 0.0; //roll, pitch, yaw in radians
    double cosRoll = 1.0;
    double sinRoll = 0.0;
    double cosPitch = 1.0;
    double sinPitch = 0.0;

    
    dv::io::DataReadHandler handler;
    dv::MultiStreamSlicer<dv::EventStore> slicer;
    cv::Mat prevEdgeImage;
    std::vector<cv::Point2f> prevPoints;
    std::vector<cv::Point2f> nextPrevPoints;    //used for the parallelization of FAST and LK. nextPrevPoints will be used to store the points detected by FAST and then used by LK
    std::vector<OFVector> flowVectors;
    std::vector<OFVector> filteredFlowVectors;      //filtered flow vectors using the rejection filter
    dv::cvector<dv::IMU> imu_cam;
    
    //create a pointer to monocamera recording to be initialized in case the mode is specified as playback
    std::unique_ptr<dv::io::MonoCameraRecording> playback;
    std::string aedat4Path;         //path to the aedat4 file INPUT
    std::string sensorFile;         //path to the sensor data file INPUT

    //timing files
    bool timingFileEnable = false;
    std::string timingFilePath;     //path to the timing file OUTPUT
    std::ofstream timingFileStream;       //timing file to store the timing of the events

    //altitude files
    bool altitudeFileEnable = false;
    std::string altitudeFilePath;   //path to the altitude file OUTPUT
    std::ofstream altitudeFileStream;     //altitude file to store the altitude data

    //sampling
    bool downsample = false;
    int downsampleCounter = 0;
    bool firstSlice = true;

    //smoothing filters
    bool smoothingFilterEnable = false;
    int smoothingFilterType = 0;
    float complementaryK = 0.5;       //coefficient for the complementary filter
    float lpfK = 0.5;                 //coefficient for the low pass filter

    float prevFilteredAltitude = 0.0;     //previous filtered altitude, OUTPUT OF THE FILTER AT TIME K-1
    float filteredAltitude = 0.0;         //filtered altitude, OUTPUT OF THE FILTER AT TIME K

    //altitude data
    double avgAltitude = 0.0;
    double prevAvgAltitude = 0.0;
    int altitudeType = 0;       //0 = average, 1 = median
    
    //sensor data
    std::vector<SensorData> sensorData;
    SensorData currentSensorData;

    int64_t currentTimestamp = 0;
    int64_t prevTimestamp = 0;

    float syncDelay_s = 0.0;
    int64_t syncDelay_us = 0;
    int startSliceIndex = 0;
    int endSliceIndex = 0;
    int sliceIndex = 0;



    // Background Activity Noise Filter and EdgeMapAccumulator objects
    std::unique_ptr<dv::EdgeMapAccumulator> accumulator;                   // EdgeMapAccumulator object

    //FAST detector declaration
    //FAST FEATURE DETECTOR
    Ptr<FastFeatureDetector> fastDetector;                                  //creation of the fast feature detector
    std::mutex pointsMutex;

    void loadParameters();
    void initializeAccumulator();
    void initializeOutputFile();
    void initializeFeatureDetector();
    void initializePlayback();
    void initializeSensorData();
    void initializeSlicer();

    void processEvents(const dv::EventStore& events, dv::cvector<dv::IMU>& imu);
    void applyCornerDetection(const cv::Mat &edgeImage);
    void calculateOpticalFlow(const cv::Mat &currFrame);
    cv::Vec3f avgGyroDataRadSec(dv::cvector<dv::IMU> &imuBatch);

    //callback functions
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
    void velocityCallback(const geometry_msgs::TwistStamped::ConstPtr& msg);
    void airspeedCallback(const mavros_msgs::VFR_HUD::ConstPtr& msg); // Change to appropriate message type


    //derotation functions
    void applyDerotation3D(OFVector &ofVector, const cv::Vec3f &avgGyroRadSec);

    //DISTANCE ESTIMATION
    double estimateDepth(OFVector &ofVector, const cv::Vec3f T_cam);
    double estimateAltitude(OFVector &ofVector, double depth, double rollCos, double rollSin, double pitchCos, double pitchSin);

};
#endif // VISION_NODE_HPP
