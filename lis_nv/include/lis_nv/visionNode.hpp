#ifndef VISION_NODE_HPP
#define VISION_NODE_HPP

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
#include <mavros_msgs/RCIn.h>
#include <sensor_msgs/Range.h>
#include <mavros_msgs/StatusText.h>
#include <tf/tf.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;

class VisionNode {
public:
    VisionNode();
    void run();
    ~VisionNode();

private:
    ros::NodeHandle nh;

    ros::Subscriber imu_sub;        //for attitude data
    ros::Subscriber velocity_body_sub;
    ros::Subscriber airspeed_sub;
    ros::Subscriber rc_sub;         //for remote control switches
    ros::Subscriber lidar_sub;      //for lidar data

    ros::Publisher vision_pose_pub;    //for publishing the altitude data
    int publishAltitudeFrequency = 40;
    ros::Publisher statustext_pub;


    //RC FLAGS
    bool readEvents = false;          //READ EVENTS FROM THE CAMERA
    bool publishAltitude = false;   //publish the altitude data

    int fps;

    CAMParameters camParams;
    
    AccumulatorParameters accParams;
    FASTParameters fastParams;
    LKParams lkParams;

    int64_t slice_duration_ms;
    int64_t dtMicroseconds;
    int64_t startTime, endTime;
    int64_t currentTimestamp = 0;
    int64_t prevTimestamp = 0;
    float MEPS = 0.0;   //million event per second

    //rejection filter
    double magnitudeThresholdPixel;
    double boundThreshold;

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
    bool firstSlice = true;

    //smoothing filters
    bool smoothingFilterEnable = false;
    int smoothingFilterType = 0;
    float complementaryK = 0.5;       //coefficient for the complementary filter
    float lpfK = 0.5;                 //coefficient for the low pass filter

    float prevFilteredAltitude = 0.0;     //previous filtered altitude, OUTPUT OF THE FILTER AT TIME K-1
    float filteredAltitude = 0.0;         //filtered altitude, OUTPUT OF THE FILTER AT TIME K
    float saturationValue = 0.0;


    //altitude data
    double avgAltitude = 0.0;
    double prevAvgAltitude = 0.0;
    int altitudeType = 0;
    double unfilteredAltitude = 0.0;

    //inertial data
    cv::Vec3f T_GPS_body_FLU;       //translation vector in m/s from GPS, received in the body frame, by ROS
    cv::Vec3f T_GPS_body_FRD;       //translation vector in m/s from GPS, converted into PX4 FRD frame
    cv::Vec3f T_GPS_cam_FRD;        //translation vector in m/s from GPS, converted into the camera frame
    cv::Vec3f T_airspeed;        //translation vector in m/s
    cv::Vec3f T_groundspeed;     //translation vector in m/s
    cv::Vec3f T_cam;                 //translational vector (chosen from source) with respect to the camera frame
    cv::Vec3f avgGyro;          //average gyro data in rad/s

    int velocitySource = 0;         // 0: airspped, 1: groundspeed


    double rollRad = 0.0, pitchRad = 0.0, yawRad = 0.0; //roll, pitch, yaw in radians
    double rollDeg = 0.0, pitchDeg = 0.0;
    double cosRoll = 1.0;
    double sinRoll = 0.0;
    double cosPitch = 1.0;
    double sinPitch = 0.0;


    //SAFETY PARAMETERS
    float lidarMax = 2.0;       //from 0m to 2m above ground publish teh lidar data
    float lidarCurrentData = 0.0;   //data read from the sensor lidar

    
    dv::io::DataReadHandler handler;
    dv::MultiStreamSlicer<dv::EventStore> slicer;
    cv::Mat prevEdgeImage;
    std::vector<cv::Point2f> prevPoints;
    std::vector<OFVector> flowVectors;
    std::vector<OFVector> filteredFlowVectors;      //filtered flow vectors using the rejection filter
    int rejectedVectors = 0;                        //number of rejected vectors
    dv::cvector<dv::IMU> imu_cam;
    

    dv::io::CameraCapture capture;          //used for live from camera

    //EdgeMapAccumulator objects
    std::unique_ptr<dv::EdgeMapAccumulator> accumulator;                   // EdgeMapAccumulator object

    //FAST detector declaration
    //FAST FEATURE DETECTOR
    Ptr<FastFeatureDetector> fastDetector;                                  //creation of the fast feature detector
    int detectedFeatures = 0;                                               //number of detected features
    bool safeFeaturesApplied = false;                                       //flag to check if the safe features are applied
    int filteredDetectedFeatures = 0;                                              //number of features after the rejection filter



    void loadParameters();
    void initializeAccumulator();
    void initializeFeatureDetector();
    void initializeCamera();
    void initializeOutputFile();
    void initializeSlicer();

    void processEvents(const dv::EventStore& events, dv::cvector<dv::IMU>& imu);
    void applyCornerDetection(const cv::Mat &edgeImage);
    void calculateOpticalFlow(const cv::Mat &currFrame);
    cv::Vec3f avgGyroDataRadSec(dv::cvector<dv::IMU> &imuBatch);

    //callback functions
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
    void velocityBodyCallback(const geometry_msgs::TwistStamped::ConstPtr& msg);
    void airspeedCallback(const mavros_msgs::VFR_HUD::ConstPtr& msg); // Change to appropriate message type
    void rcCallback(const mavros_msgs::RCIn::ConstPtr& msg);
    void lidarCallback(const sensor_msgs::Range::ConstPtr &msg);


    //derotation functions
    void applyDerotation3D(OFVector &ofVector, const cv::Vec3f &avgGyroRadSec);

    //DISTANCE ESTIMATION
    double estimateDepth(OFVector &ofVector, const cv::Vec3f T_cam);
    double estimateAltitude(OFVector &ofVector, double depth);

    //publish altitude
    void altitudePublisherThread();
    std::thread altitude_thread;
    std::mutex altitude_mutex;

    //debugging
    void sendStatusText(const std::string &message, uint8_t severity);
};

#endif // VISION_NODE_HPP
