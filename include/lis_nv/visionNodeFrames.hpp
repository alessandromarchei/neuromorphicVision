#ifndef VISION_NODE_FRAMES_HPP
#define VISION_NODE_FRAMES_HPP

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/TimeReference.h>
#include <sensor_msgs/NavSatFix.h>
#include <std_msgs/UInt32.h>
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
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/VFR_HUD.h>
#include <mavros_msgs/RCIn.h>
#include <sensor_msgs/Range.h>
#include <mavros_msgs/StatusText.h>
#include <tf/tf.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/lockfree/queue.hpp>
#include <atomic>
#include <numeric> // For std::accumulate
#include <opencv2/videoio.hpp>


//here include the libraries for the FLIR CAMERA : SPINNAKER SDK
#include <Spinnaker.h>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

using namespace cv;

class VisionNodeFrames {
public:
    VisionNodeFrames();
    void run();
    ~VisionNodeFrames();

private:
    ros::NodeHandle nh;

    ros::Subscriber imu_sub;        //for attitude data
    ros::Subscriber velocity_body_sub;
    ros::Subscriber velocity_local_sub;
    ros::Subscriber local_pos;
    ros::Subscriber airspeed_sub;
    ros::Subscriber rc_sub;         //for remote control switches
    ros::Subscriber lidar_sub;      //for lidar data
    ros::Subscriber global_pos;
    ros::Subscriber gps_fix_sub;
    ros::Subscriber rel_alt_sub;
    ros::Subscriber gps_vel_sub;

    ros::Publisher vision_pose_pub;    //for publishing the altitude data
    int publishAltitudeFrequency = 40;
    ros::Publisher statustext_pub;


    //RC FLAGS
    std::mutex state_mutex;
    bool readFrames = false;          //READ EVENTS FROM THE CAMERA
    bool publishAltitude = false;   //publish the altitude data
    bool record = false;            //record events from the camera
    cv::VideoWriter recorder;       //pointer to the recordin object
    std::string recFilePath;        //path name for recording
    bool isRecording = false;       //true only when recording
    std::mutex recorder_mutex;      //mutex for the recording parameters, avoid conflicts


    double fps;                        //frames of the camera. it would be (exposureTime)^-1
    double deltaTms;                //exposuretime between 2 consecutive frames, in ms
    cv::Mat prevImage, currImage;  

    
    CAMFrameParameters camParams;   //parameters of the FLIR camera
    DeblurParameters deblurParams;        //parameters for deblurring the images
    FASTParameters fastParams;
    LKParams lkParams;

    bool synchronizedTimestamp = false;     //flag to check if the system time is synchronized with the RPI5

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


    //imu files
    bool imuFileEnable = false;
    std::string imuFilePath;   //path to the imu file OUTPUT
    std::ofstream imuFileStream;     //imu file to store the altitude data


    //GPS file
    bool gpsFileEnable = false;
    std::string gpsFilePath;   //path to the imu file OUTPUT
    std::ofstream gpsFileStream;     //imu file to store the altitude data
    std::string gpsLocalFilePath;
    std::ofstream gpsLocalFileStream;

    bool showOF = false;     //true for showing video with OF vectors : for debugging
    bool showOFDerotated = false;


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

    //gps data
    int gpsTimeoutSeconds = 0;
    bool has_fix = false;
    bool has_velocity = false;
    bool has_local_vel = false;
    bool has_local_pos = false;

    uint64_t latest_timestamp = 0;
    uint64_t latest_timestamp_local = 0;
    double latest_gps_latitude = 0.0;
    double latest_gps_longitude = 0.0;
    double latest_gps_altitude = 0.0;
    double latest_gps_vel_x = 0.0;
    double latest_gps_vel_y = 0.0;
    double latest_gps_vel_z = 0.0;
    double latest_gps_ang_vel_x = 0.0;
    double latest_gps_ang_vel_y = 0.0;
    double latest_gps_ang_vel_z = 0.0;


    //inertial data
    cv::Vec3f T_GPS_body_FLU;       //translation vector in m/s from GPS, received in the body frame, by ROS
    cv::Vec3f T_GPS_body_FRD;       //translation vector in m/s from GPS, converted into PX4 FRD frame
    cv::Vec3f T_GPS_cam_FRD;        //translation vector in m/s from GPS, converted into the camera frame
    cv::Vec3f T_GPS_local_ENU;      //velocity in world frame in ROS reference system ENU
    cv::Vec3f T_GPS_local_NED;      //velocity in world frame in PX4 reference system NED
    cv::Vec3f T_airspeed;        //translation vector in m/s
    cv::Vec3f T_groundspeed;     //translation vector in m/s
    cv::Vec3f T_cam;                 //translational vector (chosen from source) with respect to the camera frame
    cv::Vec3f avgGyro_rad_cam;          //average gyro data in rad/s, in the camera frame
    cv::Vec3f localPosition_ENU;    //@ 30 hz
    cv::Vec3f localPosition_NED;
    tf::Quaternion quat_imu;    //@ 50 hz
    float z_local;              //z local coordinate read from the mavros topic

    int velocitySource = 0;         // 0: airspped, 1: groundspeed
    int gyroBufferLength = 5;       //buffer length to store the values of the gyro daa
    std::vector<cv::Vec3f> gyroBuffer;                //vector storing the last gyro data

    double rollRad = 0.0, pitchRad = 0.0, yawRad = 0.0; //roll, pitch, yaw in radians
    double rollDeg = 0.0, pitchDeg = 0.0;
    double cosRoll = 1.0;
    double sinRoll = 0.0;
    double cosPitch = 1.0;
    double sinPitch = 0.0;

    // Thread safety
    std::mutex recorderMutex;
    //SAFETY PARAMETERS
    float lidarMax = 2.0;       //from 0m to 2m above ground publish teh lidar data
    float lidarCurrentData = 0.0;   //data read from the sensor lidar

    
    std::vector<cv::Point2f> prevPoints, nextPrevPoints;
    std::vector<OFVectorFrame> flowVectors;
    std::vector<OFVectorFrame> filteredFlowVectors;      //filtered flow vectors using the rejection filter
    int rejectedVectors = 0;                        //number of rejected vectors
    
    uint64_t currentTimestamp = 0;                     //timestamp when the image has been acquired
    uint64_t frameID = 0;                           //ID of the current frame
    double ofTime = 0;
    double featureDetectionTime = 0;
    double totalProcessingFrameTime = 0;
    double getImageTime = 0;
    double resizeImageTime = 0;

    //FAST detector declaration
    //FAST FEATURE DETECTOR
    Ptr<FastFeatureDetector> fastDetector;                                  //creation of the fast feature detector
    int detectedFeatures = 0;                                               //number of detected features
    bool safeFeaturesApplied = false;                                       //flag to check if the safe features are applied
    int filteredDetectedFeatures = 0;                                              //number of features after the rejection filter

    //FLIR CAMERA POINTERS
    Spinnaker::SystemPtr systemCam = Spinnaker::System::GetInstance();
    Spinnaker::CameraList camList = systemCam->GetCameras();      //get the number of cameras. It should be 1
    Spinnaker::CameraPtr pCam;      //smart MAIN CAMERA POINTER
    Spinnaker::ImagePtr pImage;     //smart pointer of the image

    //PARALLEL FAST AND LK
    bool FASTLKParallel = false;
     //the thread for FAST, in parallel with LK. IT WILL PRODUCE nextprevPoints, that will be then saved in prevPoints
    std::thread FASTThread;    


    void loadParameters();
    void initializeFeatureDetector();
    void initializeCamera();
    void configureExposure(CameraPtr pCam, int exposure);
    void initializeRecordingFile(const std::string& filePath);

    //recorder member functions
    void saveFrameToVideo();
    void closeRecorder();

    cv::Mat& acquireImage();        //return the image in form of a cv::Mat
    void processFrames();
    void processFramesParallel();
    void applyCornerDetection(const cv::Mat &image);
    void applyCornerDetection(const cv::Mat &edgeImage, std::vector<cv::Point2f> &output);
    void calculateOpticalFlow(const cv::Mat &currFrame);
    cv::Vec3f getGyroData();        //get gyro data from the PX4 gyroscope


    //callback functions
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
    void velocityBodyCallback(const geometry_msgs::TwistStamped::ConstPtr& msg);
    void velocityLocalCallback(const geometry_msgs::TwistStamped::ConstPtr& msg);
    void localPositionCallback(const geometry_msgs::PoseStamped::ConstPtr &msg);
    void airspeedCallback(const mavros_msgs::VFR_HUD::ConstPtr &msg); // Change to appropriate message type
    void lidarCallback(const sensor_msgs::Range::ConstPtr &msg);
    void rcCallback(const mavros_msgs::RCIn::ConstPtr &msg);

    //derotation functions
    void applyDerotation3D(OFVectorFrame &ofVector, const cv::Vec3f &avgGyroRadSec);

    //DISTANCE ESTIMATION
    double estimateDepth(OFVectorFrame &ofVector, const cv::Vec3f T_cam);
    double estimateAltitude(OFVectorFrame &ofVector, double depth);
    void processOF();       //this function wraps the depths and altitude computations

    //publish altitude
    void altitudePublisherThread();
    std::thread altitude_thread;
    std::mutex altitude_mutex;

    //debugging
    void sendStatusText(const std::string &message, uint8_t severity);

    //GPS time sincyng
    void syncSystemTimeWithGPS();
    void gpsFixCallback(const sensor_msgs::NavSatFix::ConstPtr &msg);
    void gpsVelCallback(const geometry_msgs::TwistStamped::ConstPtr &msg);

};

#endif // VISION_NODE_FLIR_HPP

