#ifndef CONTROLLER_NODE_HPP
#define CONTROLLER_NODE_HPP

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/CommandBool.h>
#include <lis_nv/SparseOFVectorArray.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <tf/tf.h>
#include <defs.hpp>
#include <functions.hpp>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/VFR_HUD.h>


// Function declarations

// Node class
class ControllerNode {
public:
    ControllerNode();
    void spin();

private:
    ros::NodeHandle nh;
    ros::Subscriber imu_sub;
    ros::Subscriber velocity_sub;
    ros::Subscriber optical_flow_sub;
    ros::Publisher attitude_pub;
    ros::Subscriber airspeed_sub;
    ros::Publisher altitude_ground_pub;

    CAMParameters camParams;
    int fps;
    int64_t dtMicroseconds;

    cv::Vec3d T;    //translation vector got by the GPS, IMU or other sensor and sensor fusion
    


    lis_nv::SparseOFVectorArray flowVectors;
    cv::Vec3d avgGyro;

    void loadParameters();


    
    //initialization functions
    

    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
    void velocityCallback(const geometry_msgs::TwistStamped::ConstPtr& msg);
    void opticalFlowCallback(const lis_nv::SparseOFVectorArray::ConstPtr& msg);
    void airspeedCallback(const mavros_msgs::VFR_HUD::ConstPtr& msg); // Change to appropriate message type

    void processOpticalFlow();

    //function to compute the distance for each vector
    //double computeDistance(const lis_nv::SparseOFVector& ofVector, const cv::Vec3d& translation);
    void publishAltitude(double desired_altitude);
    // double computeDistance(const cv::Vec3f& direction, const cv::Vec3f& translation, const lis_nv::SparseOFVector& ofVector);
};

#endif // CONTROLLER_NODE_HPP
