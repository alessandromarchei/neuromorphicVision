#include <visionNode.hpp>


int main(int argc, char **argv) {
    ros::init(argc, argv, "vision_node");
    
    VisionNode visionNode;
    ROS_INFO("vision_node started.");
    visionNode.run();
    return 0;
}