#include <visionNodeFrames.hpp>


int main(int argc, char **argv) {
    ros::init(argc, argv, "vision_node_frame");
    
    VisionNodeFrames visionNodeFrames;
    ROS_INFO("vision_node for the conventional camera started.");
    visionNodeFrames.run();
    return 0;
}