#include <visionNodePlayback.hpp>


int main(int argc, char **argv) {
    ros::init(argc, argv, "vision_node_playback");
    
    VisionNodePlayback visionNodePlayback;
    ROS_INFO("vision_node_playback started.");
    visionNodePlayback.run();
    return 0;
}