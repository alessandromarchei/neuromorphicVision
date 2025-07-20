#include <defs.hpp>
#include <functions.hpp>
#include <cmath>


OFVectorEvents::OFVectorEvents(const cv::Point2f& pos, const cv::Point2f& nextPos, const int fps, const CAMParameters& camParams)
    : position(pos), nextPosition(nextPos) {
    // Calculate pixel displacement
    cv::Vec2f pixelDisplacement = nextPos - pos;
    deltaX = static_cast<int>(pixelDisplacement[0]);
    deltaY = static_cast<int>(pixelDisplacement[1]);

    // Calculate velocity in pixels per second
    uPixelSec = deltaX * fps;
    vPixelSec = deltaY * fps;

    // Calculate magnitude
    magnitudePixel = std::sqrt(deltaX * deltaX + deltaY * deltaY);

    // Calculate 3D vector A in meters
    AMeter = computeAVectorMeter(position, camParams);

    // Calculate the direction vector coordinate with respect to the camera frame
    directionVector = computeDirectionVector(position, camParams);
}

OFVectorFrame::OFVectorFrame(const cv::Point2f& pos, const cv::Point2f& nextPos, const int fps, const CAMFrameParameters& camParams)
    : position(pos), nextPosition(nextPos) {
    // Calculate pixel displacement
    cv::Vec2f pixelDisplacement = nextPos - pos;
    deltaX = static_cast<int>(pixelDisplacement[0]);
    deltaY = static_cast<int>(pixelDisplacement[1]);

    // Calculate velocity in pixels per second
    uPixelSec = deltaX * fps;
    vPixelSec = deltaY * fps;

    //calculate deg per second of vectors
    // Convert pixel velocity to angular velocity (degrees per second)
    uDegSec = (uPixelSec / camParams.fx) * (180.0 / CV_PI);
    vDegSec = (vPixelSec / camParams.fy) * (180.0 / CV_PI);

    // Calculate magnitude
    magnitudePixel = std::sqrt(deltaX * deltaX + deltaY * deltaY);

    // Calculate 3D vector A in meters
    AMeter = computeAVectorMeter(position, camParams);

    // Calculate the direction vector coordinate with respect to the camera frame
    directionVector = computeDirectionVector(position, camParams);
}

