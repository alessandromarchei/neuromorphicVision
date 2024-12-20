#pragma once

#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/stereo_camera_writer.hpp>
#include <dv-processing/io/stereo_capture.hpp>
#include <dv-processing/io/camera_capture.hpp>
#include <map>
#include <csignal>
#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <math.h>

// OPTICAL FLOW
struct LKParams {
    cv::Size winSize = cv::Size(21, 21);
    int maxLevel = 3;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    int flags = 0;
    double minEigThreshold = 0.001;
    bool CPU = false;
};

struct AccumulatorParameters {
    float neutralPotential;
    float eventContribution;
    float decay;
    bool ignorePolarity;
    cv::Size resolution;
};

// FAST DETECTOR PARAMETERS
struct FASTParameters {
    int threshold;
    int nonmaxSuppression;
    bool randomSampleFilterEnable;
    double randomSampleFilterRatio;
    bool gradientScoringEnable;
    int desiredFeatures; // number of desired features at the output of the gradient scoring filter
    bool safeFeatures = false;
};

struct CAMParameters {
    cv::Size resolution;
    double hfov, vfov;
    int efps;
    int sensitivity;
    double fx, fy, cx, cy; // camera intrinsic parameters
    double gyroFreq;
    double pixelSize;
    double inclination;
};

struct BackgroundActivityNoise_params {
    bool enable = false;
    int timeResolution = 10; // in ms
};

// filters
struct Filter {
    cv::Size resolution;
    enum FilterType { BackGroundActivityNoise, RefractoryPeriod, Polarity, None } type = None;
    BackgroundActivityNoise_params BGA_params;
};

struct OFVector {
    cv::Point2f position;      // Original point position (x, y)
    cv::Point2f nextPosition;  // Next point position (x, y)
    double magnitudePixel;     // Magnitude of the optical flow vector
    int deltaX;                // Horizontal flow vector component in pixels
    int deltaY;                // Vertical flow vector component in pixels
    double uPixelSec;
    double vPixelSec;
    cv::Vec3f AMeter;          // 3D vector A in meters. It is the vector pointing from the focal center into the frame (camera frame)
    cv::Vec3f directionVector; // Direction vector of the point into the frame
    cv::Vec3f P;               // 3D vector P in meters/second

    OFVector(const cv::Point2f& pos, const cv::Point2f& nextPos, const int fps, const CAMParameters& camParams);
};

// Ensure OFVector constructor implementation is in a .cpp file
