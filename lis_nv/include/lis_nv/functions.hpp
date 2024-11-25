#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <ros/ros.h>
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
#include <vector>
#include <algorithm> // For std::sort
#include <iterator>  // For std::begin, std::end>
#include <random>    // For std::mt19937 and std::random_device

// Forward declarations
struct CAMParameters;
struct OFVector;

inline dv::io::CameraCapture::DVXeFPS intToDVXeFPS(int value) {
    switch(value) {
        case 100: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_100;
        case 200: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_200;
        case 500: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_500;
        case 1000: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_1000;
        case 2000: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_LOSSY_2000;
        case 5000: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_LOSSY_5000;
        case 10000: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_LOSSY_10000;
        case 2001: return dv::io::CameraCapture::DVXeFPS::EFPS_VARIABLE_2000;
        case 5001: return dv::io::CameraCapture::DVXeFPS::EFPS_VARIABLE_5000;
        case 10001: return dv::io::CameraCapture::DVXeFPS::EFPS_VARIABLE_10000;
        case 15001: return dv::io::CameraCapture::DVXeFPS::EFPS_VARIABLE_15000;
        default: throw std::out_of_range("Invalid EFPS value");
    }
}

inline dv::io::CameraCapture::BiasSensitivity intToBiasSensitivity(int value) {
    switch(value) {
        case 1: return dv::io::CameraCapture::BiasSensitivity::VeryLow;
        case 2: return dv::io::CameraCapture::BiasSensitivity::Low;
        case 3: return dv::io::CameraCapture::BiasSensitivity::Default;
        case 4: return dv::io::CameraCapture::BiasSensitivity::High;
        case 5: return dv::io::CameraCapture::BiasSensitivity::VeryHigh;
        default: throw std::out_of_range("Invalid Bias Sensitivity value");
    }
}

inline cv::Vec2f pixelToAngle(const cv::Point2f &pos, const cv::Point2f &nextPos, CAMParameters& camParams, bool horizontal) {
    cv::Point2f displacement = nextPos - pos;
    if (horizontal) {
        double normCoord = displacement.x / camParams.fx;
        return cv::Vec2f(std::atan(normCoord) * (180.0 / CV_PI), 0);
    } else {
        double normCoord = displacement.y / camParams.fy;
        return cv::Vec2f(0, std::atan(normCoord) * (180.0 / CV_PI));
    }
}

inline cv::Vec3f computeAVectorPixel(const cv::Point2f& pos, const CAMParameters& camParams) {
    // Compute A in pixel coordinates
    cv::Vec3f a((pos.x - camParams.cx), (pos.y - camParams.cy), camParams.fx);
    return a;
}

inline cv::Vec3f computeAVectorMeter(const cv::Point2f &pos, const CAMParameters& camParams) {
    // Compute A in meters coordinates
    cv::Vec3f a((pos.x - camParams.cx) * camParams.pixelSize, (pos.y - camParams.cy) * camParams.pixelSize, camParams.fx * camParams.pixelSize);
    return a;
}

inline cv::Vec3f computeDirectionVector(const cv::Point2f &pos, const CAMParameters &camParams) {
    // Compute the direction vector in meters coordinates (unit vector)
    cv::Vec3f a = computeAVectorMeter(pos, camParams);
    return a / cv::norm(a);
}

inline double calculateMedian(std::vector<double>& data) {
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return data[data.size() / 2];
}

inline std::vector<OFVector> rejectOutliers(const std::vector<OFVector>& flowVectors, double magnitudeThresholdPixel, double boundThreshold = 1.5) {
    std::vector<OFVector> magnitudeFilteredVectors;
    magnitudeFilteredVectors.reserve(flowVectors.size());
    std::vector<double> magnitudes;

    // Step 1: Magnitude Threshold Filtering
    for (const auto& vec : flowVectors) {        
        if (vec.magnitudePixel <= magnitudeThresholdPixel) {
            magnitudeFilteredVectors.push_back(vec);
            magnitudes.push_back(vec.magnitudePixel);
        }
    }

    //print how many vectors have been filtered


    // If no vectors are left after magnitude filtering, return an empty vector
    if (magnitudeFilteredVectors.empty()) {
        return {};
    }

    // Step 2: Magnitude Variance-Based Filtering

    std::sort(magnitudes.begin(), magnitudes.end());

    // Calculate the IQR (Interquartile Range)
    double q1 = magnitudes[magnitudes.size() / 4];
    double q3 = magnitudes[3 * magnitudes.size() / 4];
    double iqr = q3 - q1;
    double lower_bound = q1 - boundThreshold * iqr;
    double upper_bound = q3 + boundThreshold * iqr;

    std::vector<OFVector> outputVectors;
    outputVectors.reserve(magnitudeFilteredVectors.size());
    for (const auto& vec : magnitudeFilteredVectors) {
        if (vec.magnitudePixel >= lower_bound && vec.magnitudePixel <= upper_bound) {
            outputVectors.push_back(vec);
        }
    }

    return outputVectors;
}

inline cv::Vec3f bodyToCam(const cv::Vec3f& vectorBody, CAMParameters& camParams) {
    // Transform a vector from body frame to camera frame
    cv::Vec3f vectorCam;
    vectorCam[0] = vectorBody[1];
    vectorCam[1] = -sin(camParams.inclination * CV_PI / 180.0) * vectorBody[0] + cos(camParams.inclination * CV_PI / 180.0) * vectorBody[2];
    vectorCam[2] = cos(camParams.inclination * CV_PI / 180.0) * vectorBody[0] + sin(camParams.inclination * CV_PI / 180.0) * vectorBody[2];

    return vectorCam;
}

inline cv::Vec3f camToBody(const cv::Vec3f& vectorCam, CAMParameters& camParams) {
    // Transform a vector from camera frame to body frame
    cv::Vec3f vectorBody;
    vectorBody[0] = -sin(camParams.inclination * CV_PI / 180.0) * vectorCam[1] + cos(camParams.inclination * CV_PI / 180.0) * vectorCam[2];
    vectorBody[1] = vectorCam[0];
    vectorBody[2] = cos(camParams.inclination * CV_PI / 180.0) * vectorCam[1] + sin(camParams.inclination * CV_PI / 180.0) * vectorCam[2];

    return vectorBody;
}

inline cv::Vec3f bodyToInertial(const cv::Vec3f& vectorBody, const double cosRoll, const double sinRoll, const double cosPitch, const double sinPitch) {
    // Extract body frame coordinates
    double x_body = vectorBody[0];
    double y_body = vectorBody[1];
    double z_body = vectorBody[2];

    // Transform coordinates from body to inertial frame
    cv::Vec3f vectorInertial;
    vectorInertial[0] = cosPitch * x_body + sinPitch * sinRoll * y_body + sinPitch * cosRoll * z_body;
    vectorInertial[1] = cosRoll * y_body - sinRoll * z_body;
    vectorInertial[2] = -sinPitch * x_body + cosPitch * sinRoll * y_body + cosPitch * cosRoll * z_body;

    return vectorInertial;
}

inline std::vector<cv::KeyPoint> randomlySampleKeypoints(const std::vector<cv::KeyPoint>& keypoints, int desiredFeatures, double randomSampleRatio)
{
    int total_features = keypoints.size();
    int num_to_keep = static_cast<int>(desiredFeatures + (desiredFeatures * randomSampleRatio));

    // Ensure num_to_keep does not exceed the number of keypoints
    if (num_to_keep > total_features) {
        num_to_keep = total_features;
    }

    std::vector<size_t> indices(total_features);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    // Select the first num_to_keep elements
    std::vector<cv::KeyPoint> sampled_keypoints;
    sampled_keypoints.reserve(num_to_keep);

    for (int i = 0; i < num_to_keep; ++i) {
        sampled_keypoints.push_back(keypoints[indices[i]]);
    }

    return sampled_keypoints;
}

inline std::vector<cv::KeyPoint> scoreAndRankKeypointsUsingGradient(const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& currEdgeImage, const int desiredFeatures)
{
    cv::Mat grad_x, grad_y;
    cv::Sobel(currEdgeImage, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(currEdgeImage, grad_y, CV_32F, 0, 1, 3);

    std::vector<cv::KeyPoint> ranked_keypoints = keypoints;

    int rows = currEdgeImage.rows;
    int cols = currEdgeImage.cols;

    // Assign gradient magnitude scores to keypoints
    for (auto& kp : ranked_keypoints) {
        int x = cvRound(kp.pt.x);
        int y = cvRound(kp.pt.y);

        // Check bounds before accessing
        if (x >= 0 && x < cols && y >= 0 && y < rows) {
            float score = std::sqrt(grad_x.at<float>(y, x) * grad_x.at<float>(y, x) + grad_y.at<float>(y, x) * grad_y.at<float>(y, x));
            kp.response = score;
        } else {
            kp.response = 0;  // Assign a default score if out of bounds
        }
    }

    // Sort keypoints based on their gradient magnitude scores
    std::sort(ranked_keypoints.begin(), ranked_keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });

    // Select the top desired features
    if (ranked_keypoints.size() > desiredFeatures) {
        ranked_keypoints.resize(desiredFeatures);
    }

    return ranked_keypoints;
}



//sensor data for the PLAYBACK mode
struct SensorData {
    double timestamp;
    double vx, vy, vz;
    double airspeed, altitude;
    double distance_ground;
    double q[4];
    double roll_angle, pitch_angle, yaw_angle;
};


inline std::vector<SensorData> parseSensorData(const std::string& filename) {
    std::vector<SensorData> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        //print ros error
        ROS_ERROR("Error opening file %s", filename.c_str());

        //EXIT THE ROS NODE
        std::raise(SIGINT);

        return data;
    }

    std::string line;
    std::getline(file, line); // Skip header line
    ROS_INFO("Header: %s", line.c_str());

    int lineCount = 0;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        SensorData entry;

        try {
            std::getline(ss, token, ',');
            entry.timestamp = std::stod(token);

            std::getline(ss, token, ',');
            entry.vx = std::stod(token);

            std::getline(ss, token, ',');
            entry.vy = std::stod(token);

            std::getline(ss, token, ',');
            entry.vz = std::stod(token);

            std::getline(ss, token, ',');
            entry.airspeed = std::stod(token);

            std::getline(ss, token, ',');
            entry.altitude = std::stod(token);

            std::getline(ss, token, ',');
            entry.distance_ground = std::stod(token);

            std::getline(ss, token, ',');
            entry.roll_angle = std::stod(token);

            std::getline(ss, token, ',');
            entry.pitch_angle = std::stod(token);

            std::getline(ss, token, ',');
            entry.yaw_angle = std::stod(token);
            

            data.push_back(entry);
            lineCount++;
            if (lineCount % 1000 == 0) {
                ROS_INFO("Processed %d lines", lineCount);
            }
        } catch (const std::exception &e) {
            //print ros error
            ROS_ERROR("Error parsing line %d: %s", lineCount, e.what());

            //EXIT THE ROS NODE
            std::raise(SIGINT);
        }
    }

    file.close();
    return data;
}



inline SensorData getClosestSensorData(const std::vector<SensorData>& sensorData, int64_t timestamp) {
    auto it = std::lower_bound(sensorData.begin(), sensorData.end(), timestamp, [](const SensorData& data, double ts) {
        return data.timestamp < ts;
    });

    if (it == sensorData.end()) {
        return sensorData.back();
    }
    if (it == sensorData.begin()) {
        return sensorData.front();
    }

    auto prev_it = std::prev(it);
    if ((it->timestamp - timestamp) < (timestamp - prev_it->timestamp)) {
        return *it;
    } else {
        return *prev_it;
    }   
}


//SMOOTHING FILTER

//LPF
inline float LPFilter(float prevValue, float newValue, float lpfK) {
    return lpfK * newValue + (1 - lpfK) * prevValue;
}

//Complementary Filter
inline float complementaryFilter(float current_measurement, float prev_filtered_value, float complementaryK, float deltaT) {
    float vk = (current_measurement - prev_filtered_value) * complementaryK;
    float filtered_value = vk * deltaT + prev_filtered_value;
    return filtered_value;
}

#include "defs.hpp" // Include defs.hpp at the end

#endif // FUNCTIONS_HPP
