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
struct OFVectorEvents;

inline dv::io::CameraCapture::DVXeFPS intToDVXeFPS(int value) {
    switch(value) {
        case 100: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_100;
        case 200: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_200;
        case 500: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_500;
        case 1000: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_1000;
        case 2001: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_LOSSY_2000;
        case 5001: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_LOSSY_5000;
        case 10001: return dv::io::CameraCapture::DVXeFPS::EFPS_CONSTANT_LOSSY_10000;
        case 2000: return dv::io::CameraCapture::DVXeFPS::EFPS_VARIABLE_2000;
        case 5000: return dv::io::CameraCapture::DVXeFPS::EFPS_VARIABLE_5000;
        case 10000: return dv::io::CameraCapture::DVXeFPS::EFPS_VARIABLE_10000;
        case 15000: return dv::io::CameraCapture::DVXeFPS::EFPS_VARIABLE_15000;
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

inline cv::Vec2f pixelToAngle(const cv::Point2f &pos, const cv::Point2f &nextPos, CAMFrameParameters& camParams, bool horizontal) {
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

inline cv::Vec3f computeAVectorPixel(const cv::Point2f& pos, const CAMFrameParameters& camParams) {
    // Compute A in pixel coordinates
    cv::Vec3f a((pos.x - camParams.cx), (pos.y - camParams.cy), camParams.fx);
    return a;
}

inline cv::Vec3f computeAVectorMeter(const cv::Point2f &pos, const CAMParameters& camParams) {
    // Compute A in meters coordinates
    cv::Vec3f a((pos.x - camParams.cx) * camParams.pixelSize, (pos.y - camParams.cy) * camParams.pixelSize, camParams.fx * camParams.pixelSize);
    return a;
}

inline cv::Vec3f computeAVectorMeter(const cv::Point2f &pos, const CAMFrameParameters& camParams) {
    // Compute A in meters coordinates
    cv::Vec3f a((pos.x - camParams.cx) * camParams.pixelSize, (pos.y - camParams.cy) * camParams.pixelSize, camParams.fx * camParams.pixelSize);
    return a;
}

inline cv::Vec3f computeDirectionVector(const cv::Point2f &pos, const CAMParameters &camParams) {
    // Compute the direction vector in meters coordinates (unit vector)
    cv::Vec3f a = computeAVectorMeter(pos, camParams);
    return a / cv::norm(a);
}

inline cv::Vec3f computeDirectionVector(const cv::Point2f &pos, const CAMFrameParameters &camParams) {
    // Compute the direction vector in meters coordinates (unit vector)
    cv::Vec3f a = computeAVectorMeter(pos, camParams);
    return a / cv::norm(a);
}

inline double calculateMedian(std::vector<double>& data) {
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    return data[data.size() / 2];
}

inline std::vector<OFVectorEvents> rejectOutliers(const std::vector<OFVectorEvents>& flowVectors, double magnitudeThresholdPixel, double boundThreshold = 1.5) {
    std::vector<OFVectorEvents> magnitudeFilteredVectors;
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

    std::vector<OFVectorEvents> outputVectors;
    outputVectors.reserve(magnitudeFilteredVectors.size());
    for (const auto& vec : magnitudeFilteredVectors) {
        if (vec.magnitudePixel >= lower_bound && vec.magnitudePixel <= upper_bound) {
            outputVectors.push_back(vec);
        }
    }

    return outputVectors;
}

inline std::vector<OFVectorFrame> rejectOutliers(const std::vector<OFVectorFrame>& flowVectors, double magnitudeThresholdPixel, double boundThreshold = 1.5) {
    std::vector<OFVectorFrame> magnitudeFilteredVectors;
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

    std::vector<OFVectorFrame> outputVectors;
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

inline cv::Vec3f bodyToCam(const cv::Vec3f& vectorBody, CAMFrameParameters& camParams) {
    // Transform a vector from body frame to camera frame

    //IT ACCEPT A VECTOR IN FRD. SO IF RECEIVED DATA FROM THE PX4, CONVERT IT from FLU -> FRD first
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

inline cv::Vec3f camToBody(const cv::Vec3f& vectorCam, CAMFrameParameters& camParams) {
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
    int64_t timestamp;
    double vx, vy, vz;
    uint64_t deltaTFrame;
    double airspeed, altitude;
    double groundspeed;
    double distance_ground = 0;
    double lidarData;
    double q[4];
    double roll_angle, pitch_angle, yaw_angle;
    double gx,gy,gz;
    double ax,ay,az;
    uint64_t frameID;
};


//IMU data for the PLAYBACK mode
struct IMUData {
    int64_t timestamp;
    double airspeed;
    double groundspeed;
    double lidarData;
    double q[4];
    double gx,gy,gz;
    double ax,ay,az;
    double roll_angle, pitch_angle, yaw_angle;
    double distance_ground;
};

struct VelocityData {
    int64_t timestamp;
    cv::Vec3f vx_frd, vx_flu;
};

struct FrameData {
    int64_t timestamp;
    uint64_t frameID;
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


template <typename sensorType> 
inline sensorType getClosestSensorData(const std::vector<sensorType>& sensorData, int64_t timestamp) {
    auto it = std::lower_bound(sensorData.begin(), sensorData.end(), timestamp, [](const sensorType& data, double ts) {
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

inline int mapRCtoSensitivity(int chanRC)
{
    //this function gets the values and returns the corresponding range

    /*     SENSITIVITY RANGES
    1 : 982 - 1185      -> VERY LOW
    2 : 1186 - 1390     -> LOW
    3 : 1391 - 1594     -> MEDIUM
    4 : 1595 - 1798     -> HIGH
    5 : 1799 - 2006     -> VERY HIGH */

    if (chanRC >= 982 && chanRC <= 1185)
    {
        return 1;
    }
    else if (chanRC >= 1186 && chanRC <= 1390)
    {
        return 2;
    }
    else if (chanRC >= 1391 && chanRC <= 1594)
    {
        return 3;
    }
    else if (chanRC >= 1595 && chanRC <= 1798)
    {
        return 4;
    }
    else if (chanRC >= 1799 && chanRC <= 2006)
    {
        return 5;
    }
    else
    {
        return 1;
    }
}


inline int mapRCtoEFPS(int chanRC)
{
    //this function gets the values and returns the corresponding range

/*     
    EFPS ranges (982 - 2006) divided in 8 ranges (so every step is 128)
    - EFPS_CONSTANT_100 : (982 - 1109)
    - EFPS_CONSTANT_200 : (1110 - 1237)    
    - EFPS_CONSTANT_500 : (1238 - 1365)         
    - EFPS_CONSTANT_1000 : (1366 - 1493)               
    - EFPS_VARIABLE_2000 : (1494 - 1621)       
    - EFPS_VARIABLE_5000 : (1622 - 1749)
    - EFPS_VARIABLE_10000 : (1750 - 1877)
    - EFPS_VARIABLE_15000 : (1878 - 2006) */
    
    if(chanRC >= 982 && chanRC <= 1109)
    {
        return 100;
    }
    else if(chanRC >= 1110 && chanRC <= 1237)
    {
        return 200;
    }
    else if(chanRC >= 1238 && chanRC <= 1365)
    {
        return 500;
    }
    else if(chanRC >= 1366 && chanRC <= 1493)
    {
        return 1000;
    }
    else if(chanRC >= 1494 && chanRC <= 1621)
    {
        return 2000;
    }
    else if(chanRC >= 1622 && chanRC <= 1749)
    {
        return 5000;
    }
    else if(chanRC >= 1750 && chanRC <= 1877)
    {
        return 10000;
    }
    else if(chanRC >= 1878 && chanRC <= 2006)
    {
        return 15000;
    }
    else
    {
        return 100;
    }
}

inline int64_t extractTimestampFromName(const std::string &str)
{
       // Example filename = "rec20210915T153000e1000s3.aedat4"
    // Example AEDAT4 file path = "/home/sharedData/test220824/recordings/rec20240822T101818e5000s1.aedat4"

    // Find the last '/' to ensure we're working with the filename
    size_t lastSlashPos = str.find_last_of("/");

    // Find the 'rec' in the filename part after the last '/'
    size_t recPos = str.find("rec", lastSlashPos);

    if (recPos == std::string::npos) {
        throw std::invalid_argument("The string does not contain 'rec'.");
    }

    // Find the first occurrence of 'e' after 'rec'
    size_t firstEPos = str.find("e", recPos);

    if (firstEPos == std::string::npos) {
        throw std::invalid_argument("The string does not contain 'e' after 'rec'.");
    }

    // Extract the timestamp, which is between 'rec' and just before the first 'e'
    std::string timestampStr = str.substr(recPos + 3, 15);

    std::cout << "Timestamp string: " << timestampStr << std::endl;

    if (timestampStr.size() != 15) {
        throw std::invalid_argument("The input string is not of the expected format.");
    }

    // Parse the string components  
    int year = std::stoi(timestampStr.substr(0, 4));        // Year
    int month = std::stoi(timestampStr.substr(4, 2));       // Month
    int day = std::stoi(timestampStr.substr(6, 2));         // Day
    int hour = std::stoi(timestampStr.substr(9, 2));        // Hour
    int minute = std::stoi(timestampStr.substr(11, 2));     // Minute
    int second = std::stoi(timestampStr.substr(13, 2));     // Second
    int millisecond = 0;                                   // Millisecond
    int microsecond = 0;                                   // Microsecond

    // Set up a struct tm with the parsed values
    std::tm timeStruct = {};
    timeStruct.tm_year = year - 1900;  // tm_year is years since 1900
    timeStruct.tm_mon = month - 1;     // tm_mon is 0-based
    timeStruct.tm_mday = day;
    timeStruct.tm_hour = hour;
    timeStruct.tm_min = minute;
    timeStruct.tm_sec = second;

    // Assume the given time is in GMT+2 (2 hours ahead of GMT)
    // Subtract 2 hours to convert it to GMT
    timeStruct.tm_hour -= 1;

    std::cout << "Unix epoch time: " << std::mktime(&timeStruct) << std::endl;

    // Convert to time_t, which is Unix timestamp
    time_t unixEpochTime = std::mktime(&timeStruct);

    // Check if conversion was successful
    if (unixEpochTime == -1) {
        throw std::runtime_error("Failed to convert time to Unix epoch.");
    }

    return static_cast<int64_t>(unixEpochTime);
    

    return std::stoll(timestampStr);
}

// Main deblurring function that uses the parameters
inline void deblurImage(cv::Mat& image, const DeblurParameters& params) {
    if (!params.general.enable) {
        return ;
    }

    cv::Mat originalImage = image.clone();
    if (params.general.method == "laplacian") {
        cv::Mat laplacian;
        cv::Laplacian(image, laplacian, CV_8U, params.laplacian.kernelSize);
        image = image - (laplacian * params.laplacian.strength);
    }
    else if (params.general.method == "unsharp") {
        cv::Mat blurred;
        cv::Size kernelSize(params.unsharp.kernelSize, params.unsharp.kernelSize);
        cv::GaussianBlur(image, blurred, kernelSize, params.unsharp.sigma);
        image = image * (1 + params.unsharp.strength) - blurred * params.unsharp.strength;
    }
    else if (params.general.method == "bilateral") {
        cv::bilateralFilter(image, image, 
                          params.bilateral.diameter,
                          params.bilateral.sigmaColor,
                          params.bilateral.sigmaSpace,
                          params.bilateral.borderType);
    }
    else if (params.general.method == "richardson_lucy") {
        // Convert input image to float and normalize to range [0, 1]
        image.convertTo(image, CV_32F, 1.0 / 255.0);

        // Create the Gaussian PSF kernel
        int kernelRadius = params.richardsonLucy.kernelSize / 2;
        cv::Mat kernel = cv::getGaussianKernel(params.richardsonLucy.kernelSize, 
                                            params.richardsonLucy.sigma, 
                                            CV_32F);
        cv::Mat psf = kernel * kernel.t(); // Create a 2D Gaussian PSF
        cv::normalize(psf, psf, 0, 1, cv::NORM_MINMAX);

        // Initialize the estimated image with the original image
        cv::Mat estimate = image.clone();
        cv::Mat estimateBlurred, ratio, correction, psfFlipped;

        // Precompute the flipped PSF for convolution
        cv::flip(psf, psfFlipped, -1);

        for (int i = 0; i < params.richardsonLucy.maxIterations; ++i) {
            // Convolve the estimate with PSF
            cv::filter2D(estimate, estimateBlurred, -1, psf);

            // Avoid division by zero by adding a small epsilon
            cv::Mat observedOverEstimate = image / (estimateBlurred + 1e-6);

            // Convolve the ratio with the flipped PSF
            cv::filter2D(observedOverEstimate, correction, -1, psfFlipped);

            // Update the estimate
            estimate = estimate.mul(correction);

            // Optional: Debugging intermediate images
            if (params.general.showDebug && (i % 10 == 0 || i == params.richardsonLucy.maxIterations - 1)) {
                cv::imshow("Intermediate Deblurred Image", estimate);
                cv::waitKey(1);
            }
        }

        // Rescale the final image to [0, 255] and convert back to 8-bit
        cv::normalize(estimate, image, 0, 255, cv::NORM_MINMAX);
        image.convertTo(image, CV_8U);
    }

    if (params.general.showDebug) {
        cv::Mat debug;

        // Concatenate images for debugging
        cv::hconcat(originalImage, image, debug);

        // Show the concatenated images
        cv::imshow("LEFT - Original | RIGHT - Deblurred", debug);
        cv::waitKey(1);
    }

    return ;
}

// Helper function to draw text with a background
inline void drawTextWithBackground(cv::Mat& image, const std::string& text, const cv::Point& org, 
                            int fontFace, double fontScale, const cv::Scalar& textColor, 
                            const cv::Scalar& bgColor, int thickness) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

    // Define the rectangle background
    cv::Point rectTopLeft(org.x - 2, org.y - textSize.height - 2);
    cv::Point rectBottomRight(org.x + textSize.width + 2, org.y + baseline + 2);
    cv::rectangle(image, rectTopLeft, rectBottomRight, bgColor, cv::FILLED);

    // Draw the text on top of the rectangle
    cv::putText(image, text, org, fontFace, fontScale, textColor, thickness);
}


#include "defs.hpp" // Include defs.hpp at the end

#endif // FUNCTIONS_HPP
