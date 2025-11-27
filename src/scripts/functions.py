#!/usr/bin/env python3

import math
import os
import numpy as np
import cv2
import random
from typing import List

###############################################################################
# Python translation of inline or utility functions from functions.hpp/cpp
###############################################################################

def randomlySampleKeypoints(keypoints, desiredFeatures, randomSampleRatio):
    """
    Equivalent of randomlySampleKeypoints(...) from your C++ code.
    """
    total_features = len(keypoints)
    # desiredFeatures + ratio * desiredFeatures
    num_to_keep = int(desiredFeatures + desiredFeatures * randomSampleRatio)
    if num_to_keep > total_features:
        num_to_keep = total_features

    indices = list(range(total_features))
    random.shuffle(indices)
    selected_indices = indices[:num_to_keep]
    return [keypoints[i] for i in selected_indices]


def scoreAndRankKeypointsUsingGradient(keypoints, currEdgeImage, desiredFeatures):
    """
    Equivalent of your 'scoreAndRankKeypointsUsingGradient' logic using Sobel.
    """
    grad_x = cv2.Sobel(currEdgeImage, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(currEdgeImage, cv2.CV_32F, 0, 1, ksize=3)

    rows, cols = currEdgeImage.shape[:2]
    ranked_keypoints = list(keypoints)

    for kp in ranked_keypoints:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        if (0 <= x < cols) and (0 <= y < rows):
            gx = grad_x[y, x]
            gy = grad_y[y, x]
            score = math.sqrt(gx*gx + gy*gy)
            kp.response = score
        else:
            kp.response = 0.0

    ranked_keypoints.sort(key=lambda k: k.response, reverse=True)
    if len(ranked_keypoints) > desiredFeatures:
        ranked_keypoints = ranked_keypoints[:desiredFeatures]
    return ranked_keypoints


def rejectOutliersFrame(flowVectors, magnitudeThresholdPixel, boundThreshold=1.5):
    """
    Equivalent of rejectOutliers(...) for OFVectorFrame in C++.
    """
    magFiltered = []
    mags = []
    for fv in flowVectors:
        mp = fv.magnitudePixel  # must be precomputed
        if mp <= magnitudeThresholdPixel:
            magFiltered.append(fv)
            mags.append(mp)

    if not magFiltered:
        return []

    mags.sort()
    q1 = mags[len(mags)//4]
    q3 = mags[(3*len(mags))//4]
    iqr = q3 - q1
    lower_bound = q1 - boundThreshold * iqr
    upper_bound = q3 + boundThreshold * iqr

    result = []
    for fv, mp in zip(magFiltered, mags):
        if (mp >= lower_bound) and (mp <= upper_bound):
            result.append(fv)
    return result


def drawTextWithBackground(image, text, org,
                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.5, textColor=(255,255,255),
                           bgColor=(0,0,255), thickness=1):
    """
    Equivalent of drawTextWithBackground(...) from your code.
    """
    textSize, baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    x, y = org
    pt1 = (x - 2, y - textSize[1] - 2)
    pt2 = (x + textSize[0] + 2, y + baseline + 2)
    cv2.rectangle(image, pt1, pt2, bgColor, cv2.FILLED)
    cv2.putText(image, text, (x, y), fontFace, fontScale, textColor, thickness)


###############################################################################
# Transforms: bodyToCam, camToBody, bodyToInertial, etc.
###############################################################################

def bodyToCam(vectorBody, camParams):
    """
    Convert from FRD to camera, or from body to camera. 
    Example logic from your c++ bodyToCam(...) inline functions.
    vectorBody is a 3-element array or np.array
    """
    incRad = camParams.inclination * math.pi / 180.0
    # If your code is x_cam = y_body, y_cam = -sin(inclination)*x_body + cos(inclination)*z_body, z_cam= ...
    x_b = vectorBody[0]
    y_b = vectorBody[1]
    z_b = vectorBody[2]

    x_cam = y_b
    y_cam = -math.sin(incRad)*x_b + math.cos(incRad)*z_b
    z_cam = math.cos(incRad)*x_b + math.sin(incRad)*z_b
    return np.array([x_cam, y_cam, z_cam], dtype=np.float32)


def camToBody(vectorCam, camParams):
    """
    Inverse of bodyToCam. 
    """
    incRad = camParams.inclination * math.pi / 180.0
    x_c = vectorCam[0]
    y_c = vectorCam[1]
    z_c = vectorCam[2]

    x_b = -math.sin(incRad)*y_c + math.cos(incRad)*z_c
    y_b = x_c
    z_b = math.cos(incRad)*y_c + math.sin(incRad)*z_c
    return np.array([x_b, y_b, z_b], dtype=np.float32)


def bodyToInertial(vectorBody, cosRoll, sinRoll, cosPitch, sinPitch):
    """
    Equivalent of your inline bodyToInertial(...).
    """
    x_b = vectorBody[0]
    y_b = vectorBody[1]
    z_b = vectorBody[2]

    # from your snippet:
    # inertial[0] = cosPitch*x_b + sinPitch*sinRoll*y_b + sinPitch*cosRoll*z_b
    # inertial[1] = cosRoll*y_b - sinRoll*z_b
    # inertial[2] = -sinPitch*x_b + cosPitch*sinRoll*y_b + cosPitch*cosRoll*z_b
    ix = cosPitch*x_b + sinPitch*sinRoll*y_b + sinPitch*math.cos(math.acos(cosRoll))*z_b
    # The above is a bit suspect if we do `math.acos(cosRoll)`. 
    # Let's do a direct approach from the original formula:
    #   inertial[0] = cosPitch*x_b + sinPitch*sinRoll*y_b + sinPitch*cosRoll*z_b
    #   inertial[1] = cosRoll*y_b - sinRoll*z_b
    #   inertial[2] = -sinPitch*x_b + cosPitch*sinRoll*y_b + cosPitch*cosRoll*z_b

    # We'll just do direct if known:
    ix = cosPitch*x_b + sinPitch*sinRoll*y_b + sinPitch*cosRoll*z_b
    iy = cosRoll*y_b - sinRoll*z_b
    iz = -sinPitch*x_b + cosPitch*sinRoll*y_b + cosPitch*cosRoll*z_b

    return np.array([ix, iy, iz], dtype=np.float32)


###############################################################################
# Smoothing filters: LPFilter, complementaryFilter
###############################################################################

def LPFilter(newValue, prevValue, lpfK):
    """
    out = lpfK*newValue + (1-lpfK)*prevValue
    """
    return lpfK*newValue + (1.0 - lpfK)*prevValue


def complementaryFilter(current_measurement, prev_filtered_value, complementaryK, deltaT):
    """
    In your code, you do something like:
        vk = (current_measurement - prev_filtered_value)*complementaryK
        filtered_value = prev_filtered_value + vk*deltaT
    """
    vk = (current_measurement - prev_filtered_value)*complementaryK
    filtered_value = prev_filtered_value + vk*deltaT
    return filtered_value


def pixel_to_angle(pos, next_pos, cam_params, horizontal=True):
    """
    Equivalent to:
      cv::Vec2f pixelToAngle(const cv::Point2f &pos, const cv::Point2f &nextPos,
                             CAMParameters or CAMFrameParameters, bool horizontal)
    """
    displacement = (next_pos[0] - pos[0], next_pos[1] - pos[1])

    if horizontal:
        normCoord = displacement[0] / cam_params.fx
        angle_x = math.atan(normCoord) * (180.0 / math.pi)
        return (angle_x, 0.0)
    else:
        normCoord = displacement[1] / cam_params.fy
        angle_y = math.atan(normCoord) * (180.0 / math.pi)
        return (0.0, angle_y)

def compute_a_vector_pixel(pos, cam_params):
    """
    Equivalent to:
      cv::Vec3f computeAVectorPixel(const cv::Point2f& pos, const CAMParameters/FrameParams&)
    Returns a 3D vector in "pixel coordinates."
    """
    x = pos[0] - cam_params.cx
    y = pos[1] - cam_params.cy
    z = cam_params.fx  # same 'fx' as in the original
    return np.array([x, y, z], dtype=float)

def compute_a_vector_meter(pos, cam_params):
    """
    Equivalent to:
      cv::Vec3f computeAVectorMeter(const cv::Point2f &pos, const CAMParameters/FrameParams&)
    Returns a 3D vector in "meter coordinates."
    """
    x = (pos[0] - cam_params.cx) * cam_params.pixelSize
    y = (pos[1] - cam_params.cy) * cam_params.pixelSize
    z = cam_params.fx * cam_params.pixelSize
    return np.array([x, y, z], dtype=float)

def compute_direction_vector(pos, cam_params):
    """
    Equivalent to:
      cv::Vec3f computeDirectionVector(const cv::Point2f &pos, const CAMParameters/FrameParams&)
    Returns the unit direction vector (in meter coordinates) from the camera center through 'pos'.
    """
    a = compute_a_vector_meter(pos, cam_params)
    norm_a = np.linalg.norm(a)
    if norm_a == 0:
        # Degenerate case - no meaningful direction if the vector is zero-length
        return np.array([0.0, 0.0, 0.0], dtype=float)
    return a / norm_a

