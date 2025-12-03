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
    """takes into consideration camera model (pinhole or fisheye)"""
    u, v = pos

    if cam_params.model.lower() == "pinhole":
        # keep pinhole projection
        x = (u - cam_params.cx) * cam_params.pixelSize
        y = (v - cam_params.cy) * cam_params.pixelSize
        z = cam_params.fx * cam_params.pixelSize
        return np.array([x, y, z], dtype=np.float32)

    elif cam_params.model.lower() == "fisheye":
        # Return meter-scaled fisheye ray
        # Using the same mapping as direction vector ensures consistent geometry
        x = (u - cam_params.cx) / cam_params.fx
        y = (v - cam_params.cy) / cam_params.fy

        r = math.sqrt(x*x + y*y)
        if r < 1e-9:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

        theta = r
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        return np.array([
            (x/r)*sin_theta,
            (y/r)*sin_theta,
            cos_theta
        ], dtype=np.float32)

    else:
        raise ValueError(f"Unknown camera model: {cam_params.model}")


def compute_direction_vector(pos, cam_params):
    """
    Returns bearing ray for pinhole OR fisheye (equidistant) model.
    """
    u, v = pos

    if cam_params.model.lower() == "pinhole":
        # --- Existing behavior (keep as-is) ---
        a = compute_a_vector_meter(pos, cam_params)  # pinhole version
        norm_a = np.linalg.norm(a)
        return a / norm_a if norm_a > 1e-9 else np.array([0.0, 0.0, 0.0], dtype=np.float32)

    elif cam_params.model.lower() == "fisheye":
        # --- Equidistant fisheye model ---
        x = (u - cam_params.cx) / cam_params.fx
        y = (v - cam_params.cy) / cam_params.fy

        r = math.sqrt(x*x + y*y)
        if r < 1e-9:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

        theta = r
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        return np.array([
            (x/r) * sin_theta,
            (y/r) * sin_theta,
            cos_theta
        ], dtype=np.float32)

    else:
        raise ValueError(f"Unknown camera model: {cam_params.model}")


import numpy as np

def quat_to_euler(qx, qy, qz, qw):
    """
    Convert quaternion (qx, qy, qz, qw) to Euler angles (roll, pitch, yaw)
    following ZYX aerospace convention:
        roll  about X
        pitch about Y
        yaw   about Z
    Returns angles in radians.
    """

    # Normalise quaternion to avoid drift
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    # Rotation matrix elements
    # Based on standard SO(3) quaternion formulation
    R11 = 1 - 2*(qy*qy + qz*qz)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)
    R23 = 2*(qy*qz - qx*qw)
    R33 = 1 - 2*(qx*qx + qy*qy)

    # === Extract Euler ===
    # roll (X)
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))

    # pitch (Y)
    # Clamp input for numerical safety
    sinp = 2*(qw*qy - qz*qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (Z)
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    return roll, pitch, yaw



def quat_to_rotmat(qx, qy, qz, qw):
    """
    Quaternion (qx,qy,qz,qw) -> rotation matrix R_WC (camera -> world).
    """
    # normalizza per sicurezza
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    # convenzione w,x,y,z
    w, x, y, z = qw, qx, qy, qz

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [    2*(x*y + z*w),  1 - 2*(x*x + z*z),        2*(y*z - x*w)],
        [    2*(x*z - y*w),      2*(y*z + x*w),    1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R


def quat_to_euler(qx, qy, qz, qw):
    """
    quaternion → (roll, pitch, yaw) in radians

    Convention:
      roll  : rotation around X
      pitch : rotation around Y
      yaw   : rotation around Z
    """

    # normalize quaternion
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    # roll (x-axis)
    sinr = 2.0 * (qw*qx + qy*qz)
    cosr = 1.0 - 2.0*(qx*qx + qy*qy)
    roll = np.arctan2(sinr, cosr)

    # pitch (y-axis)
    sinp = 2.0*(qw*qy - qz*qx)
    if abs(sinp) >= 1:
        pitch = np.pi/2 * np.sign(sinp)    # numerical clamp
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis)
    siny = 2.0*(qw*qz + qx*qy)
    cosy = 1.0 - 2.0*(qy*qy + qz*qz)
    yaw = np.arctan2(siny, cosy)

    return roll, pitch, yaw


def compute_attitude_trig(gt_cam_state_slice):
    """
    Compute cosRoll, sinRoll, cosPitch, sinPitch
    from quaternion orientation in GT cam slice.
    """

    if len(gt_cam_state_slice) == 0:
        return 1.0, 0.0, 1.0, 0.0  # identity

    qx = np.asarray(gt_cam_state_slice["qx"], dtype=np.float64)
    qy = np.asarray(gt_cam_state_slice["qy"], dtype=np.float64)
    qz = np.asarray(gt_cam_state_slice["qz"], dtype=np.float64)
    qw = np.asarray(gt_cam_state_slice["qw"], dtype=np.float64)

    N = qx.shape[0]
    rolls = np.zeros(N)
    pitches = np.zeros(N)

    for i in range(N):
        roll, pitch, _ = quat_to_euler(qx[i], qy[i], qz[i], qw[i])
        rolls[i] = roll
        pitches[i] = pitch

    roll_mean = np.mean(rolls)
    pitch_mean = np.mean(pitches)

    cosRoll = np.cos(roll_mean)
    sinRoll = np.sin(roll_mean)
    cosPitch = np.cos(pitch_mean)
    sinPitch = np.sin(pitch_mean)

    return cosRoll, sinRoll, cosPitch, sinPitch


def rotmat_to_euler(R):
    """
    Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw)
    using aerospace ZYX convention:
       roll  about X
       pitch about Y
       yaw   about Z

    Returns (roll, pitch, yaw) in radians.
    """

    # Safety clip for asin domain errors due to noise
    sy = -R[2,0]  # sin(pitch)

    if sy <= -1:
        pitch = np.pi/2
    elif sy >= 1:
        pitch = -np.pi/2
    else:
        pitch = np.arcsin(sy)

    # roll
    roll = np.arctan2(R[2,1], R[2,2])

    # yaw
    yaw  = np.arctan2(R[1,0], R[0,0])

    return roll, pitch, yaw
