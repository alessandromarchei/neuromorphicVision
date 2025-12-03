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


def bodyToInertial(vectorBody, roll_angle_rad, pitch_angle_rad):
    """
    Made for the UZH-FPV dataset
    """
    x_b = vectorBody[0]
    y_b = vectorBody[1]
    z_b = vectorBody[2]

    cosPitch = math.cos(pitch_angle_rad)
    sinPitch = math.sin(pitch_angle_rad)
    cosRoll  = math.cos(roll_angle_rad)
    sinRoll  = math.sin(roll_angle_rad)
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

def quat_to_euler(qx, qy, qz, qw, order="ZYX"):
    """
    Convert quaternion (qx, qy, qz, qw) to Euler angles in radians.

    Default order="ZYX" = yaw(Z) -> pitch(Y) -> roll(X)
    which is the standard aerospace/world convention.

    Returns (roll, pitch, yaw).
    """

    # Normalise quaternion
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    if order.upper() == "ZYX":
        # --- yaw (Z) ---
        yaw = np.arctan2(
            2*(qw*qz + qx*qy),
            1 - 2*(qy*qy + qz*qz)
        )

        # --- pitch (Y) ---
        s = 2*(qw*qy - qz*qx)
        s = np.clip(s, -1.0, 1.0)
        pitch = np.arcsin(s)

        # --- roll (X) ---
        roll = np.arctan2(
            2*(qw*qx + qy*qz),
            1 - 2*(qx*qx + qy*qy)
        )

        return roll, pitch, yaw

    else:
        raise NotImplementedError(f"Euler order '{order}' not supported yet.")



def compute_attitude_trig(gt_cam_state_slice, initial_roll_deg=0.0, initial_pitch_deg=0.0):
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

    roll_mean = np.mean(rolls) - np.radians(initial_roll_deg)
    pitch_mean = np.mean(pitches) - np.radians(initial_pitch_deg)

    return roll_mean, pitch_mean

def quat_to_rotmat(qx, qy, qz, qw):
    """
    Quaternion -> rotation matrix (body->world).
    """
    x, y, z, w = qx, qy, qz, qw
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),   1-2*(x*x+z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),   2*(y*z + w*x),   1-2*(x*x+y*y)]
    ])
    return R


def rotmat_to_euler(R):
    """
    Rotation matrix -> Euler (roll, pitch, yaw), ZYX convention.
    Returns radians.
    """
    sy = -R[2,0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    roll  = np.arctan2(R[2,1], R[2,2])
    yaw   = np.arctan2(R[1,0], R[0,0])
    return roll, pitch, yaw


def compute_initial_attitude_offset(events_dir, n_samples=10):
    """
    1) cerca groundtruth.txt
    2) legge i primi n_samples
    3) calcola roll/pitch medi
    """
    # ------------- Locate file -------------
    gt_file = None
    for f in os.listdir(events_dir):
        if f.lower().endswith("groundtruth.txt"):
            gt_file = os.path.join(events_dir, f)
            break

    if gt_file is None:
        raise FileNotFoundError(f"Groundtruth file not found under {events_dir}")

    print(f"[INFO] Reading GT file: {gt_file}")

    # ------------- Load GT -------------
    data = np.loadtxt(gt_file, comments="#")

    # ensure enough samples exist
    n = min(n_samples, data.shape[0])

    # extract quaternions (order qx,qy,qz,qw)
    quat = data[:n, 4:8]

    rolls = []
    pitches = []

    for i in range(n):
        qx, qy, qz, qw = quat[i]
        R = quat_to_rotmat(qx, qy, qz, qw)
        roll, pitch, _ = rotmat_to_euler(R)
        rolls.append(roll)
        pitches.append(pitch)

    roll_mean_deg  = np.degrees(np.mean(rolls))
    pitch_mean_deg = np.degrees(np.mean(pitches))

    print(f"\n===== INITIAL BODY ATTITUDE OFFSET =====")
    print(f"Samples evaluated: {n}")
    print(f"Initial roll  ≈ {roll_mean_deg:.2f} deg")
    print(f"Initial pitch ≈ {pitch_mean_deg:.2f} deg")
    print("========================================\n")

    return roll_mean_deg, pitch_mean_deg



def rotmat_to_quat(R):
    """
    Convert a 3x3 rotation matrix into a quaternion (qx, qy, qz, qw).
    Quaternion follows convention: q = [qx, qy, qz, qw]
    """

    # Ensure input is array
    R = np.asarray(R, dtype=float)
    
    trace = R[0,0] + R[1,1] + R[2,2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    else:
        # Find largest diagonal element
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s

    # Normalize quaternion
    q = np.array([qx, qy, qz, qw])
    q /= np.linalg.norm(q)

    return q[0], q[1], q[2], q[3]
