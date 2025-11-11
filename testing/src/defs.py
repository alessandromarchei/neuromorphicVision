#!/usr/bin/env python3

import math
import numpy as np

###############################################################################
# Python translation of structs, constants, or free functions from defs.hpp
###############################################################################

class IMUData:
    """
    Equivalent of your C++ struct IMUData
    """
    def __init__(self):
        self.timestamp = 0          # int64
        self.airspeed = 0.0
        self.groundspeed = 0.0
        self.lidarData = 0.0
        self.q = [0.0, 0.0, 0.0, 0.0]
        self.gx = 0.0
        self.gy = 0.0
        self.gz = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.roll_angle = 0.0
        self.pitch_angle = 0.0
        self.yaw_angle = 0.0


class VelocityData:
    """
    Equivalent of your C++ struct VelocityData
    """
    def __init__(self):
        self.timestamp = 0
        # in C++: cv::Vec3f vx_frd, vx_flu
        # We'll store them as simple 3-element lists or np arrays
        self.vx_frd = [0.0, 0.0, 0.0]
        self.vx_flu = [0.0, 0.0, 0.0]


class FrameData:
    """
    Equivalent of your C++ struct FrameData
    """
    def __init__(self):
        self.timestamp = 0
        self.frameID = 0


class SensorData:
    """
    Equivalent of your C++ struct SensorData
    """
    def __init__(self):
        self.timestamp = 0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.airspeed = 0.0
        self.altitude = 0.0
        self.groundspeed = 0.0
        self.distance_ground = 0.0
        self.lidarData = 0.0
        self.q = [0.0, 0.0, 0.0, 0.0]
        self.roll_angle = 0.0
        self.pitch_angle = 0.0
        self.yaw_angle = 0.0
        self.gx = 0.0
        self.gy = 0.0
        self.gz = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.frameID = 0


###############################################################################
# Camera parameter classes, etc.
###############################################################################
class FastParams:
    def __init__(self):
        self.threshold = 50
        self.nonmaxSuppression = True
        self.randomSampleFilterEnable = False
        self.randomSampleFilterRatio = 0.0
        self.gradientScoringEnable = False
        self.desiredFeatures = 300
        self.safeFeatures = False

class CameraParams:
    def __init__(self):
        self.binningEnable = False
        self.binning_x = 1
        self.binning_y = 1
        self.resolution = (640, 480)
        self.fx = 200.0
        self.fy = 200.0
        self.cx = 320.0
        self.cy = 240.0
        self.pixelSize = 1.0e-5
        self.inclination = 0.0
        self.exposureTime = 20000.0  # microseconds

class LKParams:
    def __init__(self):
        # We’ll set defaults; adapt if needed
        import cv2
        self.winSize = (21, 21)
        self.maxLevel = 3
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
        self.flags = 0
        self.minEigThreshold = 1e-4


###############################################################################
# Any other constants or enumerations from defs.cpp/hpp
###############################################################################

# If there are other global constants, define them here
MAX_FPS_FLIR = 60.0
