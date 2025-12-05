#!/usr/bin/env python3

import math
import numpy as np

# funzioni di utilità per l'OF (uguali al tuo script a frame)
from src.scripts.functions import (
    compute_a_vector_meter,
    compute_direction_vector,
)

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
        self.inclination = 45.0
        self.exposureTime = 20000.0  # microseconds
        self.model = "pinhole" # or "fisheye", in case not rectified

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

# class OFVectorFrame:
#     """
#     Python equivalent of your C++ struct OFVectorFrame.
#     I campi originali (deltaX, uPixelSec) contengono il flusso RAW.
#     Vengono aggiunti campi per il flusso DEROTATED e per l'analisi.
#     """
#     def __init__(self, p1, p2, fps, camParams):
#         # --- CAMPI ORIGINALI (RAW FLOW) ---
#         self.position = p1
#         self.nextPosition = p2
#         self.fps = fps # Nuovo: Memorizza fps per la riconversione
#         self.camParams = camParams # Nuovo: Memorizza camParams per la riconversione
        
#         # Flusso RAW in pixel/frame
#         self.deltaX = p2[0] - p1[0]
#         self.deltaY = p2[1] - p1[1]
        
#         # Flusso RAW in pixel/sec
#         self.uPixelSec = self.deltaX * fps
#         self.vPixelSec = self.deltaY * fps
#         self.magnitudePixel = math.sqrt(self.deltaX**2 + self.deltaY**2)
        
#         # Altri campi originali
#         self.uDegSec = (self.uPixelSec / camParams.fx) * (180.0 / np.pi)
#         self.vDegSec = (self.vPixelSec / camParams.fy) * (180.0 / np.pi)
#         self.AMeter = compute_a_vector_meter(self.position, camParams)
#         self.directionVector = compute_direction_vector(self.position, camParams)
        
#         # P = Flow Derotato 3D Normalizzato (necessario per _estimateDepth)
#         self.P = np.array([0.0, 0.0, 0.0], dtype=np.float32) 

#         # --- CAMPI AGGIUNTIVI PER L'ANALISI (DEROTATED FLOW) ---
#         self.deltaX_derot = 0.0
#         self.deltaY_derot = 0.0
#         self.uPixelSec_derot = 0.0
#         self.vPixelSec_derot = 0.0
#         self.magnitudePixel_derot = 0.0
        
#         # Campi per l'analisi dei rapporti
#         self.magnitude_raw = 0.0
#         self.magnitude_rotational = 0.0
#         self.magnitude_derotated = 0.0

class OFVectorFrame:
    """
    Python equivalent of your C++ struct OFVectorFrame.
    """
    def __init__(self, p1, p2, fps, camParams):
        self.position = p1
        self.nextPosition = p2
        self.deltaX = p2[0] - p1[0]
        self.deltaY = p2[1] - p1[1]
        self.uPixelSec = self.deltaX * fps
        self.vPixelSec = self.deltaY * fps
        self.magnitudePixel = math.sqrt(self.deltaX**2 + self.deltaY**2)

        self.uDegSec = (self.uPixelSec / camParams.fx) * (180.0 / np.pi)
        self.vDegSec = (self.vPixelSec / camParams.fy) * (180.0 / np.pi)
        self.AMeter = compute_a_vector_meter(self.position, camParams)
        self.directionVector = compute_direction_vector(self.position, camParams)

        self.camParams = camParams  # Store camParams for potential reconversion

        # 3d derotated vector in space (not normalized), in m/s. to fill during derotation step
        self.P = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # 🔻 This is the key
    def clone(self):
        new = OFVectorFrame(self.position.copy(), self.nextPosition.copy(), 0.0, self.camParams)
        # overwrite computed features so clone behaves identically
        new.deltaX = self.deltaX
        new.deltaY = self.deltaY
        new.uPixelSec = self.uPixelSec
        new.vPixelSec = self.vPixelSec
        new.magnitudePixel = self.magnitudePixel

        new.uDegSec = self.uDegSec
        new.vDegSec = self.vDegSec

        new.AMeter = self.AMeter.copy()
        new.directionVector = self.directionVector.copy()

        new.P = self.P.copy()
        return new
    
# ============================================================
# PID per Adaptive Slicing (versione Python del tuo C++)
# ============================================================

class AdaptiveSlicerPID:
    def __init__(self, cfg: dict, enabled: bool):
        self.enabled = enabled                     # <--- DIPENDE DA SLICING.type!

        slicer_cfg = cfg["SLICING"]

        self.initial_dt = slicer_cfg.get("gt_mode", "dt1")   # dt iniziale

        if self.initial_dt == "dt1" :
            if "outdoor" in cfg["EVENTS"]["scene"]:
                # outdoor: 21.941 ms → 45 hz
                # indoor: 31.859 ms → ~31 hz     
                self.initial_dt_ms = 22.0
            else:
                self.initial_dt_ms = 32.0
        elif self.initial_dt == "dt4":
            if "outdoor" in cfg["EVENTS"]["scene"]:
                # outdoor: 87.764 ms → ~11.4 hz
                # indoor: 127.436 ms → ~7.85 hz
                self.initial_dt_ms = 88.0
            else:
                self.initial_dt_ms = 127.0

        ad = cfg["adaptiveSlicing"]

        self.P = ad.get("P", 0.5)
        self.I = ad.get("I", 0.05)
        self.D = ad.get("D", 0.0)

        self.maxTimingWindow = ad.get("maxTimingWindow", 25)
        self.minTimingWindow = ad.get("minTimingWindow", 15)
        self.adaptiveTimingWindowStep = ad.get("adaptiveTimingWindowStep", 1)
        self.thresholdPIDEvents = ad.get("thresholdPIDEvents", 10)
        self.OFPixelSetpoint = ad.get("OFPixelSetpoint", 7)

        # stato interno
        self.adaptiveTimeWindow = self.initial_dt_ms
        self.integralError = 0.0
        self.previousError = 0.0
        self.PIDoutput = 0.0

    def get_current_dt_ms(self):
        return self.adaptiveTimeWindow

    def update(self, filteredFlowVectors):
        if not self.enabled:
            return self.adaptiveTimeWindow, False
        if len(filteredFlowVectors) == 0:
            return self.adaptiveTimeWindow, False

        magnitude = sum(
            math.sqrt(v.deltaX**2 + v.deltaY**2) for v in filteredFlowVectors
        ) / len(filteredFlowVectors)

        error = self.OFPixelSetpoint - magnitude
        self.integralError += error
        derivative = error - self.previousError

        self.PIDoutput = (
            self.P * error +
            self.I * self.integralError +
            self.D * derivative
        )

        updateTimingWindow = False

        if abs(self.PIDoutput) > self.thresholdPIDEvents:
            if self.PIDoutput > 0 and self.adaptiveTimeWindow < self.maxTimingWindow:
                self.adaptiveTimeWindow += self.adaptiveTimingWindowStep
                updateTimingWindow = True
            elif self.PIDoutput < 0 and self.adaptiveTimeWindow > self.minTimingWindow:
                self.adaptiveTimeWindow -= self.adaptiveTimingWindowStep
                updateTimingWindow = True

            self.adaptiveTimeWindow = np.clip(
                self.adaptiveTimeWindow,
                self.minTimingWindow,
                self.maxTimingWindow
            )
            self.integralError = 0.0
            self.PIDoutput = 0.0

        self.previousError = error

        if updateTimingWindow:
            print(f"[AdaptiveSlicerPID] Updated timing window to {self.adaptiveTimeWindow} ms")

        return self.adaptiveTimeWindow, updateTimingWindow

