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

class OFVectorFrame:
    """
    Python equivalent of your C++ struct OFVectorFrame.
    I campi originali (deltaX, uPixelSec) contengono il flusso RAW.
    Vengono aggiunti campi per il flusso DEROTATED e per l'analisi.
    """
    def __init__(self, p1, p2, fps, camParams):
        # --- CAMPI ORIGINALI (RAW FLOW) ---
        self.position = p1
        self.nextPosition = p2
        self.fps = fps # Nuovo: Memorizza fps per la riconversione
        self.camParams = camParams # Nuovo: Memorizza camParams per la riconversione
        
        # Flusso RAW in pixel/frame
        self.deltaX = p2[0] - p1[0]
        self.deltaY = p2[1] - p1[1]
        
        # Flusso RAW in pixel/sec
        self.uPixelSec = self.deltaX * fps
        self.vPixelSec = self.deltaY * fps
        self.magnitudePixel = math.sqrt(self.deltaX**2 + self.deltaY**2)

        # Flusso RAW in rad/s (normalizzato)
        self.uNormSec_raw = self.uPixelSec / camParams.fx
        self.vNormSec_raw = self.vPixelSec / camParams.fy
        
        # Altri campi originali
        self.uDegSec = (self.uPixelSec / camParams.fx) * (180.0 / np.pi)
        self.vDegSec = (self.vPixelSec / camParams.fy) * (180.0 / np.pi)
        self.AMeter = compute_a_vector_meter(self.position, camParams)
        self.directionVector = compute_direction_vector(self.position, camParams)
        
        # P = Flow Derotato 3D Normalizzato (necessario per _estimateDepth)
        self.P = np.array([0.0, 0.0, 0.0], dtype=np.float32) 

        # --- CAMPI AGGIUNTIVI PER L'ANALISI (DEROTATED FLOW) ---
        self.deltaX_derot = 0.0
        self.deltaY_derot = 0.0
        self.uPixelSec_derot = 0.0
        self.vPixelSec_derot = 0.0
        self.magnitudePixel_derot = 0.0
        
        # Campi per l'analisi dei rapporti
        self.magnitude_raw = 0.0
        self.magnitude_rotational = 0.0
        self.magnitude_derotated = 0.0
# ============================================================
# PID per Adaptive Slicing (versione Python del tuo C++)
# ============================================================

class AdaptiveSlicerPID:
    def __init__(self, cfg: dict, enabled: bool):
        self.enabled = enabled                     # <--- DIPENDE DA SLICING.type!

        slicer_cfg = cfg["SLICING"]

        if "mvsec" in cfg["EVENTS"]["scene"].lower():
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

        elif "fpv" in cfg["EVENTS"]["scene"].lower():
            self.initial_dt_ms = 33.0  # 30 fps initial guess for FPV scenes
        else:
            self.initial_dt_ms = 33.0  # default fallback
            

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



class AdaptiveSlicerABMOF:
    """
    Adaptive slicer inspired by Liu & Delbrück (2018) ABMOF.
    Controls event-density threshold (areaEventThr), NOT time.
    """

    def __init__(self, cfg: dict, enabled: bool):
        self.enabled = enabled

        ad = cfg["adaptiveSlicingABMOF"]

        # Spatial layout
        #from https://github.com/wzygzlm/abmof_libcaer.git
        #the area size = width/8

        self.H = cfg["EVENTS"].get("H", 260)
        self.W = cfg["EVENTS"].get("W", 346)
        self.area_size = self.W // 8

        self.grid_h = self.H // self.area_size
        self.grid_w = self.W // self.area_size

        # Threshold control
        self.areaEventThr = ad.get("initialAreaEventThr", 1000)
        self.minThr = ad.get("minAreaEventThr", 100)
        self.maxThr = ad.get("maxAreaEventThr", 1000)
        self.stepThr_factor = ad.get("AreaEventThr_incrase_factor", 0.05)      # 5% step, as from paper

        # OF histogram
        self.search_radius = ad.get("searchRadius", 3)
        r = self.search_radius
        self.ofHist = np.zeros((2*r+1, 2*r+1), dtype=np.int32)

        # Internal state
        self.slice_rotated = False

        self.reset_area_counters()

    # --------------------------------------------------
    # AREA EVENT SLICING
    # --------------------------------------------------
    def reset_area_counters(self):
        self.areaCounters = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)

    def accumulate_event(self, x, y):
        """
        Called per event.
        Returns True if slice should rotate.
        """
        gx = x // self.area_size
        gy = y // self.area_size

        if gx < 0 or gx >= self.grid_w or gy < 0 or gy >= self.grid_h:
            return False

        self.areaCounters[gy, gx] += 1

        if self.areaCounters[gy, gx] >= self.areaEventThr:
            self.slice_rotated = True
            return True

        return False

    # --------------------------------------------------
    # OPTICAL FLOW FEEDBACK
    # --------------------------------------------------
    def update_with_flow(self, flow_vectors):
        """
        Accumulate OF histogram (called once per slice).
        """
        if not self.enabled or len(flow_vectors) == 0:
            return

        r = self.search_radius

        for fv in flow_vectors:
            dx = int(np.round(fv.deltaX_derot))
            dy = int(np.round(fv.deltaY_derot))

            dx = np.clip(dx, -r, r)
            dy = np.clip(dy, -r, r)

            self.ofHist[dy + r, dx + r] += 1

    def feedback(self):
        """
        Update areaEventThr based on OF histogram.
        """
        if not self.enabled:
            return False

        hist = self.ofHist
        if hist.sum() < 10:
            self._reset_feedback()
            return False

        r = self.search_radius
        ys, xs = np.mgrid[-r:r+1, -r:r+1]

        radius_sq = xs**2 + ys**2

        avgMatchDistance = np.sum(radius_sq * hist) / hist.sum()
        avgTargetDistance = np.mean(radius_sq)

        updated = False

        if avgMatchDistance > avgTargetDistance:
            # Too much motion → slice too long → reduce threshold
            self.areaEventThr -= self.stepThr_factor * self.areaEventThr
            updated = True
        else:
            # Too little motion → slice too short → increase threshold
            self.areaEventThr += self.stepThr_factor * self.areaEventThr
            updated = True

        self.areaEventThr = int(np.clip(
            self.areaEventThr, self.minThr, self.maxThr
        ))

        self._reset_feedback()
        return updated

    def _reset_feedback(self):
        self.ofHist[:] = 0
        self.reset_area_counters()
        self.slice_rotated = False
