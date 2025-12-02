import os
import time
import math
import glob
import h5py
import yaml
import numpy as np
import cv2

import argparse
import matplotlib.pyplot as plt


# ============================================================
# IMPORT: adatta questi import ai tuoi file reali
# ============================================================

# iterator + util già esistenti (tuoi)
from testing.utils.load_utils_dvxplorer import dvxplorer_evs_iterator
from testing.utils.viz_utils import visualize_image, visualize_filtered_flow


# strutture dati e parametri (come nel tuo script a frame)
from src.scripts.defs import (
    FastParams,
    CameraParams,
    LKParams,
    AdaptiveSlicerPID,
    OFVectorFrame
)

from src.scripts.functions import (
    randomlySampleKeypoints,
    scoreAndRankKeypointsUsingGradient,
    rejectOutliersFrame,
    bodyToCam,
    camToBody,
    bodyToInertial,
    LPFilter,
    complementaryFilter,
    drawTextWithBackground,
    compute_a_vector_pixel,
    pixel_to_angle,
    compute_a_vector_meter,
    compute_direction_vector
)

# ============================================================
# VisionNodeEventsPlayback
# ============================================================

class VisionNodeDVXEventsPlayback:
    """
    Variante event-based:
    - usa eventi dvxplorer camera dataset
    - genera event frames
    - calcola FAST + LK + filtri
    - opzionalmente usa adaptive slicing PID sui dt_ms
    """

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path

        # parametri di alto livello
        self.fastParams = FastParams()
        self.camParams = CameraParams()
        self.lkParams = LKParams()

        self.visualizeImage = True
        self.delayVisualize = 1  # ms

        self.currFrame = None
        self.prevFrame = None
        self.prevPoints = []
        self.nextPrevPoints = []

        self.flowVectors = []
        self.filteredFlowVectors = []

        #list of dict containing OF magnitudes per frame and number of filtered points
        self.of_magnitudes = []

        self.curr_gyro_cam = np.array([0.0,0.0,0.0], dtype=np.float32)          #gyroscope, camera frame : gx, gy, gz
        self.curr_velocity_cam = np.array([0.0,0.0,0.0], dtype=np.float32)      #forward velocity

        self.prevFilteredAltitude = 0.0
        self.filteredAltitude = 0.0
        self.avgAltitude = 0.0
        self.unfilteredAltitude = 0.0

        self.smoothingFilterType = 0
        self.complementaryK = 0.9
        self.lpfK = 0.1
        self.altitudeType = 0
        self.saturationValue = 50.0

        self.magnitudeThresholdPixel = 10.0
        self.boundThreshold = 1.5

        self.frameID = 0

        # load config
        self._loadParametersFromYAML(yaml_path)

        # event-related config
        events_cfg = self.config["EVENTS"]
        self.events_dir = events_cfg["scene"]
        self.side = events_cfg.get("side", "left")
        self.H = events_cfg.get("H", 640)
        self.W = events_cfg.get("W", 480)
        self.rectify = events_cfg.get("rectify", True)


        self.slicing_type = self.config["SLICING"]["type"]
        self.fixed_dt_ms  = self.config["SLICING"].get("dt_ms", None) if self.slicing_type == "fixed" else None

        self.use_valid_frame_range = events_cfg.get("use_valid_frame_range", False)
        print(f"[VisionNodeEventsPlayback] use_valid_frame_range = {self.use_valid_frame_range}")

        # controlla se adaptive
        self.use_adaptive = (self.slicing_type == "adaptive")

        # crea PID SOLO se adaptive
        self.adaptiveSlicer = AdaptiveSlicerPID(
            self.config,
            enabled=self.use_adaptive         # <--- ora dipende da SLICING.type
        )

        # fps per LK, se adaptive lo calcoliamo dinamicamente
        if self.slicing_type == "fixed":
            self.fps = 1000.0 / self.fixed_dt_ms
        elif self.slicing_type == "adaptive":
            self.fps = 1000.0 / self.adaptiveSlicer.initial_dt_ms

        self.deltaTms = 1000.0 / self.fps


        # FAST detector
        self._initializeFeatureDetector()

        # pre-carica eventi per il caso adaptive
        self._initializeEventDataAdaptive()


        #print on screen the self.config for verification
        print("Loaded configuration:")
        print(yaml.dump(self.config, default_flow_style=False))

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------
    def _loadParametersFromYAML(self, yaml_path):
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # FAST
        if "FAST" in self.config:
            fast = self.config["FAST"]
            self.fastParams.threshold = fast.get("threshold", 10)
            self.fastParams.nonmaxSuppression = fast.get("nonmaxSuppression", True)
            rsf = fast.get("randomSampleFilter", {})
            self.fastParams.randomSampleFilterEnable = rsf.get("enable", False)
            self.fastParams.randomSampleFilterRatio = rsf.get("ratio", 0.0)
            gs = fast.get("gradientScoring", {})
            self.fastParams.gradientScoringEnable = gs.get("enable", False)
            self.fastParams.desiredFeatures = gs.get("desiredFeatures", 200)
            self.fastParams.safeFeatures = fast.get("safeFeatures", False)

        # CAMERA
        if "CAMERA" in self.config:
            cam = self.config["CAMERA"]

            # risoluzione: se hai width/height e non "resolution"
            if "resolution" in cam:
                w = cam["resolution"].get("width", 640)
                h = cam["resolution"].get("height", 480)
            else:
                w = cam.get("width", 640)
                h = cam.get("height", 480)
            self.camParams.resolution = (w, h)

            self.camParams.fx = cam.get("fx", 196.73614846589038)
            self.camParams.fy = cam.get("fy", 196.5361311749205)
            self.camParams.cx = cam.get("cx", 173.9835140415677)
            self.camParams.cy = cam.get("cy", 134.63020721240977)
            self.camParams.pixelSize = cam.get("pixelSize", 9e-6)
            self.camParams.inclination = cam.get("inclination", 45.0)

        # LK
        if "LK" in self.config:
            lk_conf = self.config["LK"]
            ws = lk_conf.get("winSize", {})
            self.lkParams.winSize = (ws.get("width", 21), ws.get("height", 21))
            self.lkParams.maxLevel = lk_conf.get("maxLevel", 5)
            crit = lk_conf.get("criteria", {})
            maxCount = crit.get("maxCount", 40)
            epsilon = crit.get("epsilon", 0.01)
            self.lkParams.flags = lk_conf.get("flags", 0)
            self.lkParams.minEigThreshold = lk_conf.get("minEigThreshold", 0.001)
            self.lkParams.criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                maxCount,
                epsilon
            )

        # REJECTION_FILTER
        if "REJECTION_FILTER" in self.config:
            rf = self.config["REJECTION_FILTER"]
            self.magnitudeThresholdPixel = rf.get("magnitudeThresholdPixel", 75)
            self.boundThreshold = rf.get("boundThreshold", 1.5)

        # visualization
        if "visualizeImage" in self.config:
            self.visualizeImage = bool(self.config["visualizeImage"])
        if "delayVisualize" in self.config:
            self.delayVisualize = int(self.config["delayVisualize"])

    # --------------------------------------------------------
    # Feature detector
    # --------------------------------------------------------
    def _initializeFeatureDetector(self):
        self.fastDetector = cv2.FastFeatureDetector_create(
            threshold=self.fastParams.threshold,
            nonmaxSuppression=self.fastParams.nonmaxSuppression
        )

    # --------------------------------------------------------
    # Preload eventi per modalità adaptive
    # --------------------------------------------------------
    def _initializeEventDataAdaptive(self):
        """
        Carica eventi e rectify map per implementare slicing adaptive
        (caso adaptiveSlicing.enable=True).
        """
        if not self.use_adaptive:
            self.all_evs = None
            self.ts_us = None
            return

        print("[VisionNodeEventsPlayback] Preloading events for adaptive slicing.")

        # HDF5 events
        h5in = glob.glob(os.path.join(self.events_dir, f"*_data.hdf5"))
        assert len(h5in) == 1, f"Found {len(h5in)} HDF5 files, expected 1."
        datain = h5py.File(h5in[0], 'r')

        self.all_evs = datain["davis"][self.side]["events"][:]  # (x,y,t,pol)
        datain.close()

        ts_seconds = self.all_evs[:, 2].astype(np.float64)
        self.ts_us = (ts_seconds * 1e6).astype(np.int64)
        self.ts_start = self.ts_us[0]
        self.ts_end = self.ts_us[-1]

        rect_file = os.path.join(self.events_dir, f"rectify_map_{self.side}.h5")

        print(f"[VisionNodeEventsPlayback] Loaded {len(self.all_evs)} events, ts range [{self.ts_start}, {self.ts_end}]")

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------
    def run(self):
        if self.slicing_type == "mvsec" or self.slicing_type == "fixed":
            self._run_fixed_slicing()

        else:
            raise ValueError(f"Unknown slicing type: {self.slicing_type}")

        # self.plot_of_magnitudes()

    # --------------------------------------------------------
    # RUN: fixed slicing con mvsec_evs_iterator
    # --------------------------------------------------------
    def _run_fixed_slicing(self):
        print("[VisionNodeEventsPlayback] Running unified event+GT iterator.")

        iterator = dvxplorer_evs_iterator(
            self.events_dir,
            dT_ms=self.fixed_dt_ms,
            H=self.H,
            W=self.W,
            adaptive_slicer_pid=self.adaptiveSlicer,
            slicing_type=self.slicing_type,
            use_valid_frame_range=self.use_valid_frame_range
        )

        for slice_data in iterator:
            #retrieve data from the returned dict
            event_frame = slice_data["event_frame"]
            t_us = slice_data["t1_us"]
            dt_ms = slice_data["dt_ms"]

            dv_imu = slice_data["dv_imu"]
            px4_imu = slice_data["px4_imu"]
            px4_state = slice_data["px4_state"]


            print(f"Current event frame timestamp : {t_us} us, dt_ms={dt_ms:.2f}")

            self._processEventFrame(event_frame, t_us, dv_imu, px4_state)
            self.frameID += 1


    # --------------------------------------------------------
    # Processa UN event frame (come processFrames(), ma senza altitude)
    # --------------------------------------------------------
    def _processEventFrame(self, event_frame, t_us, dv_imu_slice, px4_state_slice):
        """
        Per ora: optical flow + feature detection, esattamente come nel caso frames.
        (La GT è già in self.current_gt_flow se disponibile.)
        """
        self.currFrame = event_frame

        #get current imu data
        # ============================================================
        # EXTRACT IMU + PX4 DATA FOR THIS EVENT SLICE
        # ============================================================

        # 1) Filter IMU → gyro in camera frame
        self.curr_gyro_cam = self._imu_to_cam(dv_imu_slice)

        # 2) Extract PX4 velocity → transform to camera frame
        self.curr_velocity_cam = self._px4_velocity_to_cam(px4_state_slice)

        # 3) Needed for altitude orientation
        self.cosRoll = math.cos(np.mean(px4_state_slice.get("RollAngle", 0.0)))
        self.sinRoll = math.sin(np.mean(px4_state_slice.get("RollAngle", 0.0)))
        self.cosPitch = math.cos(np.mean(px4_state_slice.get("PitchAngle", 0.0)))
        self.sinPitch = math.sin(np.mean(px4_state_slice.get("PitchAngle", 0.0)))

        # se è il primissimo frame, inizializza solo i punti
        if self.prevFrame is None:
            self.prevFrame = self.currFrame.copy()
            self.prevPoints = []
            self._applyCornerDetection(self.currFrame, outputArray='prevPoints')
            return

        self._calculateOpticalFlow(self.currFrame)

        #apply visualization eventually
        if self.visualizeImage:
            # visualize_image(self.currFrame,self.currPoints,self.prevPoints,self.status)
            visualize_filtered_flow(self.currFrame, self.filteredFlowVectors, win_name="OF_filtered")
            cv2.waitKey(self.delayVisualize)



        self.prevPoints.clear()

        self._applyCornerDetection(self.currFrame, outputArray='prevPoints')

        self.prevFrame = self.currFrame.copy()


        #estimate altitude
        T_cam = self.curr_velocity_cam

        altitudes = []
        for fv in self.filteredFlowVectors:
            depth = self._estimateDepth(fv, T_cam)
            alt = self._estimateAltitude(fv, depth)
            if not math.isnan(alt):
                altitudes.append(alt)

        if not altitudes:
            self.avgAltitude = self.prevFilteredAltitude
        else:
            if self.altitudeType == 1:
                altitudes.sort()
                self.avgAltitude = altitudes[len(altitudes)//2]
            else:
                self.avgAltitude = sum(altitudes)/len(altitudes)

        if self.avgAltitude >= self.saturationValue:
            self.avgAltitude = self.saturationValue

        if math.isnan(self.avgAltitude):
            self.avgAltitude = self.prevFilteredAltitude

        self.unfilteredAltitude = self.avgAltitude

        # smoothing
        if self.smoothingFilterType == 0:
            dt_s = self.deltaTms/1000.0
            self.filteredAltitude = complementaryFilter(self.avgAltitude,
                                                        self.prevFilteredAltitude,
                                                        self.complementaryK,
                                                        dt_s)
        else:
            self.filteredAltitude = LPFilter(self.avgAltitude,
                                            self.prevFilteredAltitude,
                                            self.lpfK)

        if not math.isnan(self.filteredAltitude):
            self.prevFilteredAltitude = self.filteredAltitude


        #log telemetry data + estimated altitude

        # 1) timestamp
        curr_t = t_us

        # 2) estimated altitude
        est_alt = float(self.filteredAltitude)

        # 3) GT altitude from PX4
        # px4_state_slice is a dict of arrays → take mean if exists
        # if "DistanceGround" in px4_state_slice:
        #     gt_alt = float(np.mean(px4_state_slice["DistanceGround"]))
        if "Lidar" in px4_state_slice:
            gt_alt = float(np.mean(px4_state_slice["Lidar"]))
        elif "distance_ground" in px4_state_slice:
            gt_alt = float(np.mean(px4_state_slice["distance_ground"]))
        else:
            gt_alt = float("nan")   # no GT available

        print(f"[Altitude] t={curr_t} us | est={est_alt:.3f} m | gt lidar={gt_alt:.3f} m")


    # --------------------------------------------------------
    # FAST corner detection (copiato dal tuo applyCornerDetection)
    # --------------------------------------------------------
    def _applyCornerDetection(self, image, outputArray='prevPoints'):
        keypoints = self.fastDetector.detect(image, None)
        detectedFeatures = len(keypoints)

        if self.fastParams.randomSampleFilterEnable:
            keypoints = randomlySampleKeypoints(
                keypoints,
                self.fastParams.desiredFeatures,
                self.fastParams.randomSampleFilterRatio
            )
        if self.fastParams.gradientScoringEnable:
            keypoints = scoreAndRankKeypointsUsingGradient(
                keypoints,
                image,
                self.fastParams.desiredFeatures
            )
        if (not self.fastParams.randomSampleFilterEnable) and (not self.fastParams.gradientScoringEnable):
            # safeFeatures logic
            if self.fastParams.safeFeatures and detectedFeatures < 0.5 * self.fastParams.desiredFeatures:
                self.fastDetector.setThreshold(50)
                keypoints = self.fastDetector.detect(image, None)
                keypoints = randomlySampleKeypoints(keypoints, self.fastParams.desiredFeatures, 0.5)
                keypoints = scoreAndRankKeypointsUsingGradient(keypoints, image, self.fastParams.desiredFeatures)
                self.fastDetector.setThreshold(self.fastParams.threshold)
            else:
                keypoints = randomlySampleKeypoints(keypoints, self.fastParams.desiredFeatures, 0.0)

        points = cv2.KeyPoint_convert(keypoints)
        if outputArray == 'prevPoints':
            self.prevPoints = points.tolist()
        else:
            self.nextPrevPoints = points.tolist()

    # --------------------------------------------------------
    # Optical flow Lucas–Kanade (copiato e adattato)
    # --------------------------------------------------------
    def _calculateOpticalFlow(self, currFrame):
        if self.prevFrame is not None and len(self.prevPoints) > 0:
            self.currPoints, self.status, self.err = cv2.calcOpticalFlowPyrLK(
                self.prevFrame, currFrame,
                np.float32(self.prevPoints),
                None,
                winSize=self.lkParams.winSize,
                maxLevel=self.lkParams.maxLevel,
                criteria=self.lkParams.criteria,
                flags=self.lkParams.flags,
                minEigThreshold=self.lkParams.minEigThreshold
            )

            self.flowVectors.clear()

            if self.currPoints is not None and self.status is not None:
                for i in range(len(self.currPoints)):
                    if self.status[i] == 1:
                        p1 = self.prevPoints[i]
                        p2 = self.currPoints[i]
                        fv = OFVectorFrame(p1, p2, self.fps, self.camParams)
                        self.flowVectors.append(fv)

            # Outlier rejection
            self.filteredFlowVectors = rejectOutliersFrame(
                self.flowVectors,
                self.magnitudeThresholdPixel,
                self.boundThreshold
            )

            #apply derotation with the current gyro data
            for fv in self.filteredFlowVectors:
                self._applyDerotation3D_events(fv, self.curr_gyro_cam)

        else:
            print("FIRST EVENT FRAME, skipping OF...")

    # --------------------------------------------------------
    # IMU → FRD → CAMERA FRAME MAPPINGS
    # --------------------------------------------------------
    def _imu_to_cam(self, dv_imu_slice):
        """
        dv_imu_slice has Nx1 arrays: ax,ay,az,gx,gy,gz,temperature...
        We take the MEAN over the window.
        Convert FLU → FRD → camera frame (same logic as frames version).
        """
        if len(dv_imu_slice) == 0:
            return np.zeros(3, dtype=np.float32)

        gx = np.mean(dv_imu_slice.get("gx", 0.0))
        gy = np.mean(dv_imu_slice.get("gy", 0.0))
        gz = np.mean(dv_imu_slice.get("gz", 0.0))

        # FLU → FRD (gx, -gy, -gz)
        gyro_frd = np.array([gx, -gy, -gz], dtype=np.float32)

        # FRD → CAM
        return bodyToCam(gyro_frd, self.camParams)


    def _px4_velocity_to_cam(self, px4_state_slice):
        """
        px4_state_slice holds velocities in FLU or FRD depending on logs.
        We take vx, vy, vz and convert them to camera frame.
        """
        if len(px4_state_slice) == 0:
            return np.zeros(3, dtype=np.float32)

        vx = np.mean(px4_state_slice.get("Airspeed", 0.0))
        vy = 0.0
        vz = 0.0

        vel_frd = np.array([vx, vy, vz], dtype=np.float32)
        return bodyToCam(vel_frd, self.camParams)

    def _applyDerotation3D_events(self, ofVector, gyro_cam):
        """
        Same as applyDerotation3D from frames version,
        but using the IMU extracted from events.
        """
        norm_a = np.linalg.norm(ofVector.AMeter)

        Pprime_ms = np.array([
            ofVector.uPixelSec * self.camParams.pixelSize,
            ofVector.vPixelSec * self.camParams.pixelSize
        ], dtype=np.float32)

        PpPprime_ms = np.array([Pprime_ms[0]/norm_a, Pprime_ms[1]/norm_a, 0.0], dtype=np.float32)

        dot_val = np.dot(PpPprime_ms, ofVector.directionVector)
        P = PpPprime_ms - dot_val * ofVector.directionVector

        # rotation term
        cross_val = np.cross(gyro_cam, ofVector.directionVector)
        RotOF = -cross_val

        ofVector.P = P - RotOF

        OF_derotated = self._getDerotatedOF_ms_events(ofVector.P, ofVector.directionVector, ofVector.AMeter)

        derotNextX = ofVector.position[0] + OF_derotated[0]*self.deltaTms/(self.camParams.pixelSize*1e3)
        derotNextY = ofVector.position[1] + OF_derotated[1]*self.deltaTms/(self.camParams.pixelSize*1e3)

        ofVector.nextPosition = (derotNextX, derotNextY)
        ofVector.deltaX = derotNextX - ofVector.position[0]
        ofVector.deltaY = derotNextY - ofVector.position[1]


    def _getDerotatedOF_ms_events(self, P_derotated, d_direction, aVector):
        dot_val = np.dot(P_derotated, d_direction)
        Pprime = P_derotated + dot_val*d_direction
        scale = np.linalg.norm(aVector)
        Pprime *= scale
        return np.array([Pprime[0], Pprime[1]], dtype=np.float32)


    def _estimateDepth(self, ofVector, T_cam):
        TdotD = np.dot(T_cam, ofVector.directionVector)
        tmp = T_cam - TdotD * ofVector.directionVector
        num = np.linalg.norm(tmp)
        denom = np.linalg.norm(ofVector.P)
        if denom < 1e-9:
            return float('nan')
        return num / denom


    def _estimateAltitude(self, ofVector, depth):
        directionVector_body = camToBody(ofVector.directionVector, self.camParams)
        directionVector_inertial = bodyToInertial(
            directionVector_body,
            self.cosRoll, self.sinRoll,
            self.cosPitch, self.sinPitch
        )
        cosTheta = directionVector_inertial[2]
        return depth * cosTheta
