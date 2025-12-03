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
from testing.utils.load_utils_fpv import fpv_evs_iterator
from testing.utils.viz_utils import visualize_image, visualize_image_log


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
    quat_to_rotmat,
    quat_to_euler,
    compute_attitude_trig,
    compute_initial_attitude_offset,
    
)

# ============================================================
# VisionNodeEventsPlayback
# ============================================================

class VisionNodeUZHFPVEventsPlayback:
    """
    Variante event-based:
    - usa eventi UZH-FPV camera dataset
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
        self.H = events_cfg.get("H", 260)
        self.W = events_cfg.get("W", 346)
        self.rectify = events_cfg.get("rectify", True)      #if rectify, apply pinhole model, otherwise fisheye

        if self.rectify: 
            self._updateCameraParameters()

        self.initial_altitude_offset = self._get_initial_offset()   #compute the initial altitude over ground

        self.initial_roll_angle = 0.0
        self.initial_pitch_angle = 0.0
        self.initial_roll_angle, self.initial_pitch_angle = compute_initial_attitude_offset(self.events_dir)

        #self.pitch angle is the only one which has a large offset, so the roll angle is ok
        self.initial_pitch_angle = 0.0

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

            self.camParams.model = "fisheye"    #since FPV uses wide lens (120°). will change later eventually to "pinhole" if rectify is True

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

    def _updateCameraParameters(self):
        #udpate instrinsics parameters if the rectification is performed
        #check if the rectification flag is on. if so, we will rectify the image
        #so the true calibration parameters are the rectified ones, found at : path/calib_undist.txt
        if self.rectify:
            calib_path = os.path.join(self.events_dir, "calib_undist.txt")
            if os.path.isfile(calib_path):
                with open(calib_path, 'r') as f:
                    line = f.readline().strip()
                    params = line.split()
                    assert len(params) >= 4, "calib_undist.txt badly formatted"
                    self.camParams.fx = float(params[0])
                    self.camParams.fy = float(params[1])
                    self.camParams.cx = float(params[2])
                    self.camParams.cy = float(params[3])

                    self.camParams.model = "pinhole"  # set model to pinhole if rectified

                print(f"[VisionNodeUZHFPVEventsPlayback] Using RECTIFIED intrinsics:"
                    f" fx={self.camParams.fx:.3f}, fy={self.camParams.fy:.3f}, "
                    f"cx={self.camParams.cx:.3f}, cy={self.camParams.cy:.3f}")
            else:
                print(f"[WARNING] Rectification enabled but calib_undist.txt NOT FOUND at {calib_path}")

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

        print("[VisionNodeUZHFPVEventsPlayback] Preloading events for adaptive slicing.")

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

        iterator = fpv_evs_iterator(
            self.events_dir,
            dT_ms=self.fixed_dt_ms,
            H=self.H,
            W=self.W,
            adaptive_slicer_pid=self.adaptiveSlicer,
            slicing_type=self.slicing_type,
            use_valid_frame_range=self.use_valid_frame_range,
            rectify=self.rectify
        )

        for slice_data in iterator:
            #retrieve data from the returned dict
            event_frame = slice_data["event_frame"]
            t_us = slice_data["t1_us"]
            dt_ms = slice_data["dt_ms"]

            cam_imu = slice_data["cam_imu"]
            gt_state = slice_data["gt_state"]   #relative to body frame

            #relative to camera frame. use this for getting the velocity
            gt_cam_state = slice_data["gt_cam_state"]   


            print(f"Current event frame timestamp : {t_us} us, dt_ms={dt_ms:.2f}")

            self._processEventFrame(event_frame, t_us, cam_imu, gt_cam_state, gt_state)
            self.frameID += 1


    # --------------------------------------------------------
    # Processa UN event frame (come processFrames(), ma senza altitude)
    # --------------------------------------------------------
    def _processEventFrame(self, event_frame, t_us, cam_imu_slice, gt_cam_state_slice, gt_state_slice=None):
        """
        Per ora: optical flow + feature detection, esattamente come nel caso frames.
        (La GT è già in self.current_gt_flow se disponibile.)
        """
        self.currFrame = event_frame

        #get current imu data
        # ============================================================
        # EXTRACT IMU 
        # ============================================================

        # 1) Average IMU over time window
        self.curr_gyro_cam = self._get_imu(cam_imu_slice)

        self.curr_position_world = self._get_position_world(gt_state_slice)

        # 2) Extract velocity from GT cam state. so velocity is already in camera frame.
        self.curr_velocity_cam = self._gt_velocity_to_cam(gt_state_slice)
        self.curr_velocity_body = self._gt_velocity_and_attitude(gt_state_slice)

        # 3) Needed for altitude orientation
        self.current_roll_angle_rad, self.current_pitch_angle_rad = compute_attitude_trig(gt_state_slice, self.initial_roll_angle, self.initial_pitch_angle)

        print(f"[Attitude] t={t_us} us \t \t| roll={math.degrees(self.current_roll_angle_rad):.3f} deg, pitch={math.degrees(self.current_pitch_angle_rad):.3f} deg")

        # se è il primissimo frame, inizializza solo i punti
        if self.prevFrame is None:
            self.prevFrame = self.currFrame.copy()
            self.prevPoints = []
            self._applyCornerDetection(self.currFrame, outputArray='prevPoints')
            return

        self._calculateOpticalFlow(self.currFrame)


        #loggin data
        # Print gyro in deg/s
        gyro_deg = np.degrees(self.curr_gyro_cam)
        # print(f"[IMU] t={t_us} us \t \t \t| gyro_cam = \t[{gyro_deg[0]:.3f}, {gyro_deg[1]:.3f}, {gyro_deg[2]:.3f}] deg/s")
        # print(f"[GT CAM VELOCITY] t={t_us} us \t \t| vel_cam = [{self.curr_velocity_cam[0]:.3f}, {self.curr_velocity_cam[1]:.3f}, {self.curr_velocity_cam[2]:.3f}] m/s")
        # print(f"[GT BODY VELOCITY] t={t_us} us \t \t| vel_body = [{self.curr_velocity_body[0]:.3f}, {self.curr_velocity_body[1]:.3f}, {self.curr_velocity_body[2]:.3f}] m/s")
        # print(f"[GT Cam Position] t={t_us} us \t| pos_cam = \t[{np.mean(gt_cam_state_slice.get('px', 0.0)):.3f}, {np.mean(gt_cam_state_slice.get('py', 0.0)):.3f}, {np.mean(gt_cam_state_slice.get('pz', 0.0)):.3f}] m")
        # print(f"[Attitude] t={t_us} us \t \t| roll={math.degrees(math.acos(self.cosRoll)):.3f} deg, pitch={math.degrees(math.acos(self.cosPitch)):.3f} deg")


        #apply visualization eventually
        if self.visualizeImage:
            # visualize_image(self.currFrame,self.currPoints,self.prevPoints,self.status)
            visualize_image_log(self.currFrame, self)
            # visualize_filtered_flow(self.currFrame, self.filteredFlowVectors, win_name="OF_filtered")
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

        # 3) GT altitude from GT state, using "pz". it points upwards. the z value is set around 1 meters above ground, so sum up 1.0 m
        if "pz" in gt_cam_state_slice:
            gt_alt = float(np.mean(gt_cam_state_slice["pz"])) + self.initial_altitude_offset
        else:
            gt_alt = float("nan")   # no GT available

        # print(f"[Altitude] t={curr_t} us | est={est_alt:.3f} m | gt ={gt_alt:.3f} m")


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
    def _get_imu(self, cam_imu_slice):
        """
        cam_imu_slice has Nx1 arrays: ax,ay,az,gx,gy,gz
        We take the MEAN over the window.
        """
        if len(cam_imu_slice) == 0:
            return np.zeros(3, dtype=np.float32)

        gx = np.mean(cam_imu_slice.get("gx", 0.0))
        gy = np.mean(cam_imu_slice.get("gy", 0.0))
        gz = np.mean(cam_imu_slice.get("gz", 0.0))

        #imu is already in camera frame (from miniDAVIS346)
        gyro = np.array([gx, gy, gz], dtype=np.float32)
        return gyro


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
            self.current_roll_angle_rad,
            self.current_pitch_angle_rad
        )
        cosTheta = directionVector_inertial[2]
        return depth * cosTheta


    def _gt_velocity_to_cam(self, gt_cam_state_slice):
        """
        gt_cam_state_slice:
            dict-like with arrays:
            - 'timestamp' [us]
            - 'px','py','pz'  (posizione camera nel mondo)
            - 'qx','qy','qz','qw' (orientazione camera -> world)
        Ritorna velocità media [vx, vy, vz] nel frame camera.
        """
        # nessun dato
        if len(gt_cam_state_slice) == 0:
            return np.zeros(3, dtype=np.float32)

        t_us = np.asarray(gt_cam_state_slice["timestamp"], dtype=np.int64)
        px = np.asarray(gt_cam_state_slice["px"], dtype=np.float64)
        py = np.asarray(gt_cam_state_slice["py"], dtype=np.float64)
        pz = np.asarray(gt_cam_state_slice["pz"], dtype=np.float64)
        qx = np.asarray(gt_cam_state_slice["qx"], dtype=np.float64)
        qy = np.asarray(gt_cam_state_slice["qy"], dtype=np.float64)
        qz = np.asarray(gt_cam_state_slice["qz"], dtype=np.float64)
        qw = np.asarray(gt_cam_state_slice["qw"], dtype=np.float64)

        N = t_us.shape[0]
        if N < 2:
            # con un solo campione non puoi stimare la velocità
            return np.zeros(3, dtype=np.float32)

        # posizione nel mondo
        p_W = np.stack([px, py, pz], axis=1)  # (N,3)

        # converti i timestamp in secondi (da us)
        t = t_us.astype(np.float64) * 1e-6

        # velocità nel mondo con differenze finite (centrali, avanti/indietro ai bordi)
        v_W = np.zeros_like(p_W)
        # centrali per gli interni
        v_W[1:-1] = (p_W[2:] - p_W[:-2]) / (t[2:, None] - t[:-2, None])
        # forward/backward ai bordi
        v_W[0] = (p_W[1] - p_W[0]) / (t[1] - t[0])
        v_W[-1] = (p_W[-1] - p_W[-2]) / (t[-1] - t[-2])

        # ruota le velocità nel frame camera
        v_C = np.zeros_like(v_W)
        for i in range(N):
            R_WC = quat_to_rotmat(qx[i], qy[i], qz[i], qw[i])  # camera -> world
            R_CW = R_WC.T
            v_C[i] = R_CW @ v_W[i]

        # velocità media nello slice (in camera frame)
        v_cam_mean = v_C.mean(axis=0).astype(np.float32)
        return v_cam_mean
    

    def _gt_velocity_and_attitude(self,
                              gt_state_slice,
                              quat_is_body_to_world=True):

        if len(gt_state_slice) < 2:
            return np.zeros(3, dtype=np.float32)

        # === extract arrays ===
        t_us = np.asarray(gt_state_slice["timestamp"], dtype=np.int64)
        pos = np.vstack([gt_state_slice["px"],
                        gt_state_slice["py"],
                        gt_state_slice["pz"]]).T

        dt = (t_us[-1] - t_us[0]) * 1e-6
        if dt <= 0:
            return np.zeros(3, dtype=np.float32)

        # === world velocity ===
        dp_W = pos[-1] - pos[0]
        v_W  = dp_W / dt

        # === orientation at midpoint ===
        mid_idx = len(t_us) // 2
        qx = gt_state_slice["qx"][mid_idx]
        qy = gt_state_slice["qy"][mid_idx]
        qz = gt_state_slice["qz"][mid_idx]
        qw = gt_state_slice["qw"][mid_idx]

        R_WB = quat_to_rotmat(qx, qy, qz, qw)

        # convert to body velocity if needed
        if quat_is_body_to_world:
            v_B = R_WB.T @ v_W
        else:
            v_B = R_WB @ v_W

        # === extract Euler ===
        roll, pitch, yaw = quat_to_euler(qx, qy, qz, qw)  # radians
        roll_deg  = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg   = np.degrees(yaw)

        # print(f"[GT] roll = {roll_deg:.2f} deg, pitch = {pitch_deg:.2f} deg, yaw = {yaw_deg:.2f} deg")
        # print(f"[GT] v_W = [{v_W[0]:.3f}, {v_W[1]:.3f}, {v_W[2]:.3f}] m/s")
        # print(f"[GT] v_B = [{v_B[0]:.3f}, {v_B[1]:.3f}, {v_B[2]:.3f}] m/s")

        return v_B.astype(np.float32)


    def _gt_velocity_to_body(self,
                            gt_state_slice,
                            quat_is_body_to_world=True,
                            output_convention="FLU"):

        if len(gt_state_slice) < 2:
            return np.zeros(3, dtype=np.float32)

        t_us = np.asarray(gt_state_slice["timestamp"], dtype=np.int64)
        pos = np.vstack([gt_state_slice["px"],
                        gt_state_slice["py"],
                        gt_state_slice["pz"]]).T

        dt = (t_us[-1] - t_us[0]) * 1e-6
        if dt <= 0:
            return np.zeros(3, dtype=np.float32)

        # world velocity
        dp_W = pos[-1] - pos[0]
        # print(f"[POS WORLD] pos_W: {pos[-1]} m")
        # print(f"[delta POS WORLD] dp_W: {dp_W} ")
        

        v_W = dp_W / dt

        # use midpoint orientation (better instantaneous estimate)
        mid_idx = len(t_us) // 2
        q = [gt_state_slice["qx"][mid_idx],
            gt_state_slice["qy"][mid_idx],
            gt_state_slice["qz"][mid_idx],
            gt_state_slice["qw"][mid_idx]]

        R_WB = quat_to_rotmat(*q)

        # convert to body frame
        if quat_is_body_to_world:
            v_B = R_WB.T @ v_W  # world -> body
        else:
            v_B = R_WB @ v_W

        # convert body convention if needed
        if output_convention.upper() == "FRD":
            v_B = np.array([v_B[0], -v_B[1], -v_B[2]])

        return v_B.astype(np.float32)




    def _get_initial_offset(self):
        """
        GT pose and altitude "pz" is relative to inertial frame. their 0 altitude is not applied to the exact ground level.
        So read the stamped_gt_cam_us.txt inside the events_dir to get the initial offset.
        """

        #read first line of stamped_gt_cam_us.txt
        gt_file = os.path.join(self.events_dir, "stamped_groundtruth_us_cam.txt")
        if not os.path.isfile(gt_file):
            print(f"[WARNING] GT file not found: {gt_file}")
            print("[WARNING] Cannot compute initial altitude offset. Using 0.0 m.")
            return 0.0
        
        with open(gt_file, 'r') as f:
            first_line = f.readline().strip()
            parts = first_line.split()
            if len(parts) < 5:
                print(f"[WARNING] GT file badly formatted: {gt_file}")
                print("[WARNING] Cannot compute initial altitude offset. Using 0.0 m.")
                return 0.0
            #first line is : 22833964.000000 -3.736644 4.424751 -1.010792 0.911352 -0.166714 0.074742 -0.368861
            #where columns are: timestamp [us], px, py, pz, qx, qy, qz, qw
            pz = float(parts[3])  # assuming pz is the 4th column
            initial_offset = -pz
            print(f"[INFO] Initial altitude offset computed: {initial_offset:.3f} m")
            return initial_offset
    
    def _get_position_world(self, gt_state_slice):
        """
        Extracts the mean position in world frame from the gt_state_slice.
        """
        if len(gt_state_slice) == 0:
            return np.zeros(3, dtype=np.float32)

        px = np.mean(gt_state_slice.get("px", 0.0))
        py = np.mean(gt_state_slice.get("py", 0.0))
        pz = np.mean(gt_state_slice.get("pz", 0.0))

        position_world = np.array([px, py, pz], dtype=np.float32)
        return position_world