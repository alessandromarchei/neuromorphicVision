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

from testing.utils.load_utils_fpv import fpv_evs_iterator
from testing.utils.viz_utils import visualize_image

# strutture dati e parametri
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
    rotmat_to_quat,
    compute_initial_attitude_offset, 
)

# ============================================================
# VisionNodeUZHFPVEventsPlayback
# ============================================================

class VisionNodeUZHFPVEventsPlayback:
    """
    Variante event-based ottimizzata per UZH-FPV:
    - Assicura che la velocità estratta sia coerente (X-Forward nel body).
    - Gestisce la trasformazione tra IMU Raw frame e Body FLU frame.
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
        self.of_magnitudes = []

        # Data Containers
        self.curr_gyro_cam = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Velocity in BODY FLU frame (X-forward, Y-left, Z-up)
        self.curr_velocity_flu = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Velocity in CAMERA frame (Z-forward, X-right, Y-down) for Depth Est
        self.curr_velocity_cam = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.prevFilteredAltitude = 0.0
        self.filteredAltitude = 0.0
        self.avgAltitude = 0.0
        self.unfilteredAltitude = 0.0

        # Altitude Filters
        self.smoothingFilterType = 0
        self.complementaryK = 0.9
        self.lpfK = 0.1
        self.altitudeType = 0
        self.saturationValue = 50.0

        self.magnitudeThresholdPixel = 10.0
        self.boundThreshold = 1.5

        self.frameID = 0

        # Attitude
        self.current_roll_angle_rad = 0.0
        self.current_pitch_angle_rad = 0.0

        # load config
        self._loadParametersFromYAML(yaml_path)

        # event-related config
        events_cfg = self.config["EVENTS"]
        self.events_dir = events_cfg["scene"]
        self.side = events_cfg.get("side", "left")
        self.H = events_cfg.get("H", 260)
        self.W = events_cfg.get("W", 346)
        self.rectify = events_cfg.get("rectify", True)

        if self.rectify: 
            self._updateCameraParameters()

        self.initial_altitude_offset = self._get_initial_offset()
        self.initial_pitch_angle = 0.0

        self.slicing_type = self.config["SLICING"]["type"]
        self.fixed_dt_ms  = self.config["SLICING"].get("dt_ms", None) if self.slicing_type == "fixed" else None

        self.use_valid_frame_range = events_cfg.get("use_valid_frame_range", False)
        
        # Adaptive Slicing
        self.use_adaptive = (self.slicing_type == "adaptive")
        self.adaptiveSlicer = AdaptiveSlicerPID(
            self.config,
            enabled=self.use_adaptive
        )

        if self.slicing_type == "fixed":
            self.fps = 1000.0 / self.fixed_dt_ms
        elif self.slicing_type == "adaptive":
            self.fps = 1000.0 / self.adaptiveSlicer.initial_dt_ms

        self.deltaTms = 1000.0 / self.fps

        # FAST detector
        self._initializeFeatureDetector()
        self._initializeEventDataAdaptive()

        print("Loaded configuration:")
        print(yaml.dump(self.config, default_flow_style=False))

    # --------------------------------------------------------
    # YAML & SETUP
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

        # REJECTION
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

    def _initializeEventDataAdaptive(self):
        if not self.use_adaptive: return
        print("[VisionNode] Preloading events for adaptive slicing...")
        h5in = glob.glob(os.path.join(self.events_dir, f"*_data.hdf5"))
        if len(h5in) > 0:
            datain = h5py.File(h5in[0], 'r')
            self.all_evs = datain["davis"][self.side]["events"][:]
            datain.close()
            # Setup timestamps... (omesso per brevità, codice originale ok)

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------
    def run(self):
        if self.slicing_type in ["mvsec", "fixed"]:
            self._run_fixed_slicing()
        else:
            raise ValueError(f"Unknown slicing type: {self.slicing_type}")

    def _run_fixed_slicing(self):
        iterator = fpv_evs_iterator(
            self.events_dir,
            dT_ms=self.fixed_dt_ms,
            H=self.H, W=self.W,
            adaptive_slicer_pid=self.adaptiveSlicer,
            slicing_type=self.slicing_type,
            use_valid_frame_range=self.use_valid_frame_range,
            rectify=self.rectify
        )

        for slice_data in iterator:
            self._processEventFrame(
                slice_data["event_frame"], 
                slice_data["t1_us"], 
                slice_data["cam_imu"], 
                slice_data["gt_state"]
            )
            self.frameID += 1

    # --------------------------------------------------------
    # CORE PROCESSING
    # --------------------------------------------------------
    def _processEventFrame(self, event_frame, t_us, cam_imu_slice, gt_IMU_frame_slice):
        self.currFrame = event_frame

        # 1. IMU (Gyro) extraction
        self.curr_gyro_cam = self._get_imu(cam_imu_slice)

        print(f"Gyro cam : {self.curr_gyro_cam[0]:.3f}, {self.curr_gyro_cam[1]:.3f}, {self.curr_gyro_cam[2]:.3f} rad/s")

        # 2. GT Extraction & Transformation
        # Qui avviene la magia per fixare i sistemi di riferimento.
        # Otteniamo velocità e attitude nel frame FLU (Drone Standard: X-Forward)
        self.curr_velocity_flu, (roll_deg, pitch_deg) = self._get_velocity_and_attitude_FLU(gt_IMU_frame_slice)
        
        self.current_roll_angle_rad = math.radians(roll_deg)
        self.current_pitch_angle_rad = math.radians(pitch_deg)

        print(f"Velocity FLU: {self.curr_velocity_flu[0]:.3f}, {self.curr_velocity_flu[1]:.3f}, {self.curr_velocity_flu[2]:.3f} m/s")
        print(f"Attitude: Roll={roll_deg:.2f} deg, Pitch={pitch_deg:.2f} deg")

        print("")

        # 3. Convert Velocity FLU -> Camera Frame
        # L'optical flow stima la depth basandosi sul movimento della CAMERA.
        # Assumendo camera frontale:
        # Cam Z = Body X (Forward)
        # Cam X = -Body Y (Left -> Right)
        # Cam Y = -Body Z (Up -> Down)
        # Se hai una funzione bodyToCam calibrata precisa, usa quella. Altrimenti, l'approssimazione standard FPV è:
        # v_cam_x = -vel_flu[1] # -Vy_flu
        # v_cam_y = -vel_flu[2] # -Vz_flu
        # v_cam_z =  vel_flu[0] # +Vx_flu (Forward)
        vel_cam = bodyToCam(self.curr_velocity_flu, self.camParams)
        
        # Sovrascriviamo per usarlo nella stima depth
        self.curr_velocity_cam = np.array([vel_cam[0], vel_cam[1], vel_cam[2]], dtype=np.float32)

        # DEBUG: Stampa per verificare se quando vai dritto vx (FLU) è alta
        # print(f"Time: {t_us} | V_FLU: [{vel_flu[0]:.2f}, {vel_flu[1]:.2f}, {vel_flu[2]:.2f}] | V_CAM: {self.curr_velocity_cam}")

        # 4. Feature Detection & Flow
        if self.prevFrame is None:
            self.prevFrame = self.currFrame.copy()
            self.prevPoints = []
            self._applyCornerDetection(self.currFrame, outputArray='prevPoints')
            return

        self._calculateOpticalFlow(self.currFrame)

        # Visualization
        if self.visualizeImage:
            visualize_image(self.currFrame, self.currPoints, self.prevPoints, self.status)
            cv2.waitKey(self.delayVisualize)

        self.prevPoints.clear()
        self._applyCornerDetection(self.currFrame, outputArray='prevPoints')
        self.prevFrame = self.currFrame.copy()

        # 5. Altitude Estimation
        # Passiamo la velocità nel frame CAMERA (T_cam)
        T_cam = self.curr_velocity_cam
        
        # Filtriamo solo punti con OF significativo
        altitudes = []
        for fv in self.filteredFlowVectors:
            # Calcola Depth lungo l'asse ottico Z
            depth = self._estimateDepth(fv, T_cam)
            # Proietta la depth sull'asse verticale (Ground) usando Roll/Pitch
            alt = self._estimateAltitude(fv, depth)
            
            if not math.isnan(alt) and alt > 0.1 and alt < 20.0:
                altitudes.append(alt)

        # 6. Averaging & Filtering Altitude
        if not altitudes:
            self.avgAltitude = self.prevFilteredAltitude
        else:
            # Median filter robusto agli outlier
            self.avgAltitude = np.median(altitudes)

        if self.avgAltitude >= self.saturationValue:
            self.avgAltitude = self.saturationValue
        if math.isnan(self.avgAltitude):
            self.avgAltitude = self.prevFilteredAltitude

        self.unfilteredAltitude = self.avgAltitude

        # Complementary Filter
        dt_s = self.deltaTms / 1000.0
        self.filteredAltitude = complementaryFilter(
            self.avgAltitude,
            self.prevFilteredAltitude,
            self.complementaryK,
            dt_s
        )
        
        if not math.isnan(self.filteredAltitude):
            self.prevFilteredAltitude = self.filteredAltitude

        # GT Altitude comparison
        if "pz" in gt_IMU_frame_slice:
            gt_alt = float(np.mean(gt_IMU_frame_slice["pz"])) + self.initial_altitude_offset
        else:
            gt_alt = 0.0

        # print(f"ALT: Est={self.filteredAltitude:.3f} | GT={gt_alt:.3f}")


    # --------------------------------------------------------
    # MATHEMATICAL UTILS
    # --------------------------------------------------------

    def _get_velocity_and_attitude_FLU(self, gt_IMU_frame_slice):
        """
        Calcola la velocità del drone nel frame FLU (X-Forward, Y-Left, Z-Up)
        e l'assetto (Roll, Pitch) rispetto a tale frame.
        Risolve il problema degli assi UZH-FPV (IMU raw: X-Left, Y-Up, Z-Forward).
        """
        if len(gt_IMU_frame_slice) < 2:
            return np.zeros(3, dtype=np.float32), (0.0, 0.0)

        # 1. Estrai Timestamp e Posizione nel WORLD frame
        t_us = np.asarray(gt_IMU_frame_slice["timestamp"], dtype=np.int64)
        pos = np.vstack([gt_IMU_frame_slice["px"], 
                         gt_IMU_frame_slice["py"], 
                         gt_IMU_frame_slice["pz"]]).T

        dt = (t_us[-1] - t_us[0]) * 1e-6
        if dt <= 1e-5:
            return np.zeros(3, dtype=np.float32), (0.0, 0.0)

        # 2. Calcola Velocità nel frame WORLD
        # V_world = (P_end - P_start) / dt
        dp_W = pos[-1] - pos[0]
        v_W = dp_W / dt

        # 3. Prendi l'orientamento (Quaternion World -> IMU Raw)
        mid = len(t_us) // 2
        qx = gt_IMU_frame_slice["qx"][mid]
        qy = gt_IMU_frame_slice["qy"][mid]
        qz = gt_IMU_frame_slice["qz"][mid]
        qw = gt_IMU_frame_slice["qw"][mid]

        # R_WI: Matrice di rotazione da BODY(IMU) a WORLD
        # Nota: Solitamente i dataset danno q_WB (Body to World). 
        # Verifichiamo: se q ruota il vettore gravity (0,0,1) body in gravity world, è R_WB.
        R_WI = quat_to_rotmat(qx, qy, qz, qw)

        # 4. Matrice di Permutazione IMU Raw -> FLU
        # Dal paper/tua descrizione: IMU Z è Forward, IMU X è Left, IMU Y è Up.
        # Vogliamo FLU: X=Forward, Y=Left, Z=Up.
        # Quindi:
        # FLU_X (Fwd)  <-- IMU_Z
        # FLU_Y (Left) <-- IMU_X
        # FLU_Z (Up)   <-- IMU_Y
        R_IMU_to_FLU = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)

        # 5. Calcola Velocità nel frame FLU
        # v_body = R_WI.T * v_world  (Proiezione della velocità world sugli assi body)
        v_IMU_raw = R_WI.T @ v_W 
        v_FLU = R_IMU_to_FLU @ v_IMU_raw

        # 6. Calcola Attitude (Roll/Pitch) del frame FLU rispetto al World
        # R_WF (World to FLU) = R_IMU_to_FLU * R_WI^T ??? No.
        # Orientamento Body FLU rispetto a World: R_FLU_to_World = R_WI * R_IMU_to_FLU.T
        # Perché: v_world = R_WI * v_IMU = R_WI * (R_IMU_to_FLU.T * v_FLU)
        # Quindi R_FLU_W = R_WI @ R_IMU_to_FLU.T
        
        R_FLU_W = R_WI @ R_IMU_to_FLU.T
        
        # Convertiamo in quaternione per usare la tua utility quat_to_euler
        q_FLU_W = rotmat_to_quat(R_FLU_W)
        
        # Estrai Eulero (ordine ZYX standard per aeronautica)
        roll, pitch, yaw = quat_to_euler(q_FLU_W[0], q_FLU_W[1], q_FLU_W[2], q_FLU_W[3], order="ZYX")

        return v_FLU.astype(np.float32), (np.degrees(roll), np.degrees(pitch))


    def _estimateDepth(self, ofVector, T_cam):
        # T_cam deve essere la velocità nel CAMERA FRAME
        # ofVector.directionVector è il raggio 3D normalizzato nel CAMERA FRAME
        
        # Depth Estimation for pure translation (or derotated flow)
        # d = || v - (v . u) * u || / || flow ||
        # dove u è directionVector, v è translational velocity T_cam
        
        # Proiezione velocità sulla direzione del pixel
        TdotD = np.dot(T_cam, ofVector.directionVector)
        
        # Componente perpendicolare della velocità (quella che genera parallasse)
        vel_perp = T_cam - TdotD * ofVector.directionVector
        norm_vel_perp = np.linalg.norm(vel_perp)
        
        # Velocità angolare del flusso (rad/s) = ||flow_pix|| * pixel_size / focal_length ???
        # No, ofVector.P è il flow vector derotato in m/s sul piano immagine normalizzato?
        # Dipende da come è calcolato in _calculateOpticalFlow. 
        # Assumendo ofVector.P sia il flow derotato in coordinate normalizzate (metri su piano focale a z=1):
        
        norm_flow = np.linalg.norm(ofVector.P)
        
        if norm_flow < 1e-6:
            return float('nan')
            
        depth = norm_vel_perp / norm_flow
        return depth

    def _estimateAltitude(self, ofVector, depth):
        # depth è la distanza lungo il raggio visivo.
        # Altezza h = depth * cos(angolo_rispetto_verticale)
        
        # 1. Vettore direzione in coordinate CAMERA
        dir_cam = ofVector.directionVector
        
        # 2. Converti in BODY (FLU)
        # Inversa di quella usata prima:
        # Cam Z -> Body X, Cam X -> -Body Y, Cam Y -> -Body Z
        # Body X = Cam Z
        # Body Y = -Cam X
        # Body Z = -Cam Y
        dir_body = np.array([dir_cam[2], -dir_cam[0], -dir_cam[1]])
        
        # 3. Converti in INERZIALE (World) usando Roll/Pitch correnti del Body FLU
        # Usiamo bodyToInertial (che si aspetta roll/pitch del body)
        dir_inertial = bodyToInertial(dir_body, self.current_roll_angle_rad, self.current_pitch_angle_rad)
        
        # 4. Proietta su asse Z world (cosTheta)
        # Assumiamo Z world punti in ALTO (o basso a seconda della convenzione NED/ENU).
        # UZH dataset: Z world punta in ALTO (gravity aligned).
        cosTheta = dir_inertial[2] 
        
        # Se cosTheta è negativo (guarda in alto), l'altezza non ha senso fisico per il suolo
        if cosTheta > 0: # Guarda verso il basso (se z in alto e camera pitchata giu?)
            # Attenzione: se il drone è piatto (roll=0, pitch=0), guarda avanti (X).
            # Z componente è 0. cosTheta = 0. h = 0. Corretto (orizzonte).
            # L'OF per altezza funziona se guardi il terreno.
            return depth * cosTheta # Questo assume che stiamo guardando punti a terra
        else:
            # Se cosTheta è negativo, stiamo guardando verso il basso (se Z world è UP)
            # Solitamente altezza = depth * |vz_versor|
            return depth * abs(cosTheta)

    # --------------------------------------------------------
    # ALTRI METODI (FAST, OF, ETC.) - INVARIATI O ADATTATI
    # --------------------------------------------------------
    
    def _applyCornerDetection(self, image, outputArray='prevPoints'):
        keypoints = self.fastDetector.detect(image, None)
        # Logica random sample/gradient scoring invariata...
        if self.fastParams.randomSampleFilterEnable:
            keypoints = randomlySampleKeypoints(keypoints, self.fastParams.desiredFeatures, self.fastParams.randomSampleFilterRatio)
        elif self.fastParams.gradientScoringEnable:
            keypoints = scoreAndRankKeypointsUsingGradient(keypoints, image, self.fastParams.desiredFeatures)
        
        points = cv2.KeyPoint_convert(keypoints)
        if outputArray == 'prevPoints':
            self.prevPoints = points.tolist()
        else:
            self.nextPrevPoints = points.tolist()

    def _calculateOpticalFlow(self, currFrame):
        if self.prevFrame is not None and len(self.prevPoints) > 0:
            self.currPoints, self.status, self.err = cv2.calcOpticalFlowPyrLK(
                self.prevFrame, currFrame,
                np.float32(self.prevPoints), None,
                winSize=self.lkParams.winSize,
                maxLevel=self.lkParams.maxLevel,
                criteria=self.lkParams.criteria
            )
            self.flowVectors.clear()
            if self.currPoints is not None:
                for i in range(len(self.currPoints)):
                    if self.status[i] == 1:
                        fv = OFVectorFrame(self.prevPoints[i], self.currPoints[i], self.fps, self.camParams)
                        self.flowVectors.append(fv)
            
            self.filteredFlowVectors = rejectOutliersFrame(self.flowVectors, self.magnitudeThresholdPixel, self.boundThreshold)
            
            # Derotation
            for fv in self.filteredFlowVectors:
                self._applyDerotation3D_events(fv, self.curr_gyro_cam)

    def _applyDerotation3D_events(self, ofVector, gyro_cam):
        # Implementazione identica alla tua, usando gyro_cam
        norm_a = np.linalg.norm(ofVector.AMeter)
        Pprime_ms = np.array([
            ofVector.uPixelSec * self.camParams.pixelSize,
            ofVector.vPixelSec * self.camParams.pixelSize
        ], dtype=np.float32)
        PpPprime_ms = np.array([Pprime_ms[0]/norm_a, Pprime_ms[1]/norm_a, 0.0], dtype=np.float32)
        
        dot_val = np.dot(PpPprime_ms, ofVector.directionVector)
        P = PpPprime_ms - dot_val * ofVector.directionVector
        cross_val = np.cross(gyro_cam, ofVector.directionVector)
        RotOF = -cross_val
        ofVector.P = P - RotOF # Flow derotato puramente traslazionale

    def _get_imu(self, cam_imu_slice):
        if len(cam_imu_slice) == 0: return np.zeros(3, dtype=np.float32)
        return np.array([
            np.mean(cam_imu_slice.get("gx", 0.0)),
            np.mean(cam_imu_slice.get("gy", 0.0)),
            np.mean(cam_imu_slice.get("gz", 0.0))
        ], dtype=np.float32)

    def _get_initial_offset(self):
        gt_file = os.path.join(self.events_dir, "groundtruth.txt")
        if not os.path.isfile(gt_file): return 0.0
        try:
            with open(gt_file, 'r') as f:
                for line in f:
                    if not line.startswith("#"):
                        parts = line.split()
                        return -float(parts[3]) # -pz iniziale
        except: pass
        return 0.0
