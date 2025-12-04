import os
import time
import math
import glob
import h5py
import yaml
import numpy as np
import cv2
import csv
import argparse
import matplotlib.pyplot as plt

# ============================================================
# IMPORT: adatta questi import ai tuoi file reali
# ============================================================

from testing.utils.load_utils_fpv import fpv_evs_iterator
from testing.utils.viz_utils import visualize_image, visualize_filtered_flow

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
    wrap_angle_rad
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

    def __init__(self, yaml_path: str, run_id: str, out_dir: str = "results_fpv"):
        self.yaml_path = yaml_path
        self.run_id = run_id
        self.results_dir = os.path.join(out_dir, self.run_id)

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
        self.raw_altitude = 0.0

        # Data Containers (Potenziati per il Logging)
        self.log_data = {
            'timestamp': [],
            'frame_id': [],
            'dt_ms': [],
            # IMU (Gyro Camera Frame)
            'gx_cam': [],
            'gy_cam': [],
            'gz_cam': [],
            # Altitude
            'gt_alt': [],
            'raw_alt': [],
            'filtered_alt': [],
            # Flow Componenti Pixel/s
            'u_pix_raw_mean': [],
            'v_pix_raw_mean': [],
            'u_pix_derot_mean': [],
            'v_pix_derot_mean': [],
            # Flow Componenti Delta Pixel
            'dx_raw_mean': [],
            'dy_raw_mean': [],
            'dx_derot_mean': [],
            'dy_derot_mean': [],
        }

        # Variabili di stato per il calcolo dell'errore
        self.total_alt_error = 0.0
        self.total_rel_error = 0.0
        self.total_frames_with_alt = 0
        self.initial_altitude_offset = 0.0 # Rimosso l'inizializzazione statica, è gestita in _get_initial_offset()
        
        
        # Altitude Filters
        self.complementaryK = 0.9
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

        if "forward" in self.events_dir.lower():
            self.camParams.inclination = 0.0
            print(f"Setting camera inclination to {self.camParams.inclination}° for 'forward' dataset.")
        elif "45" in self.events_dir.lower():
            self.camParams.inclination = 45.0
            print(f"Setting camera inclination to {self.camParams.inclination}° for '45' dataset.")


        if self.rectify: 
            self._updateCameraParameters()

        self.initial_altitude_offset = self._get_initial_offset()
        print(f"Initial altitude offset from GT: {self.initial_altitude_offset:.3f} m")

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

        # print("Loaded configuration:")
        # print(yaml.dump(self.config, default_flow_style=False))

        #load complementary filter quantities
        # SMOOTHINGFILTER:
        # enable: true
        # type: 0   # 0 : COMPLEMENTARY FILTER, 1 : LOW PASS FILTER
        # lpfK: 0.7  #coefficient for the low pass filter. 0.5 is the default value
        # complementaryK: 2.3 #coefficient for the complementary filter.
        smoothing_cfg = self.config.get("SMOOTHINGFILTER", {})
        self.complementaryK = smoothing_cfg.get("complementaryK", 2.3)


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

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------
    def run(self):
        if self.slicing_type in ["fixed", "adaptive"]:
            self._run_slicing()

            # self.plot_flow_components()

        else:
            raise ValueError(f"Unknown slicing type: {self.slicing_type}")

    def _run_slicing(self):
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
        
        # Calcola dt_ms (tempo tra frame attuali e precedente per il log)
        dt_ms = (t_us - self.prev_t_us) / 1000.0 if self.frameID > 0 else self.deltaTms
        self.prev_t_us = t_us

        # 1. IMU (Gyro) extraction
        self.curr_gyro_cam = self._get_imu(cam_imu_slice)

        print(f"Gyro cam : {math.degrees(self.curr_gyro_cam[0]):.3f}, {math.degrees(self.curr_gyro_cam[1]):.3f}, {math.degrees(self.curr_gyro_cam[2]):.3f} deg/s")

        # 2. GT Extraction & Transformation
        # Qui avviene la magia per fixare i sistemi di riferimento.
        # Otteniamo velocità e attitude nel frame FLU (Drone Standard: X-Forward)
        self.curr_velocity_flu, (roll_deg, pitch_deg) = self._get_velocity_and_attitude_FLU(gt_IMU_frame_slice)
        
        raw_roll = math.radians(roll_deg)
        raw_pitch = math.radians(pitch_deg)
            

        if self.frameID == 1:
            # Al primo frame, salviamo "come è girato il mondo rispetto a noi"
            self.initial_roll_offset = raw_roll
            self.initial_pitch_offset = raw_pitch
            print(f"[AHRS] Initializing Attitude. Raw Roll: {roll_deg:.1f}°, Raw Pitch: {pitch_deg:.1f}°")
            print(f"[AHRS] Applying offset correction. New Roll/Pitch should be ~0.")

            # 3. Applica la correzione (sottrai l'offset e normalizza)
            # Usa getattr per sicurezza nel caso frameID != 0 ma offset non settato (caso raro)
        roll_offset = getattr(self, 'initial_roll_offset', 0.0)
        pitch_offset = getattr(self, 'initial_pitch_offset', 0.0)

        self.current_roll_angle_rad = wrap_angle_rad(raw_roll - roll_offset)
        self.current_pitch_angle_rad = wrap_angle_rad(raw_pitch - pitch_offset)

        # print(f"Velocity FLU: {self.curr_velocity_flu[0]:.3f}, {self.curr_velocity_flu[1]:.3f}, {self.curr_velocity_flu[2]:.3f} m/s")
        # print(f"Attitude: Roll={roll_deg:.2f} deg, Pitch={pitch_deg:.2f} deg")

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
        
        # Sovrascriviamo per usarlo nella stima depth
        self.curr_velocity_cam = bodyToCam(self.curr_velocity_flu, self.camParams)
        print(f"Velocity Cam: {self.curr_velocity_cam[0]:.3f}, {self.curr_velocity_cam[1]:.3f}, {self.curr_velocity_cam[2]:.3f} m/s")

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
            visualize_filtered_flow(self.currFrame, self.filteredFlowVectors)
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
            self.raw_altitude = self.prevFilteredAltitude
        else:
            # Median filter robusto agli outlier
            self.raw_altitude = np.median(altitudes)

        if self.raw_altitude >= self.saturationValue:
            self.raw_altitude = self.saturationValue
        if math.isnan(self.raw_altitude):
            self.raw_altitude = self.prevFilteredAltitude


        # Complementary Filter
        dt_s = self.deltaTms / 1000.0
        self.filteredAltitude = complementaryFilter(
            self.raw_altitude,
            self.prevFilteredAltitude,
            self.complementaryK,
            dt_s
        )
        
        if not math.isnan(self.filteredAltitude):
            self.prevFilteredAltitude = self.filteredAltitude

        # GT Altitude comparison
        if "pz" in gt_IMU_frame_slice:
            gt_alt = float(np.mean(gt_IMU_frame_slice["pz"])) - self.initial_altitude_offset
        else:
            gt_alt = 0.0

        # 7. Logging Data
        self.append_log_data(t_us, dt_ms, gt_alt)


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
        
        # --- 1. Calcolo di Pprime_ms e Norm_a (Come in C++) ---
        
        # Norm_a = norma di AMeter (vettore di profondità stimata/inversa, se presente)
        # Assunzione: ofVector.AMeter è un array/list di 3 elementi
        norm_a = np.linalg.norm(ofVector.AMeter)
        
        # Pprime_ms: Flusso RAW scalato da pixel/s a un'unità equivalente a metro/secondo
        # Pprime_ms[0] = uPixelSec * pixelSize
        # Pprime_ms[1] = vPixelSec * pixelSize
        
        # Nota: Usiamo i campi RAW originali (non uNormSec_raw)
        Pprime_ms = np.array([
            ofVector.uPixelSec * ofVector.camParams.pixelSize, 
            ofVector.vPixelSec * ofVector.camParams.pixelSize
        ], dtype=np.float32)
        
        # PpPprime_ms: Pprime_ms normalizzato dalla norma del vettore A (3D)
        if norm_a < 1e-6:
             # Evita divisione per zero. Se norm_a è 0, i calcoli successivi sono invalidi.
             PpPprime_ms = np.array([0.0, 0.0, 0.0])
        else:
            PpPprime_ms = np.array([
                Pprime_ms[0] / norm_a, 
                Pprime_ms[1] / norm_a, 
                0.0
            ], dtype=np.float32)
        
        # --- 2. Step 1 C++: Proiezione Perpendicolare (P) ---
        # P = PpPprime_ms - (PpPprime_ms . directionVector) * directionVector
        
        # ofVector.directionVector è il raggio ottico 3D (unitario) [x/d, y/d, 1/d]
        dot_product = np.dot(PpPprime_ms, ofVector.directionVector)
        
        # Vettore P (non ancora derotato)
        P = PpPprime_ms - (dot_product * ofVector.directionVector)
        
        
        # --- 3. Step 2 C++: Componente Rotazionale e Sottrazione ---
        
        # RotOF C++: -avgGyroRadSec.cross(ofVector.directionVector)
        # RotOF Python: -np.cross(gyro_cam, ofVector.directionVector)
        # (Ricorda, il Python np.cross(A, B) è come il C++ A.cross(B))
        RotOF_3D = -np.cross(gyro_cam, ofVector.directionVector)
        
        # Flusso Derotato Finale (ofVector.P)
        # ofVector.P = P - RotOF
        u_derotata_3D = P - RotOF_3D
        
        # Forza Z=0 per il flusso 2D sul piano focale (NOTA: questo passo NON è in C++,
        # ma è necessario se ofVector.P viene trattato come flow 2D/3D normalizzato altrove)
        # Lo omettiamo per la fedeltà al C++, assumendo che P verrà usato come un vettore 3D.
        # u_derotata_3D[2] = 0.0 
        
        # Assegna il risultato (Flow 3D Normalizzato/Scalato)
        ofVector.P = u_derotata_3D
        
        
        # --- 4. Riconversione per Logging (Logica Invariata) ---

        # Componenti derotate normalizzate
        # Usiamo solo X e Y dal vettore 3D P (Assunzione: queste sono le componenti 2D del flow)
        derot_u_norm_sec = ofVector.P[0]
        derot_v_norm_sec = ofVector.P[1]

        # Invertiamo il calcolo per Pixel/s (scalando indietro)
        # NOTA: La logica C++ usa Pprime_ms scalato con pixelSize, ma il Python logging vuole Pixel/s.
        # Dobbiamo trovare il fattore di scaling inverso.

        # Se PpPprime_ms[0] = uPixelSec * pixelSize / norm_a
        # Allora uPixelSec = PpPprime_ms[0] * norm_a / pixelSize
        
        if norm_a < 1e-6 or ofVector.camParams.pixelSize < 1e-9:
             # Caso di emergenza per evitare NaN nel log
             ofVector.uPixelSec_derot = 0.0
             ofVector.vPixelSec_derot = 0.0
        else:
            # Calcolo inverso delle unità di flow in pixel/s
            u_derot_raw_units = (derot_u_norm_sec * norm_a) / ofVector.camParams.pixelSize
            v_derot_raw_units = (derot_v_norm_sec * norm_a) / ofVector.camParams.pixelSize
            
            ofVector.uPixelSec_derot = u_derot_raw_units
            ofVector.vPixelSec_derot = v_derot_raw_units

        # Delta X/Y in pixel (rispetto a dt = 1/fps)
        if ofVector.fps > 0:
            ofVector.deltaX_derot = ofVector.uPixelSec_derot / ofVector.fps
            ofVector.deltaY_derot = ofVector.vPixelSec_derot / ofVector.fps
            ofVector.magnitudePixel_derot = math.sqrt(
                ofVector.deltaX_derot**2 + ofVector.deltaY_derot**2
            )
        else:
            ofVector.magnitudePixel_derot = 0.0
            
        # 5. Salva le Magnitudini per l'analisi dei rapporti
        
        # Flusso RAW in coordinate Normalizzate Originali
        raw_norm_3D = np.array([ofVector.uNormSec_raw, ofVector.vNormSec_raw, 0.0])
        
        ofVector.magnitude_raw = np.linalg.norm(raw_norm_3D[:2])
        ofVector.magnitude_rotational = np.linalg.norm(RotOF_3D[:2])
        
        # Per la derotated magnitude, usiamo la norma 2D del risultato C++ (non le unità del log)
        # Usiamo il flow derotato in coordinate normalizzate (rad/s)
        ofVector.magnitude_derotated = np.linalg.norm(u_derotata_3D[:2])

    def _get_imu(self, cam_imu_slice):
        """
        cam imu is in reality defined in the D frame in the UZH-FPV paper.
        so it is the imu on the camera, however the reference system is X-left, Y-up, Z-forward
        return IMU on the proper CAMERA FRAME : X-right, Y-down, Z-forward
        """

        R_IC = np.array([
            [-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0,  1]
        ])

        if len(cam_imu_slice) == 0: return np.zeros(3, dtype=np.float32)
        
        imu_raw = np.array([
            np.mean(cam_imu_slice.get("gx", 0.0)),
            np.mean(cam_imu_slice.get("gy", 0.0)),
            np.mean(cam_imu_slice.get("gz", 0.0))
        ], dtype=np.float32)

        return R_IC @ imu_raw

    def _get_initial_offset(self):
        gt_file = os.path.join(self.events_dir, "groundtruth.txt")
        print(f"Looking for GT file at: {gt_file}")

        if not os.path.isfile(gt_file):
            print("GT file not found.")
            return 0.0

        try:
            data = np.loadtxt(gt_file, comments="#", delimiter=" ", skiprows=1)
            # data shape -> N x 8   (ts, px, py, pz, qx, qy, qz, qw)

            return float(data[0, 3])  # 4th column = pz
        except Exception as e:
            print(f"loadtxt failed: {e}")
            return 0.0


    def append_log_data(self, t_us, dt_ms, gt_alt):
        if self.frameID > 0:
            # Flusso (le medie sono già state calcolate nella sezione precedente)
            u_pix_raw_mean = np.median([fv.uPixelSec for fv in self.filteredFlowVectors]) if self.filteredFlowVectors else 0.0
            v_pix_raw_mean = np.median([fv.vPixelSec for fv in self.filteredFlowVectors]) if self.filteredFlowVectors else 0.0
            u_pix_derot_mean = np.median([fv.uPixelSec_derot for fv in self.filteredFlowVectors]) if self.filteredFlowVectors else 0.0
            v_pix_derot_mean = np.median([fv.vPixelSec_derot for fv in self.filteredFlowVectors]) if self.filteredFlowVectors else 0.0
            dx_raw_mean = np.median([fv.deltaX for fv in self.filteredFlowVectors]) if self.filteredFlowVectors else 0.0
            dy_raw_mean = np.median([fv.deltaY for fv in self.filteredFlowVectors]) if self.filteredFlowVectors else 0.0
            dx_derot_mean = np.median([fv.deltaX_derot for fv in self.filteredFlowVectors]) if self.filteredFlowVectors else 0.0
            dy_derot_mean = np.median([fv.deltaY_derot for fv in self.filteredFlowVectors]) if self.filteredFlowVectors else 0.0

            self.log_data['timestamp'].append(t_us)
            self.log_data['frame_id'].append(self.frameID)
            self.log_data['dt_ms'].append(dt_ms)
            
            # IMU
            self.log_data['gx_cam'].append(self.curr_gyro_cam[0])
            self.log_data['gy_cam'].append(self.curr_gyro_cam[1])
            self.log_data['gz_cam'].append(self.curr_gyro_cam[2])
            
            # Altitudine
            self.log_data['gt_alt'].append(gt_alt)
            self.log_data['raw_alt'].append(self.raw_altitude)
            self.log_data['filtered_alt'].append(self.filteredAltitude)

            # Flow
            self.log_data['u_pix_raw_mean'].append(u_pix_raw_mean)
            self.log_data['v_pix_raw_mean'].append(v_pix_raw_mean)
            self.log_data['u_pix_derot_mean'].append(u_pix_derot_mean)
            self.log_data['v_pix_derot_mean'].append(v_pix_derot_mean)
            self.log_data['dx_raw_mean'].append(dx_raw_mean)
            self.log_data['dy_raw_mean'].append(dy_raw_mean)
            self.log_data['dx_derot_mean'].append(dx_derot_mean)
            self.log_data['dy_derot_mean'].append(dy_derot_mean)

            # Calcolo degli Errori per il Summary
            if gt_alt > 0.1: # Evita la divisione per zero o altitudini non significative
                abs_error = abs(self.filteredAltitude - gt_alt)
                rel_error = abs_error / gt_alt
                
                self.total_alt_error += abs_error
                self.total_rel_error += rel_error
                self.total_frames_with_alt += 1

        print(f"ALT: Raw Altitude = {self.raw_altitude:.3f} m | Est={self.filteredAltitude:.3f} m | GT={gt_alt:.3f} m")
        self.frameID += 1

    def log(self):
        """Crea la directory dei risultati, salva i dati, i plot e il summary."""
        print(f"\n--- Inizio Logging per Run: {self.run_id} ---")
        
        # 1. Creazione della directory di destinazione
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Directory risultati creata: {self.results_dir}")

        # 2. Salvataggio del file YAML di configurazione
        config_output_path = os.path.join(self.results_dir, f"{self.run_id}_config.yaml")
        with open(self.yaml_path, 'r') as f_in, open(config_output_path, 'w') as f_out:
            f_out.write(f_in.read())
        print(f"Configurazione YAML copiata in: {config_output_path}")

        # 3. Salvataggio del CSV dei dati per frame
        self._save_csv()

        # 4. Generazione dei Plot
        self._plot_flow_and_imu()
        self._plot_altitude()

        # 5. Salvataggio del Log di Riepilogo
        self._save_summary_log()

        print("--- Logging Completato! ---")

    def _save_csv(self):
        """Salva tutti i dati raccolti nel CSV."""
        csv_path = os.path.join(self.results_dir, f"{self.run_id}_data.csv")
        
        # Controlla se ci sono dati
        if not self.log_data['frame_id']:
            print("[WARNING] Nessun dato da salvare nel CSV.")
            return

        # Scrivi intestazioni
        headers = list(self.log_data.keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            # Scrivi righe (trasponi la struttura del dizionario)
            rows = zip(*[self.log_data[key] for key in headers])
            writer.writerows(rows)
            
        print(f"Dati per frame salvati in: {csv_path}")

    def _save_summary_log(self):
        """Calcola e salva gli errori medi e un riepilogo nel file .log."""
        log_path = os.path.join(self.results_dir, f"results.log")
        
        mean_alt_error = 0.0
        mean_rel_error = 0.0
        
        if self.total_frames_with_alt > 0:
            mean_alt_error = self.total_alt_error / self.total_frames_with_alt
            mean_rel_error = self.total_rel_error / self.total_frames_with_alt
            
        with open(log_path, 'w') as f:
            f.write(f"--- Riepilogo Esecuzione: {self.run_id} ---\n")
            f.write(f"Timestamp Esecuzione: {time.ctime()}\n")
            f.write(f"File di Configurazione: {self.yaml_path}\n")
            f.write(f"Totale Frame Processati (con Altitudine > 0.1m): {self.total_frames_with_alt}\n\n")
            f.write("--- Prestazioni Stima Altitudine ---\n")
            f.write(f"Errore Assoluto Medio (MAE) sull'Altitudine: {mean_alt_error:.4f} m\n")
            f.write(f"Errore Relativo Medio (MRE) sull'Altitudine: {mean_rel_error * 100:.2f} %\n")
            f.write("-------------------------------------\n")
            f.write("\nConfigurazione:\n")
            # Aggiungi un piccolo riepilogo della configurazione
            f.write(f"Slicing: {self.slicing_type} (dt_ms: {self.fixed_dt_ms})\n")
            f.write(f"Filtro Smoothing: {'Complementary' if self.smoothingFilterType == 0 else 'LPF'} (K: {self.complementaryK if self.smoothingFilterType == 0 else self.lpfK})\n")
            
        print(f"Log di riepilogo salvato in: {log_path}")

    def _plot_flow_and_imu(self):
        """Genera il plot del Flusso Ottico (RAW vs DEROTATED) e Gyro."""
        
        if not self.log_data['frame_id']: return

        frames = self.log_data['frame_id']
        
        # Dati IMU
        gx = np.array(self.log_data['gx_cam'])
        gy = np.array(self.log_data['gy_cam'])
        gz = np.array(self.log_data['gz_cam'])

        # Dati Pixel/s
        u_raw = np.array(self.log_data['u_pix_raw_mean'])
        v_raw = np.array(self.log_data['v_pix_raw_mean'])
        u_derot = np.array(self.log_data['u_pix_derot_mean'])
        v_derot = np.array(self.log_data['v_pix_derot_mean'])

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 12))
        
        # --- 1. Flusso U (Pixel/sec) ---
        axes[0].plot(frames, u_raw, label='Flow RAW $\\bar{u}$ (px/s)', alpha=0.7)
        axes[0].plot(frames, u_derot, label='Flow DEROTATED $\\bar{u}$ (px/s)', color='red', linewidth=2)
        axes[0].set_title('Flusso Ottico Orizzontale ($u$) - RAW vs DEROTATED')
        axes[0].set_ylabel('$u$ (px/s)')
        axes[0].grid(True, linestyle=':')
        axes[0].legend()

        # --- 2. Flusso V (Pixel/sec) ---
        axes[1].plot(frames, v_raw, label='Flow RAW $\\bar{v}$ (px/s)', alpha=0.7)
        axes[1].plot(frames, v_derot, label='Flow DEROTATED $\\bar{v}$ (px/s)', color='red', linewidth=2)
        axes[1].set_title('Flusso Ottico Verticale ($v$) - RAW vs DEROTATED')
        axes[1].set_ylabel('$v$ (px/s)')
        axes[1].grid(True, linestyle=':')
        axes[1].legend()

        # --- 3. Velocità Angolare (Gyro) ---
        axes[2].plot(frames, gx, label='Gyro $G_x$ (Roll)', alpha=0.7)
        axes[2].plot(frames, gy, label='Gyro $G_y$ (Pitch)', alpha=0.7)
        axes[2].plot(frames, gz, label='Gyro $G_z$ (Yaw)', alpha=0.7)
        axes[2].set_title('Velocità Angolari (Gyro) nel Frame Telecamera (rad/s)')
        axes[2].set_xlabel('Frame ID')
        axes[2].set_ylabel('Angoli (rad/s)')
        axes[2].grid(True, linestyle=':')
        axes[2].legend()

        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, f"{self.run_id}_flow_imu_comparison.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Plot Flow/IMU salvato in: {plot_path}")

    def _plot_altitude(self):
        """Genera il plot delle stime di altitudine vs Ground Truth."""
        
        if not self.log_data['frame_id']: return

        frames = self.log_data['frame_id']
        gt_alt = np.array(self.log_data['gt_alt'])
        raw_alt = np.array(self.log_data['raw_alt'])
        filtered_alt = np.array(self.log_data['filtered_alt'])

        fig, ax = plt.subplots(figsize=(14, 6))
        
        # --- Altitudine ---
        ax.plot(frames, gt_alt, label='Ground Truth $Z_{GT}$', color='green', linewidth=3, linestyle='--')
        ax.plot(frames, raw_alt, label='Raw Altitude $H_{RAW}$ (Before Filter)', alpha=0.6, linestyle=':')
        ax.plot(frames, filtered_alt, label='Filtered Altitude $H_{FILT}$', color='red', linewidth=2)
        
        ax.set_title('Stima Altitudine vs Ground Truth')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Altitudine (metri)')
        ax.grid(True, linestyle=':')
        ax.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, f"{self.run_id}_altitude_estimation.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Plot Altitudine salvato in: {plot_path}")