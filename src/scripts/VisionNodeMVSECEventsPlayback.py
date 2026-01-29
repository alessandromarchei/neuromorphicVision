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
from testing.utils.load_utils_mvsec import mvsec_evs_iterator, read_rmap, mvsec_evs_iterator_adaptive, VALID_FRAME_RANGES
from testing.utils.event_utils import to_event_frame
from testing.utils.viz_utils import visualize_image, visualize_gt_flow, visualize_filtered_flow
from testing.utils.loss import compute_AEE


# strutture dati e parametri (come nel tuo script a frame)
from src.scripts.defs import (
    FastParams,
    CameraParams,
    LKParams,
    AdaptiveSlicerPID,
    OFVectorFrame
)

# funzioni di utilità per l'OF (uguali al tuo script a frame)
from src.scripts.functions import (
    randomlySampleKeypoints,
    scoreAndRankKeypointsUsingGradient,
    rejectOutliersFrame,
    drawTextWithBackground,
    compute_a_vector_meter,
    compute_direction_vector,
)

# ============================================================
# VisionNodeEventsPlayback
# ============================================================

class VisionNodeEventsPlayback:
    """
    Variante event-based:
    - usa eventi MVSEC
    - genera event frames
    - calcola FAST + LK + filtri
    - opzionalmente usa adaptive slicing PID sui dt_ms
    - carica GT optical flow maps se presente *_gt.hdf5
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

        # ---- Evaluation logs ----
        self.eval_AEE = []      #average endpoint error
        self.eval_REE = []      #relative endpoint error
        self.eval_outliers = []
        self.eval_dt_ms = []
        self.eval_dtgt_ms = []
        self.eval_Npoints = []
        self.eval_frameID = []
        self.eval_timestamp = []

        #list of dict containing OF magnitudes per frame and number of filtered points
        self.of_magnitudes = []


        self.ofTime = 0.0
        self.featureDetectionTime = 0.0

        self.magnitudeThresholdPixel = 10.0
        self.boundThreshold = 1.5

        self.frameID = 0

        # GT flow corrente (se disponibile)
        self.current_gt_flow = None
        self.current_gt_ts_us = None
        self.dt_gt_flow_ms = None
        self.current_flow_gt_id = 0

        # load config
        self._loadParametersFromYAML(yaml_path)

        # event-related config
        events_cfg = self.config["EVENTS"]
        self.events_dir = events_cfg["scene"]
        self.side = events_cfg.get("side", "left")
        self.H = events_cfg.get("H", 260)
        self.W = events_cfg.get("W", 346)
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
        if self.slicing_type == "mvsec":
            self.fps = 1000.0 / 10.0       # placeholder, aggiornato da mvsec iterator
        elif self.slicing_type == "fixed":
            self.fps = 1000.0 / self.fixed_dt_ms
        elif self.slicing_type == "adaptive":
            self.fps = 1000.0 / self.adaptiveSlicer.initial_dt_ms

        self.deltaTms = 1000.0 / self.fps

        self.gt_mode = self.config["SLICING"]["gt_mode"]

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

    def printFinalReport(self):
        if len(self.eval_AEE) == 0:
            print("\n[FINAL REPORT] No valid flow comparisons found.\n")
            return

        AEE = np.array(self.eval_AEE)
        REE = np.array(self.eval_REE)
        OUT = np.array(self.eval_outliers)
        DT  = np.array(self.eval_dt_ms)
        DTGT = np.array(self.eval_dtgt_ms)
        NPTS = np.array(self.eval_Npoints)

        print("\n============================================")
        print("              FINAL EVALUATION REPORT       ")
        print("============================================")
        print(f"Total valid frames: {len(AEE)}")
        print("--------------------------------------------")
        print(f"Mean AEE             : {AEE.mean():.4f}")
        print(f"Median AEE           : {np.median(AEE):.4f}")
        print(f"Mean REE             : {REE.mean():.4f}")
        print(f"Median REE           : {np.median(REE):.4f}")
        print(f"Min AEE              : {AEE.min():.4f}")
        print(f"Max AEE              : {AEE.max():.4f}")
        print("--------------------------------------------")
        print(f"Mean Outliers (%)    : {(OUT.mean()*100):.2f}")
        print(f"Median Outliers (%)  : {(np.median(OUT)*100):.2f}")
        print("--------------------------------------------")
        print(f"Mean Event dt (ms)   : {DT.mean():.3f}")
        print(f"Mean GT dt (ms)      : {DTGT.mean():.3f}")
        print("--------------------------------------------")
        print(f"Mean #points/frame   : {NPTS.mean():.2f}")
        print("============================================\n")


    def cleanNaNEntries(self):
        """Remove all indices where any logged quantity contains NaN."""

        # Stack everything in a dict to generalize
        logs = {
            "AEE": self.eval_AEE,
            "REE": self.eval_REE,
            "OUT": self.eval_outliers,
            "DT": self.eval_dt_ms,
            "DTGT": self.eval_dtgt_ms,
            "NPTS": self.eval_Npoints,
            "FRAME": self.eval_frameID,
            "TS": self.eval_timestamp,
        }

        # Convert all lists → numpy arrays
        logs_np = {k: np.array(v) for k, v in logs.items()}

        # Create mask of valid indices (no nan in ANY array)
        mask = np.ones(len(self.eval_AEE), dtype=bool)

        for k, arr in logs_np.items():
            mask &= ~np.isnan(arr)

        # Apply mask back to all lists
        for k, arr in logs_np.items():
            logs_np[k] = arr[mask]

        # Assign back
        self.eval_AEE = logs_np["AEE"].tolist()
        self.eval_REE = logs_np["REE"].tolist()
        self.eval_outliers = logs_np["OUT"].tolist()
        self.eval_dt_ms = logs_np["DT"].tolist()
        self.eval_dtgt_ms = logs_np["DTGT"].tolist()
        self.eval_Npoints = logs_np["NPTS"].tolist()
        self.eval_frameID = logs_np["FRAME"].tolist()
        self.eval_timestamp = logs_np["TS"].tolist()

        # Debug print
        print(f"[CLEAN] Removed {mask.size - mask.sum()} NaN entries.")

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
        self.rectify_map = read_rmap(rect_file, H=self.H, W=self.W)

        print(f"[VisionNodeEventsPlayback] Loaded {len(self.all_evs)} events, ts range [{self.ts_start}, {self.ts_end}]")

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------
    def run(self):
        if self.slicing_type == "mvsec" or self.slicing_type == "fixed":
            self._run_fixed_slicing()

        elif self.slicing_type == "adaptive":
            self._run_adaptive_slicing()

        else:
            raise ValueError(f"Unknown slicing type: {self.slicing_type}")

        # self.plot_of_magnitudes()

    # --------------------------------------------------------
    # RUN: fixed slicing con mvsec_evs_iterator
    # --------------------------------------------------------
    def _run_fixed_slicing(self):
        print("[VisionNodeEventsPlayback] Running unified event+GT iterator.")

        iterator = mvsec_evs_iterator(
            self.events_dir,
            side=self.side,
            dT_ms=self.fixed_dt_ms,
            H=self.H,
            W=self.W,
            rectify=self.rectify,
            gt_mode=self.gt_mode,
            use_valid_frame_range=self.use_valid_frame_range
        )

        for i, (event_frame, t_us, dt_ms, flow_map, ts_gt, dt_gt_ms, flow_id) in enumerate(iterator):
            self.deltaTms = dt_ms
            self.current_gt_flow = flow_map
            self.current_gt_ts_us = ts_gt
            self.dt_gt_flow_ms = dt_gt_ms
            self.current_flow_gt_id = flow_id

            self._processEventFrame(event_frame, t_us)
            self.frameID += 1

    # --------------------------------------------------------
    # RUN: adaptive slicing con PID
    # --------------------------------------------------------
    def _run_adaptive_slicing(self):
        print("[VisionNode] Running ADAPTIVE slicing with unified iterator")

        iterator = mvsec_evs_iterator_adaptive(
            self.events_dir,
            side=self.side,
            adaptive_slicer=self.adaptiveSlicer,
            H=self.H,
            W=self.W,
            rectify=self.rectify,
            use_valid_frame_range=self.use_valid_frame_range
        )

        for (event_frame,
            t_us,
            dt_ms,
            flow_map,
            ts_gt,
            dt_gt_ms,
            flow_id,
            in_valid_range) in iterator:

            self.deltaTms = dt_ms
            self.current_gt_flow = flow_map
            self.current_gt_ts_us = ts_gt
            self.dt_gt_flow_ms = dt_gt_ms
            self.current_flow_gt_id = flow_id

            # Process frame (LK + FAST)
            self._processEventFrame(event_frame, t_us)

            # update PID based on filtered flow, 
            if in_valid_range:

                if self.adaptiveSlicer_type == "PID":
                    # print(f"[Adaptive PID] inside valid frame range")
                    new_dt, updated = self.adaptiveSlicer.update(self.filteredFlowVectors)
                    if updated:
                        print(f"[Adaptive PID] dt_\ms updated → {new_dt:.2f}")

                elif self.adaptiveSlicer_type == "ABMOF":
                    self.adaptiveSlicer.update_with_flow(self.filteredFlowVectors)
                    updated = self.adaptiveSlicer.feedback()
                    if updated:
                        print(f"[ABMOF] areaEventThr → {self.adaptiveSlicer.areaEventThr}")

            self.frameID += 1


    # --------------------------------------------------------
    # Processa UN event frame (come processFrames(), ma senza altitude)
    # --------------------------------------------------------
    def _processEventFrame(self, event_frame, timestamp_us):
        """
        Per ora: optical flow + feature detection, esattamente come nel caso frames.
        (La GT è già in self.current_gt_flow se disponibile.)
        """
        self.currFrame = event_frame

        # se è il primissimo frame, inizializza solo i punti
        if self.prevFrame is None:
            self.prevFrame = self.currFrame.copy()
            self.prevPoints = []
            self._applyCornerDetection(self.currFrame, outputArray='prevPoints')
            return

        startOF = time.perf_counter()
        self._calculateOpticalFlow(self.currFrame)
        endOF = time.perf_counter()
        self.ofTime = (endOF - startOF) * 1e6

        self.evaluateOpticalFlow()


        #apply visualization eventually
        if self.visualizeImage:
            # visualize_gt_flow(self.current_gt_flow, self.currFrame, win_name="GT Flow", apply_mask=False)
            # visualize_gt_flow(self.flow_prediction_map, self.currFrame, win_name="GT Flow Prediction", apply_mask=False)
            # visualize_image(self.currFrame,self.currPoints,self.prevPoints,self.status)
            visualize_filtered_flow(self.currFrame, self.filteredFlowVectors, win_name="OF_filtered")
            cv2.waitKey(self.delayVisualize)


        self.prevPoints.clear()

        startFD = time.perf_counter()
        self._applyCornerDetection(self.currFrame, outputArray='prevPoints')
        endFD = time.perf_counter()
        self.featureDetectionTime = (endFD - startFD) * 1e6

        self.prevFrame = self.currFrame.copy()

        self.saveOFMagnitudes()

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
        else:
            print("FIRST EVENT FRAME, skipping OF...")

    def evaluateOpticalFlow(self):
        """
        1) convert discrete point wise optical flow into 2D optical flow
        2) apply the AEE metric on the OF map
        """

        #apply a 2D (2,H,W) mask containing the flow per pixel value, in the self.filteredFlowVectors values
        # flow_prediction = sparseTo2Dflow(self)
        
        #create the 2D map with the same shape of flow gt (sparse)
        self.flow_prediction_map = np.zeros_like(self.current_gt_flow)

        for point in self.filteredFlowVectors:
            #insert each feature inside the map
            (x_coord, y_coord) = np.round(point.position).astype(int)

            #check shape
            if self.flow_prediction_map.ndim == 3 and self.flow_prediction_map.shape[2] == 2:
                #assuming the shape is (H, W, 2), so reshape to (2, H, W)
                self.flow_prediction_map = self.flow_prediction_map.transpose(2, 0, 1)

            self.flow_prediction_map[0][y_coord][x_coord] = point.deltaX
            self.flow_prediction_map[1][y_coord][x_coord] = point.deltaY

        # #apply AEE evaluation
        if self.deltaTms is not None and self.dt_gt_flow_ms is not None:

            AEE, outlier_percentage, N_points, REE = compute_AEE(
                estimated_flow=self.flow_prediction_map,
                gt_flow=self.current_gt_flow,
                dt_input_ms=self.deltaTms,
                dt_gt_ms=self.dt_gt_flow_ms
            )

            # ----- LOGGING -----
            self.eval_AEE.append(AEE)
            self.eval_REE.append(REE)
            self.eval_outliers.append(outlier_percentage)
            self.eval_dt_ms.append(self.deltaTms)
            self.eval_dtgt_ms.append(self.dt_gt_flow_ms)
            self.eval_Npoints.append(N_points)
            self.eval_frameID.append(self.frameID)
            self.eval_timestamp.append(self.current_gt_ts_us)


            #print every 100 frames
            if self.frameID % 100 == 0:
                print(f"[EVAL] Frame {self.frameID:05d} | "
                    f"AEE={AEE:.3f}, REE={REE:.3f}, Outliers={outlier_percentage*100.0:.2f}%, "
                    f"dt={self.deltaTms:.2f} ms, gt_dt={self.dt_gt_flow_ms:.2f} ms, "
                    f"N={N_points}, GT FLOW ID={self.current_flow_gt_id}")
    

    def saveOFMagnitudes(self):
        magnitudes = []
        for vector in self.filteredFlowVectors:
            mag = math.sqrt(vector.deltaX ** 2 + vector.deltaY ** 2)
            magnitudes.append(mag)

        if len(magnitudes) > 0:
            avg_magnitude = sum(magnitudes) / len(magnitudes)
        else:
            avg_magnitude = 0.0

        self.of_magnitudes.append({
            "magnitude": avg_magnitude,
            "N_vectors": len(self.filteredFlowVectors),
            "dt_ms": self.deltaTms
        })


    def plot_of_magnitudes(self, filename):
        magnitudes = [entry["magnitude"] for entry in self.of_magnitudes]
        N_vectors = [entry["N_vectors"] for entry in self.of_magnitudes]
        dt_ms = [entry["dt_ms"] for entry in self.of_magnitudes]
        

        # Create figure with 3 rows, custom height ratios
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1])  # last row smaller

        # --- Row 1: Magnitudes ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(magnitudes, label='Optical Flow Magnitude (pixels/frame)')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Magnitude (pixels/frame)')
        ax1.set_title('Optical Flow Magnitude over Time')
        ax1.legend()
        ax1.grid()

        # --- Row 2: dt_ms ---
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(dt_ms, label='Delta Time (ms)', color='green')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Δt (ms)')
        ax2.set_title('Adaptive Slicing Δt (ms)')
        ax2.legend()
        ax2.grid()

        # --- Row 3: Number of vectors (smaller height) ---
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(N_vectors, label='Number of Filtered Vectors', color='orange')
        ax3.set_xlabel('Frame Index')
        ax3.set_ylabel('N vectors')
        ax3.set_title('Filtered OF Vectors Count')
        ax3.legend()
        ax3.grid()

        plt.tight_layout()
        plt.savefig(f"{filename}.png", dpi=250)
        plt.close()

        print(f"[VisionNodeEventsPlayback] Saved OF magnitude plot to {filename}.png")