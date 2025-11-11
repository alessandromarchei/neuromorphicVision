#!/usr/bin/env python3

import os
import cv2
import csv
import math
import time
import yaml  # PyYAML
import numpy as np
import threading

# ------------------------------------------------------------------
# Import data structures + parameters from defs.py
# ------------------------------------------------------------------
from defs import (
    IMUData,
    VelocityData,
    FrameData,
    SensorData,
    FastParams,
    CameraParams,
    LKParams,
    MAX_FPS_FLIR
)

# ------------------------------------------------------------------
# Import utility functions from functions.py
# ------------------------------------------------------------------
from functions import (
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

# ------------------------------------------------------------------
# Create a Python class to replicate VisionNodeFramesPlayback
# ------------------------------------------------------------------

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
        self.AMeter = compute_a_vector_meter(self.position,camParams)
        self.directionVector = compute_direction_vector(self.position,camParams)
        self.P = np.array([0.0, 0.0, 0.0], dtype=np.float32)


class VisionNodeFramesPlayback:
    def __init__(self, yaml_path: str):
        """
        In C++, you used the constructor + loadParameters. 
        Here we read from a YAML config file (like your .yaml).
        """
        self.aviPath = ""
        self.outputFileEnable = True
        self.outputFilePath = "output.csv"

        self.fastParams = FastParams()
        self.camParams = CameraParams()
        self.lkParams = LKParams()

        self.applyDerotation = True
        self.visualizeImage = True
        self.delayVisualize = 1

        self.startFrameID = 0
        self.endFrameID = 0
        self.initialFrameID = 0
        self.frameID = 0

        self.outputFileStream = None
        self.playback = None

        self.currFrame = None
        self.prevFrame = None
        self.prevPoints = []
        self.nextPrevPoints = []

        self.flowVectors = []
        self.filteredFlowVectors = []

        self.fps = 30.0
        self.deltaTms = 1000.0 / self.fps
        self.ofTime = 0.0
        self.featureDetectionTime = 0.0

        self.magnitudeThresholdPixel = 10.0
        self.boundThreshold = 1.5

        self.smoothingFilterEnable = True
        self.smoothingFilterType = 0
        self.complementaryK = 0.9
        self.lpfK = 0.1

        self.altitudeType = 0
        self.saturationValue = 45.0
        self.avgAltitude = 0.0
        self.filteredAltitude = 0.0
        self.unfilteredAltitude = 0.0
        self.prevFilteredAltitude = 0.0

        self.detectedFeatures = 0
        self.filteredDetectedFeatures = 0
        self.safeFeaturesApplied = False
        self.rejectedVectors = 0

        self.currentIMUData = IMUData()
        self.currentVelocityData = VelocityData()
        self.currentSensorData = SensorData()
        self.imuData = []
        self.velocityData = []
        self.frameData = []
        self.currentTimestamp = 0
        self.prevTimestamp = 0
        self.endTimestamp = 0

        self.FASTLKParallel = False
        self.FASTThread = None

        # For 3D rotation
        self.avgGyro_rad_cam = np.array([0.0,0.0,0.0], dtype=np.float32)

        # For orientation
        self.cosRoll = 1.0
        self.sinRoll = 0.0
        self.cosPitch = 1.0
        self.sinPitch = 0.0

        # Load from YAML
        self.loadParametersFromYAML(yaml_path)

        # Initialize
        self.initializePlayback()
        self.initializeOutputFile()
        self.initializeSensorData()
        self.initializeFeatureDetector()

    def loadParametersFromYAML(self, yaml_path):
        # very similar logic to your code:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # e.g. playback
        if "PLAYBACK" in config and "filepath" in config["PLAYBACK"]:
            self.aviPath = config["PLAYBACK"]["filepath"]

        # output
        if "outputFileEnable" in config:
            self.outputFileEnable = bool(config["outputFileEnable"])
        if "outputFilePath" in config:
            self.outputFilePath = config["outputFilePath"]

        # FAST
        if "FAST" in config:
            self.fastParams.threshold = config["FAST"].get("threshold", 10)
            self.fastParams.nonmaxSuppression = config["FAST"].get("nonmaxSuppression", True)
            rsf = config["FAST"].get("randomSampleFilter", {})
            self.fastParams.randomSampleFilterEnable = rsf.get("enable", False)
            self.fastParams.randomSampleFilterRatio = rsf.get("ratio", 0.0)
            gs = config["FAST"].get("gradientScoring", {})
            self.fastParams.gradientScoringEnable = gs.get("enable", False)
            self.fastParams.desiredFeatures = gs.get("desiredFeatures", 200)
            self.fastParams.safeFeatures = config["FAST"].get("safeFeatures", False)

        # CAMERA
        if "CAMERA" in config:
            cam_conf = config["CAMERA"]
            # resolution
            if "resolution" in cam_conf:
                w = cam_conf["resolution"].get("width", 640)
                h = cam_conf["resolution"].get("height", 480)
                self.camParams.resolution = (w, h)
            # downSampling
            if "downSampling" in cam_conf:
                self.camParams.binningEnable = cam_conf["downSampling"].get("enable", False)
                self.camParams.binning_x = cam_conf["downSampling"].get("width", 640)
                self.camParams.binning_y = cam_conf["downSampling"].get("height", 480)
            self.camParams.exposureTime = cam_conf.get("exposureTime", 0)
            self.camParams.fx = cam_conf.get("fx", 1786.89)
            self.camParams.fy = cam_conf.get("fy", 1785.46)
            self.camParams.cx = cam_conf.get("cx", 681.47)
            self.camParams.cy = cam_conf.get("cy", 522.97)
            self.camParams.pixelSize = cam_conf.get("pixelSize", 3.45e-6)
            self.camParams.inclination = cam_conf.get("inclination", 45.0)

        # LK
        if "LK" in config:
            lk_conf = config["LK"]
            ws = lk_conf.get("winSize", {})
            self.lkParams.winSize = (ws.get("width",21), ws.get("height",21))
            self.lkParams.maxLevel = lk_conf.get("maxLevel",5)
            crit = lk_conf.get("criteria",{})
            maxCount = crit.get("maxCount",40)
            epsilon = crit.get("epsilon",0.01)
            self.lkParams.flags = lk_conf.get("flags",0)
            self.lkParams.minEigThreshold = lk_conf.get("minEigThreshold",0.001)
            import cv2
            self.lkParams.criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                maxCount,
                epsilon
            )

        if "FASTLKParallel" in config:
            self.FASTLKParallel = bool(config["FASTLKParallel"])

        if "REJECTION_FILTER" in config:
            rf = config["REJECTION_FILTER"]
            self.magnitudeThresholdPixel = rf.get("magnitudeThresholdPixel", 75)
            self.boundThreshold = rf.get("boundThreshold", 1.5)

        if "SMOOTHINGFILTER" in config:
            sf = config["SMOOTHINGFILTER"]
            self.smoothingFilterEnable = sf.get("enable", True)
            self.smoothingFilterType = sf.get("type", 0)
            self.lpfK = sf.get("lpfK", 0.0)
            self.complementaryK = sf.get("complementaryK",2.0)

        if "ALTITUDE" in config:
            alt = config["ALTITUDE"]
            self.altitudeType = alt.get("type",0)
            self.saturationValue = alt.get("saturationValue",30)

        if "startFrameID" in config:
            self.startFrameID = config["startFrameID"]
        if "endFrameID" in config:
            self.endFrameID = config["endFrameID"]
        if "visualizeImage" in config:
            self.visualizeImage = bool(config["visualizeImage"])
        if "delayVisualize" in config:
            self.delayVisualize = config["delayVisualize"]
        if "applyDerotation" in config:
            self.applyDerotation = bool(config["applyDerotation"])

        # We can parse more as needed...
        # e.g. you do "fpsVideo", "derivativeK", etc.

    def initializePlayback(self):
        print(f"Opening file {self.aviPath}")
        self.playback = cv2.VideoCapture(self.aviPath)
        if not self.playback.isOpened():
            print(f"Error opening video file: {self.aviPath}")

        # If camera’s exposureTime => compute fps
        if self.camParams.exposureTime > 0:
            fps_calc = 1e6 / self.camParams.exposureTime
            if fps_calc >= MAX_FPS_FLIR:
                fps_calc = MAX_FPS_FLIR
            self.fps = fps_calc
        else:
            # fallback to what's in the file
            fps_from_file = self.playback.get(cv2.CAP_PROP_FPS)
            if fps_from_file > 0:
                self.fps = fps_from_file
            else:
                self.fps = 30.0
        self.deltaTms = 1000.0 / self.fps
        print(f"Playback FPS: {self.fps}, deltaTms: {self.deltaTms}")

    def initializeOutputFile(self):
        if self.outputFileEnable:
            # create directory if needed
            dirpath = os.path.dirname(self.outputFilePath)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)

            self.outputFileStream = open(self.outputFilePath, 'w')
            print(f"Output file created: {self.outputFilePath}")
            self.outputFileStream.write(
                "timestamp,frameID,opticalFlow,featureDetection,"
                "totalProcessingTime,unfilteredAltitude,filteredAltitude,"
                "lidarData,distanceGround,rollAngle,pitchAngle,airspeed,"
                "groundspeed,vx,vy\n"
            )

    def initializeSensorData(self):
        """
        Mirroring your C++ logic for building CSV paths from the .avi path, 
        extracting 'rec...' substring, etc. 
        """
        lastSlash = self.aviPath.rfind("/")
        recPos = self.aviPath.find("rec", lastSlash+1)
        if recPos == -1:
            print("No 'rec' found in path, skipping sensor data load.")
            return

        dotAviPos = self.aviPath.find(".avi", recPos)
        if dotAviPos == -1:
            print("No '.avi' in path, skipping sensor data load.")
            return

        timestampStr = self.aviPath[recPos+3 : dotAviPos]
        # base directory
        recIndex = self.aviPath.find("/recordings/")
        if recIndex == -1:
            baseDir = os.path.dirname(self.aviPath)
        else:
            baseDir = self.aviPath[:recIndex]

        altitudeDir = os.path.join(baseDir, "altitude", f"altitude_{timestampStr}.csv")
        imuDir = os.path.join(baseDir, "imu", f"imu_{timestampStr}.csv")
        timeDir = os.path.join(baseDir, "time", f"timing_{timestampStr}.csv")
        # velocity data is in altitude csv in your code
        # or separate? 
        # We'll assume the same logic:
        print("Altitude file:", altitudeDir)
        print("IMU file:", imuDir)
        print("Time file:", timeDir)

        if os.path.exists(altitudeDir) and os.path.exists(timeDir) and os.path.exists(imuDir):
            self.loadIMUData(imuDir, self.imuData)
            print(f"IMU data loaded. Size: {len(self.imuData)}")

            self.loadVelocityData(altitudeDir, self.velocityData)
            print(f"Velocity data loaded. Size: {len(self.velocityData)}")

            self.loadFrameData(timeDir, self.frameData)
            if len(self.frameData) > 0:
                self.initialFrameID = self.frameData[0].frameID
                self.frameID = self.initialFrameID
            print("Frame data loaded.")
        else:
            print("Some sensor files do not exist. Creating fake data if needed...")

    def initializeFeatureDetector(self):
        self.fastDetector = cv2.FastFeatureDetector_create(
            threshold=self.fastParams.threshold,
            nonmaxSuppression=self.fastParams.nonmaxSuppression
        )

    # CSV loading
    def loadIMUData(self, filePath, imuDataList):
        with open(filePath, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row:
                    continue
                data = IMUData()
                data.timestamp = int(float(row[0]))
                data.q[0] = float(row[1])
                data.q[1] = float(row[2])
                data.q[2] = float(row[3])
                data.q[3] = float(row[4])
                data.gx = float(row[5])
                data.gy = float(row[6])
                data.gz = float(row[7])
                data.ax = float(row[8])
                data.ay = float(row[9])
                data.az = float(row[10])
                data.lidarData = float(row[11])
                data.airspeed = float(row[12])
                data.groundspeed = float(row[13])
                data.roll_angle = float(row[14])
                data.pitch_angle = float(row[15])
                if len(row) > 16:
                    data.yaw_angle = float(row[16])
                imuDataList.append(data)

    def loadVelocityData(self, filePath, velocityDataList):
        """
        In your code, the 'altitude' CSV might store velocity. 
        So adapt to parse the correct columns. 
        We'll mimic your logic. 
        """
        with open(filePath, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row:
                    continue
                v = VelocityData()
                # e.g. row: Timestamp, Vx_body_FRD, Vy_body_FRD, Vz_body_FRD, ...
                v.timestamp = int(float(row[0]))
                v.vx_frd[0] = float(row[1])
                v.vx_frd[1] = float(row[2])
                v.vx_frd[2] = float(row[3])
                # if there's more columns for FLU, parse them
                velocityDataList.append(v)

    def loadFrameData(self, filePath, frameDataList):
        with open(filePath, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row:
                    continue
                fd = FrameData()
                fd.timestamp = int(float(row[0]))
                fd.frameID = int(float(row[1]))
                frameDataList.append(fd)

    # Searching for closest
    def getClosestSensorData(self, dataList, timestamp):
        if not dataList:
            return None
        closest = dataList[0]
        minDiff = abs(dataList[0].timestamp - timestamp)
        for d in dataList:
            diff = abs(d.timestamp - timestamp)
            if diff < minDiff:
                minDiff = diff
                closest = d
        return closest

    def run(self):
        print("Starting main loop in Python ...")
        while True:
            ret, self.currFrame = self.playback.read()
            if not ret:
                print(f"No more frames. Exiting. Last frameID: {self.frameID}")
                break
            
            if (self.frameID - self.initialFrameID > self.startFrameID) and \
               ((self.frameID - self.initialFrameID < self.endFrameID) or (self.endFrameID == 0)):

                startTotal = time.perf_counter()

                if len(self.currFrame.shape) == 3:
                    self.currFrame = cv2.cvtColor(self.currFrame, cv2.COLOR_BGR2GRAY)

                idx = self.frameID - self.initialFrameID
                if 0 <= idx < len(self.frameData):
                    self.currentTimestamp = self.frameData[idx].timestamp
                else:
                    self.currentTimestamp = self.prevTimestamp + int(1e6/self.fps)

                self.deltaTms = (self.currentTimestamp - self.prevTimestamp)/1e6

                # get IMU, velocity
                self.currentIMUData = self.getClosestSensorData(self.imuData, self.currentTimestamp)
                
                if self.currentIMUData is None:
                    self.currentIMUData = IMUData()
                self.currentVelocityData = self.getClosestSensorData(self.velocityData, self.currentTimestamp)
                if self.currentVelocityData is None:
                    self.currentVelocityData = VelocityData()

                # fill currentSensorData
                self.currentSensorData.timestamp = self.currentTimestamp
                self.currentSensorData.vx = self.currentVelocityData.vx_frd[0]
                self.currentSensorData.vy = self.currentVelocityData.vx_frd[1]
                self.currentSensorData.vz = self.currentVelocityData.vx_frd[2]
                self.currentSensorData.gx = self.currentIMUData.gx
                self.currentSensorData.gy = self.currentIMUData.gy
                self.currentSensorData.gz = self.currentIMUData.gz
                self.currentSensorData.ax = self.currentIMUData.ax
                self.currentSensorData.ay = self.currentIMUData.ay
                self.currentSensorData.az = self.currentIMUData.az
                self.currentSensorData.airspeed = self.currentIMUData.airspeed
                self.currentSensorData.groundspeed = self.currentIMUData.groundspeed
                self.currentSensorData.lidarData = self.currentIMUData.lidarData
                self.currentSensorData.q = self.currentIMUData.q[:]
                self.currentSensorData.roll_angle = self.currentIMUData.roll_angle
                self.currentSensorData.pitch_angle = self.currentIMUData.pitch_angle
                self.currentSensorData.yaw_angle = self.currentIMUData.yaw_angle
                self.currentSensorData.frameID = self.frameID

                # if you want to compute cosRoll, sinRoll, etc.
                self.cosRoll = math.cos(self.currentIMUData.roll_angle)
                self.sinRoll = math.sin(self.currentIMUData.roll_angle)
                self.cosPitch = math.cos(self.currentIMUData.pitch_angle)
                self.sinPitch = math.sin(self.currentIMUData.pitch_angle)


                #now print every file of the self.currentSensorData
                # print(f"Timestamp: {self.currentSensorData.timestamp}")
                # print(f"vx: {self.currentSensorData.vx}")
                # print(f"vy: {self.currentSensorData.vy}")
                # print(f"vz: {self.currentSensorData.vz}")
                # print(f"gx: {self.currentSensorData.gx}")
                # print(f"gy: {self.currentSensorData.gy}")
                # print(f"gz: {self.currentSensorData.gz}")
                # print(f"ax: {self.currentSensorData.ax}")
                # print(f"ay: {self.currentSensorData.ay}")
                # print(f"az: {self.currentSensorData.az}")
                # print(f"airspeed: {self.currentSensorData.airspeed}")
                # print(f"groundspeed: {self.currentSensorData.groundspeed}")
                # print(f"lidarData: {self.currentSensorData.lidarData}")
                # print(f"q: {self.currentSensorData.q}")
                # print(f"roll_angle: {self.currentSensorData.roll_angle}")
                # print(f"pitch_angle: {self.currentSensorData.pitch_angle}")
                # print(f"yaw_angle: {self.currentSensorData.yaw_angle}")
                # print(f"frameID: {self.currentSensorData.frameID}")
                

                # process frames
                if not self.FASTLKParallel:
                    self.processFrames()
                else:
                    self.processFramesParallel()

                # clear flow vectors
                self.flowVectors.clear()
                self.filteredFlowVectors.clear()

                endTotal = time.perf_counter()
                totalDuration = (endTotal - startTotal)*1e6

                if self.outputFileStream:
                    self.outputFileStream.write(
                        f"{self.currentTimestamp},"
                        f"{self.frameID},"
                        f"{self.ofTime},"
                        f"{self.featureDetectionTime},"
                        f"{totalDuration},"
                        f"{self.avgAltitude},"
                        f"{self.filteredAltitude},"
                        f"{self.currentSensorData.lidarData},"
                        f"{self.currentSensorData.distance_ground},"
                        f"{self.currentSensorData.roll_angle},"
                        f"{self.currentSensorData.pitch_angle},"
                        f"{self.currentSensorData.airspeed},"
                        f"{self.currentSensorData.groundspeed},"
                        f"{self.currentSensorData.vx},"
                        f"{self.currentSensorData.vy}\n"
                    )

                if self.endFrameID != 0 and (self.frameID - self.initialFrameID) >= self.endFrameID:
                    print("Reached endFrameID => stopping.")
                    break

                self.prevTimestamp = self.currentTimestamp

            self.frameID += 1

        print("Done. Closing resources.")
        self.closeOutputFile()
        if self.visualizeImage:
            cv2.destroyAllWindows()

    def closeOutputFile(self):
        if self.outputFileStream:
            print(f"Closing output CSV: {self.outputFilePath}")
            self.outputFileStream.close()
            self.outputFileStream = None
        if self.playback:
            self.playback.release()

    ################################################################
    # The main processing code (Optical Flow, derotation, altitude)
    ################################################################
    def processFrames(self):
        startOpticalFlow = time.perf_counter()
        self.calculateOpticalFlow(self.currFrame)
        endOpticalFlow = time.perf_counter()
        self.ofTime = (endOpticalFlow - startOpticalFlow)*1e6

        self.prevPoints.clear()

        startFeatureDetection = time.perf_counter()
        self.applyCornerDetection(self.currFrame)
        endFeatureDetection = time.perf_counter()
        self.featureDetectionTime = (endFeatureDetection - startFeatureDetection)*1e6

        self.prevFrame = self.currFrame.copy()
        self.processOF()

    def processFramesParallel(self):
        if self.visualizeImage:
            cv2.imshow("Image", self.currFrame)

        startLKFAST = time.perf_counter()

        self.FASTThread = threading.Thread(
            target=self.applyCornerDetectionParallelHelper,
            args=(self.currFrame,)
        )
        self.FASTThread.start()

        self.calculateOpticalFlow(self.currFrame)
        self.prevFrame = self.currFrame.copy()
        self.processOF()

        self.FASTThread.join()
        self.prevPoints = self.nextPrevPoints[:]

        endLKFAST = time.perf_counter()
        self.ofTime = (endLKFAST - startLKFAST)*1e6

    def applyCornerDetectionParallelHelper(self, image):
        self.applyCornerDetection(image, outputArray='nextPrevPoints')

    def applyCornerDetection(self, image, outputArray='prevPoints'):
        keypoints = self.fastDetector.detect(image, None)
        self.detectedFeatures = len(keypoints)

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
            if self.fastParams.safeFeatures and self.detectedFeatures < 0.5*self.fastParams.desiredFeatures:
                self.safeFeaturesApplied = True
                self.fastDetector.setThreshold(50)
                keypoints = self.fastDetector.detect(image, None)
                keypoints = randomlySampleKeypoints(keypoints, self.fastParams.desiredFeatures, 0.5)
                keypoints = scoreAndRankKeypointsUsingGradient(keypoints, image, self.fastParams.desiredFeatures)
                self.fastDetector.setThreshold(self.fastParams.threshold)
            else:
                keypoints = randomlySampleKeypoints(keypoints, self.fastParams.desiredFeatures, 0.0)

        self.filteredDetectedFeatures = len(keypoints)
        points = cv2.KeyPoint_convert(keypoints)
        if outputArray == 'prevPoints':
            self.prevPoints = points.tolist()
        else:
            self.nextPrevPoints = points.tolist()

    def calculateOpticalFlow(self, currFrame):
        if self.prevFrame is not None and len(self.prevPoints) > 0:
            status, err = None, None
            currPoints, status, err = cv2.calcOpticalFlowPyrLK(
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

            if currPoints is not None and status is not None:
                for i in range(len(currPoints)):
                    if status[i] == 1:
                        p1 = self.prevPoints[i]
                        p2 = currPoints[i]
                        fv = OFVectorFrame(p1, p2, self.fps, self.camParams)
                        self.flowVectors.append(fv)

                if self.visualizeImage:
                    flowVis = cv2.cvtColor(currFrame, cv2.COLOR_GRAY2BGR)
                    for i in range(len(currPoints)):
                        if status[i] == 1:
                            p1 = tuple(map(int, self.prevPoints[i]))
                            p2 = tuple(map(int, currPoints[i]))
                            cv2.arrowedLine(flowVis, p1, p2, (0,0,255),2)
                    cv2.imshow("OF", flowVis)
                    ##CORRECT UNTIL HERE

            self.filteredFlowVectors = rejectOutliersFrame(
                self.flowVectors,
                self.magnitudeThresholdPixel,
                self.boundThreshold
            )
            self.rejectedVectors = len(self.flowVectors) - len(self.filteredFlowVectors)

            self.avgGyro_rad_cam = self.getGyroData()

            print(f"gyro : {self.avgGyro_rad_cam}")
            if self.applyDerotation:
                for fv in self.filteredFlowVectors:
                    self.applyDerotation3D(fv, self.avgGyro_rad_cam)
            else:
                for fv in self.filteredFlowVectors:
                    self.applyDerotation3D(fv, np.array([0.0,0.0,0.0], dtype=np.float32))

            if self.visualizeImage:
                flowVisDerotated = cv2.cvtColor(currFrame, cv2.COLOR_GRAY2BGR)
                for fv in self.filteredFlowVectors:
                    p1 = (int(fv.position[0]), int(fv.position[1]))
                    p2 = (int(fv.nextPosition[0]), int(fv.nextPosition[1]))
                    cv2.arrowedLine(flowVisDerotated, p1, p2, (0,255,0),2)
                drawTextWithBackground(flowVisDerotated, f"Altitude: {self.filteredAltitude:.2f}", (10,20))
                drawTextWithBackground(flowVisDerotated, f"FrameID: {self.frameID}", (10,40))
                drawTextWithBackground(flowVisDerotated, f"Lidar: {self.currentSensorData.lidarData}", (10,60))
                drawTextWithBackground(flowVisDerotated, f"Gyro PX4: {self.currentSensorData.gx*180/np.pi,self.currentSensorData.gy*180/np.pi,self.currentSensorData.gz*180/np.pi}", (10,80))
                drawTextWithBackground(flowVisDerotated, f"PITCH, ROLL: {self.currentSensorData.pitch_angle,self.currentSensorData.roll_angle}", (10,100))

                cv2.imshow("OF_Derotated", flowVisDerotated)
                cv2.waitKey(self.delayVisualize)
        else:
            print("FIRST FRAME, skipping ..")
    ################################################################
    # Additional methods from your C++ code 
    ################################################################

    def getGyroData(self):
        """
        VisionNodeFramesPlayback::getGyroData() 
        => GYRO in FLU => convert to FRD => convert to camera
        """
        gx = self.currentSensorData.gx
        gy = self.currentSensorData.gy
        gz = self.currentSensorData.gz
        # FLU => FRD => (gx, -gy, -gz)
        output_FRD = np.array([gx, -gy, -gz], dtype=np.float32)
        # Then FRD => camera
        return bodyToCam(output_FRD, self.camParams)

    def applyDerotation3D(self, ofVector, avgGyroRadSec):
        """
        Same logic you posted for applyDerotation3D(...).
        We replicate the steps to compute a new nextPosition in 2D.
        """
        norm_a = np.linalg.norm(ofVector.AMeter)
        # Pprime_ms = [uPixelSec*pixelSize, vPixelSec*pixelSize]
        Pprime_ms = np.array([
            ofVector.uPixelSec * self.camParams.pixelSize,
            ofVector.vPixelSec * self.camParams.pixelSize
        ], dtype=np.float32)

        # PpPprime_ms => 3D: (x, y, 0) / norm_a
        PpPprime_ms = np.array([Pprime_ms[0]/norm_a, Pprime_ms[1]/norm_a, 0.0], dtype=np.float32)

        # P = PpPprime_ms - dot(...) * directionVector
        dot_val = np.dot(PpPprime_ms, ofVector.directionVector)
        P = PpPprime_ms - dot_val*ofVector.directionVector

        # RotOF = -(avgGyroRadSec cross directionVector)
        cross_val = np.cross(avgGyroRadSec, ofVector.directionVector)
        RotOF = -cross_val

        ofVector.P = P - RotOF

        # getDerotatedOF_ms => next
        OF_derotated = self.getDerotatedOF_ms(ofVector.P, ofVector.directionVector, ofVector.AMeter)

        # Now in pixel space for nextPosition
        derotNextX = ofVector.position[0] + OF_derotated[0]*self.deltaTms/(self.camParams.pixelSize*1e3)
        derotNextY = ofVector.position[1] + OF_derotated[1]*self.deltaTms/(self.camParams.pixelSize*1e3)
        ofVector.nextPosition = (derotNextX, derotNextY)
        ofVector.deltaX = ofVector.nextPosition[0] - ofVector.position[0]
        ofVector.deltaY = ofVector.nextPosition[1] - ofVector.position[1]

    def getDerotatedOF_ms(self, P_derotated, d_direction, aVector):
        """
        from your getDerotatedOF_ms(...) 
        """
        dot_val = np.dot(P_derotated, d_direction)
        Pprime_derotated = P_derotated + dot_val*d_direction
        scale = np.linalg.norm(aVector)
        Pprime_derotated *= scale
        return np.array([Pprime_derotated[0], Pprime_derotated[1]], dtype=np.float32)

    def estimateDepth(self, ofVector, T_cam):
        TdotD = np.dot(T_cam, ofVector.directionVector)
        tmp = T_cam - TdotD*ofVector.directionVector
        numerator = np.linalg.norm(tmp)
        denom = np.linalg.norm(ofVector.P)
        if denom < 1e-9:
            return float('nan')
        return numerator / denom

    def estimateAltitude(self, ofVector, depth):
        directionVector_body = camToBody(ofVector.directionVector, self.camParams)
        directionVector_inertial = bodyToInertial(directionVector_body,
                                                 self.cosRoll,self.sinRoll,
                                                 self.cosPitch,self.sinPitch)
        cosTheta = directionVector_inertial[2]
        return depth*cosTheta

    def processOF(self):
        T_airspeed = np.array([self.currentSensorData.airspeed,0.0,0.0], dtype=np.float32)
        T_airspeed_cam_sensordata = bodyToCam(T_airspeed, self.camParams)


        altitudes = []
        for ofv in self.filteredFlowVectors:
            depth = self.estimateDepth(ofv, T_airspeed_cam_sensordata)
            alt = self.estimateAltitude(ofv, depth)
            if not math.isnan(alt):
                altitudes.append(alt)

        # average/median
        if not altitudes:
            self.avgAltitude = self.prevFilteredAltitude
        else:
            if self.altitudeType == 1:
                altitudes.sort()
                self.avgAltitude = altitudes[len(altitudes)//2]
            else:
                self.avgAltitude = sum(altitudes)/len(altitudes)

        # saturate
        if self.avgAltitude >= self.saturationValue:
            self.avgAltitude = self.saturationValue

        if math.isnan(self.avgAltitude) or self.avgAltitude == 0.0:
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

        if math.isnan(self.filteredAltitude):
            self.prevFilteredAltitude = 0.0
        else:
            # check dropped
            if self.deltaTms <= (3/self.fps)*1000.0:
                self.prevFilteredAltitude = self.filteredAltitude
            else:
                print(f"FRAME DROPPED: {self.frameID}")


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    yaml_file = "/home/sharedData/config/config_flir_playback.yaml"  # adapt to your path
    node = VisionNodeFramesPlayback(yaml_file)
    node.run()
    print("All done.")
