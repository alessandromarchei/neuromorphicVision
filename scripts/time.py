#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu, TimeReference
from datetime import datetime

class IMUTimestampCorrector:
    def __init__(self):
        rospy.init_node('imu_timestamp_corrector', anonymous=True)

        self.time_offset_ns = None  # Offset between system time and GPS time

        # Get system time at startup
        system_time_ns = rospy.Time.now().to_nsec()

        # Wait for GPS time reference
        rospy.loginfo("Waiting for GPS time from /mavros/time_reference...")
        gps_time_msg = rospy.wait_for_message('/mavros/time_reference', TimeReference, timeout=5.0)

        if gps_time_msg:
            gps_time_ns = gps_time_msg.time_ref.to_nsec()
            self.time_offset_ns = gps_time_ns - system_time_ns
            rospy.loginfo(f"System Time: {system_time_ns} ns")
            rospy.loginfo(f"GPS Time Reference: {gps_time_ns} ns")
            rospy.loginfo(f"Calculated Offset: {self.time_offset_ns} ns")
        else:
            rospy.logwarn("Failed to receive GPS time. IMU timestamps will not be corrected.")

        # Subscribe to IMU data
        self.imu_sub = rospy.Subscriber('/mavros/imu/data', Imu, self.imu_callback)

    def imu_callback(self, msg):
        imu_time_ns = msg.header.stamp.to_nsec()

        if self.time_offset_ns is not None:
            corrected_imu_time_ns = imu_time_ns + self.time_offset_ns
            corrected_imu_time_human = datetime.utcfromtimestamp(corrected_imu_time_ns / 1e9).strftime('%Y-%m-%d %H:%M:%S.%f')

            rospy.loginfo(f"IMU Timestamp (Original): {imu_time_ns} ns")
            rospy.loginfo(f"IMU Timestamp (Corrected): {corrected_imu_time_ns} ns | {corrected_imu_time_human}")
        else:
            rospy.logwarn("IMU timestamp not corrected (No GPS time received).")
            imu_time_human = datetime.utcfromtimestamp(imu_time_ns / 1e9).strftime('%Y-%m-%d %H:%M:%S.%f')
            rospy.loginfo(f"IMU Timestamp (Uncorrected): {imu_time_ns} ns | {imu_time_human}")

if __name__ == '__main__':
    try:
        IMUTimestampCorrector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
