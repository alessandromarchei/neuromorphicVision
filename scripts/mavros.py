#!/usr/bin/env python

import rospy
from sensor_msgs.msg import TimeReference
from std_msgs.msg import UInt32
import time
import datetime
import os
import pytz

def time_callback(data):
    # Extract the time reference (Unix timestamp)
    gps_time = data.time_ref.to_sec()

    # Convert to UTC datetime object
    utc_datetime = datetime.datetime.utcfromtimestamp(gps_time)

    # Convert UTC time to local time (Switzerland)
    local_tz = pytz.timezone('Europe/Zurich')
    local_datetime = utc_datetime.replace(tzinfo=pytz.utc).astimezone(local_tz)

    # Format the local datetime to a string
    local_time_str = local_datetime.strftime('%Y-%m-%d %H:%M:%S')

    print("Synchronized GPS Date and Time (Local):", local_time_str)

    # Set the Raspberry Pi system time to the local GPS time
    os.system(f'sudo date -s "{local_time_str}"')

    # Once the time is set, shutdown the node
    rospy.signal_shutdown("Time synchronized with Pixhawk.")

def satellites_callback(data):
    print("satellites number:", data)

def main():
    rospy.init_node('mavros_data_reader', anonymous=True)

    # Subscribe to the time_reference topic from MAVROS
    rospy.Subscriber("/mavros/time_reference", TimeReference, time_callback)

    rospy.Subscriber("/mavros/global_position/raw/satellites", UInt32, satellites_callback)

    # Keep the node running until it shuts down after setting the time
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
