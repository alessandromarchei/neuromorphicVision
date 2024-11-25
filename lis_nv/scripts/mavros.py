#!/usr/bin/env python

import rospy
from mavros_msgs.msg import StatusText

def send_statustext(message, severity):
    # Initialize the ROS node
    rospy.init_node('statustext_sender', anonymous=True)
    
    # Create a publisher for the /mavros/statustext/send topic
    pub = rospy.Publisher('/mavros/statustext/send', StatusText, queue_size=10)
    
    # Set the loop rate to 1 Hz
    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        # Create a StatusText message
        status_text = StatusText()
        status_text.text = message
        status_text.severity = severity  # Set the severity level
        
        # Publish the message
        rospy.loginfo("Sending message to Pixhawk: %s", message)
        pub.publish(status_text)
        
        # Sleep to maintain the 1 Hz rate
        rate.sleep()

if __name__ == '__main__':
    try:
        # Example message and severity to send
        message = "Hello, Pixhawk!"
        severity = 6  # Notice level 6 = info
        send_statustext(message, severity)
    except rospy.ROSInterruptException:
        pass
