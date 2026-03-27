#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

def pointcloud_callback(msg, topic_name):
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not points:
        rospy.logwarn(f"[{topic_name}] No valid points in this frame")
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    total_points = len(points)
    x_range = [min(xs), max(xs)]
    y_range = [min(ys), max(ys)]
    z_range = [min(zs), max(zs)]
    print(f"Topic: {topic_name}")
    print(f"Timestamp: {msg.header.stamp.to_sec():.3f}")
    print(f"Total points: {total_points}")
    print(f"X range: [{x_range[0]:.2f}, {x_range[1]:.2f}] m")
    print(f"Y range: [{y_range[0]:.2f}, {y_range[1]:.2f}] m")
    print(f"Z range: [{z_range[0]:.2f}, {z_range[1]:.2f}] m")
    print("-" * 50)

def main():
    rospy.init_node("pc_profiler", anonymous=True)
    topics = ["/ouster/top_122219002200", "/ars548/radar_front/detections"]
    for topic in topics:
        rospy.Subscriber(topic, PointCloud2, pointcloud_callback, callback_args=topic)
    rospy.loginfo("PointCloud Profiler node started. Listening to topics...")
    rospy.spin()

if __name__ == "__main__":
    main()
