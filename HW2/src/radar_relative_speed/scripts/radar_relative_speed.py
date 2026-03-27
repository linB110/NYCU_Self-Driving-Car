#!/usr/bin/env python3
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

def extract_point_fields(msg):

    points = []
    for point in point_cloud2.read_points(msg, skip_nans=True, field_names=[f.name for f in msg.fields]):
        field_dict = {f.name: point[i] for i, f in enumerate(msg.fields)}
        points.append(field_dict)
        
    return points

def is_vehicle_class(field_dict):
    vehicle_fields = ['ClassificationCar', 'ClassificationTruck']
    for f in vehicle_fields:
        if f in field_dict and field_dict[f] > 0:
            return True
            
    return False

def create_marker(x, y, z, text, marker_id, color=(1.0,0.0,0.0)):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "radar_velocity"
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z + 0.15
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.7  
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0
    marker.text = text
    marker.lifetime = rospy.Duration(0.2)
    
    return marker

def radar_callback(msg, marker_pub):
    points = extract_point_fields(msg)
    marker_array = MarkerArray()
    marker_id = 0
    for pt in points:
        if is_vehicle_class(pt):
            speed = pt['DynamicsRelVelX']
            x = pt['x'] if 'x' in pt else 0
            y = pt['y'] if 'y' in pt else 0
            z = pt['z'] if 'z' in pt else 0
            text = "{:.1f} m/s".format(speed)
                
            marker_bg = create_marker(x, y, z, text, marker_id, color=(0.0,0.0,0.0))
            marker_fg = create_marker(x, y, z, text, marker_id + 1000, color=(0.0,0.0,0.0))
                
            marker_array.markers.append(marker_bg)
            marker_array.markers.append(marker_fg)
                
            marker_id += 1

    marker_pub.publish(marker_array)

def main():
    rospy.init_node('radar_relative_speed', anonymous=True)
    marker_pub = rospy.Publisher('/radar_relative_speed_markers', MarkerArray, queue_size=10)
    rospy.Subscriber('/ars548/radar_front/objects', PointCloud2, radar_callback, marker_pub)
    rospy.loginfo("Radar relative speed node started...")
    rospy.spin()

if __name__ == '__main__':
    main()
