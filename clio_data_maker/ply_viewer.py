# Publish a dense colmap mesh to ros to visualize in rviz.
# Note the pose transform must be provided if you want metric scale and aligned.
import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np

file_path = '/home/dominic/cam_ws/imgs/office_2_2024-03-08-07-16-39/dense/meshed-poisson.ply'
# file_path = '/home/dominic/Documents/office_modified_mesh.ply'
# file_path = '/home/dominic/Downloads/building33_5cm.ply'

scale = 2.10
transform_matrix = np.array([[-0.85042481,  0.46934804, -0.23767638, -7.08815597],
[ 0.50329045,  0.59422864, -0.62736835, -4.50304495],
[-0.15321999, -0.65314986, -0.74156517,  0.64755245],
[ 0.,          0.,          0.,          1.        ]])

def load_ply(file_path):
    cloud = o3d.io.read_point_cloud(file_path)
    return cloud

def convert_open3d_to_ros(cloud):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "world"  # Change the frame_id as needed

    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)

    # Combine points and colors into a structured array
    cloud_data = np.empty(len(points), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('rgb', np.uint32)  # Use a single UINT32 field for color
    ])

    points = scale * points
    points = np.hstack((points, np.ones((points.shape[0],1))))
    print(points.shape)
    points = np.transpose(transform_matrix@np.transpose(points))
    points = points/points[:,3:]


    cloud_data['x'] = points[:, 0]
    cloud_data['y'] = points[:, 1]
    cloud_data['z'] = points[:, 2]

    # Pack RGB channels into a single UINT32 field
    cloud_data['rgb'] = ((colors[:, 0] * 255.0).astype(np.uint32) << 16 |
                         (colors[:, 1] * 255.0).astype(np.uint32) << 8 |
                         (colors[:, 2] * 255.0).astype(np.uint32))

    # Convert the structured array to bytes
    cloud_msg_data = cloud_data.tobytes()

    cloud_msg = PointCloud2(
        header=header,
        height=1,
        width=len(points),
        is_dense=False,
        is_bigendian=False,
        fields=[
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1)
        ],
        point_step=16,  # Size of one point (4 fields * 4 bytes each)
        row_step=16 * len(points),  # Full length of the message
        data=cloud_msg_data
    )

    return cloud_msg

def publish_ply_as_pointcloud(pub, cloud_msg):
    pub.publish(cloud_msg)

def main():
    rospy.init_node('ply_viewer_node')

    cloud = load_ply(file_path)

    pub = rospy.Publisher('ply_cloud', PointCloud2, queue_size=1)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        cloud_msg = convert_open3d_to_ros(cloud)
        publish_ply_as_pointcloud(pub, cloud_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
