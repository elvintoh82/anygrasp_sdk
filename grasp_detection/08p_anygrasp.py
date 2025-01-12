import os
import argparse
import torch
import numpy as np
import open3d as o3d

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

from my_libs.myframe import Myframe
from my_libs.mycamera import Camera
from my_libs.myconstants import ConsPercept
from my_custom_interfaces.srv import TriggerVisionGetAll, TriggerVisionGetCropped, TriggerVisionGetContours, TriggerVisionGetPose, TriggerVisionGetSolutions
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from datetime import datetime

from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default="log/checkpoint_detection.tar", help='Model checkpoint path')
parser.add_argument('--top_down_grasp', default=True, help='Output top-down grasps.')

parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--debug', default=True, action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

class Perception(Node):
  def __init__(self):
    super().__init__('Perception_node_anygrasp')
    self.logger = self.get_logger()
    assert torch.cuda.is_available(), 'No CUDA available!'
    self.camera = Camera.load_from_path("D455", Path("./results.json"))
    self.o3d_intrin = o3d.camera.PinholeCameraIntrinsic(*self.camera.intrinsic)
    self.callback_group_A = MutuallyExclusiveCallbackGroup()
    self.callback_group_B = MutuallyExclusiveCallbackGroup()
    self.bridge = CvBridge()
    self.img_raw = None
    self.img_dep = None
    self.mask = cv2.rectangle(np.zeros((self.camera.intrinsic[1], self.camera.intrinsic[0]), dtype=np.uint8), (150, 100), (500, 400), 255, -1)

    self.create_service(TriggerVisionGetSolutions, "/triggervisiongetsolutions", callback=self.cb_trigger_get_solutions, callback_group=self.callback_group_A)
    self.create_subscription(Image, "/D455/color/image_raw", callback=self.cb_img_raw, callback_group=self.callback_group_B, qos_profile=10)
    self.create_subscription(Image, "/D455/aligned_depth_to_color/image_raw", callback=self.cb_img_dep, callback_group=self.callback_group_B, qos_profile=10)

    self.anygrasp = AnyGrasp(cfgs)
    self.anygrasp.load_net()

    # set workspace to filter output grasps
    # xmin, xmax = -0.19, 0.12
    # ymin, ymax = 0.02, 0.15
    # zmin, zmax = 0.0, 1.0
    xmin, xmax = -1.0, 1.0
    ymin, ymax = -1.0, 1.0
    zmin, zmax = 0.0, 2.0
    self.lims = [xmin, xmax, ymin, ymax, zmin, zmax]


    # set your workspace to crop point cloud
    # mask = (points_z > 0) & (points_z < 1)
    # points = np.stack([points_x, points_y, points_z], axis=-1)



  def cb_trigger_get_solutions(self, req, response):
    self.logger.info("Triggered")
    if (self.img_raw is None) or (self.img_dep is None):
      self.logger.warn("Rgb and depth images have not been received yet. Try again.")
      response.successcode = ConsPercept.FAIL.value
      return response
    
    #we randomly save 50% of the images
    if np.random.rand() > 0.5:
      now = datetime.now().strftime("%y%m%d_%H%M%SH")
      cv2.imwrite(str(Path(f"./data/{now}_rgb.png")), self.img_raw)
      cv2.imwrite(str(Path(f"./data/{now}_dep.png")), self.img_dep)

    pcd = self._pcd_maker(self.img_raw, self.img_dep, None)
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)

    gg, cloud = self.anygrasp.get_grasp(points, colors, lims=self.lims, apply_object_mask=False, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
      print('No Grasp detected!')
      return ConsPercept.NOTFOUND.value

    gg = gg.nms().sort_by_score()
    #we are only interested in the top 20 scores
    # gg_pick = gg[0:20]
    # u, v = self._get_uv_from_grasp(gg_pick[0])

    coords = map(self._get_uv_from_grasp, gg[:20])
    for idx, each in enumerate(coords):
      print(f"{idx}: Score: {gg[idx].score:.2f}\tCoord: {each}")

    # visualization
    if cfgs.debug:
      grippers = gg.to_open3d_geometry_list()
      grippers[0].paint_uniform_color([1, 0, 0])
      sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
      sphere.translate(gg[0].translation)
      sphere.paint_uniform_color([1, 0, 0])
      # o3d.visualization.draw_geometries([*grippers, cloud, sphere])
      # o3d.visualization.draw_geometries([*grippers, cloud])
      try:
        o3d.visualization.draw_geometries([*grippers[:20], cloud, sphere])
      except Exception as e:
        self.logger.error(f"Error displaying o3d visualisation: {e}")
    
    return response
  
  def cb_img_raw(self, msg):
    self.img_raw = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
  
  def cb_img_dep(self, msg):
    self.img_dep = self.bridge.imgmsg_to_cv2(msg, '16UC1')

  def _pcd_maker(self, rgb, dep, mask):
    if mask is None:
      # if no mask is provided, we assume the whole image is the mask
      mask = np.ones(dep.shape, dtype=np.uint8) * 255
    # else:
    masked_rgb = cv2.bitwise_and(rgb, rgb, mask=mask)
    masked_dep = cv2.bitwise_and(dep, dep, mask=mask)
    # we remove depth values that are more than 900mm
    masked_dep[masked_dep > 900] = 0

    o3d_color = o3d.geometry.Image(cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2RGB))
    o3d_depth = o3d.geometry.Image(masked_dep)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d_color, depth=o3d_depth, depth_scale=1000, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.o3d_intrin, np.eye(4))
    return pcd
  
  def _get_uv_from_grasp(self, grasp):
    x, y, z = grasp.translation
    point = np.array([x, y, z])
    return self._get_uv_from_xyz(point)

  def _get_uv_from_xyz(self, point_array):
    # Project the 3D point array to 2D using the intrinsic matrix
    # extrinsic_matrix = np.eye(4)  # Assuming no extrinsic transformation

    #since extrinsic mtx is identity, so we skip multiplying it, and just use the intrinsic mtx in the 3x3 form directly
    # point_2d = self.camera.intrinsic_mtx @ (extrinsic_matrix @ point_3d)[:3]
    point_2d = self.camera.intrinsic_mtx @ point_array
    u = int(point_2d[0] / point_2d[2])
    v = int(point_2d[1] / point_2d[2])
    return u, v

def main(args=None):
  rclpy.init(args=args)
  abc = Perception()
  all_execs = MultiThreadedExecutor()
  all_execs.add_node(abc)
  try:
    abc.logger.info("Spinning...")
    all_execs.spin()
  except KeyboardInterrupt:
    pass
  finally:
    abc.destroy_node()
    abc.logger.warn(f"Kthx Bye {abc.__class__}")
    all_execs.shutdown()


if __name__ == '__main__':
  # demo('./example_data/')
  main()
