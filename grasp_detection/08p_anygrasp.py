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
from my_custom_interfaces.srv import TriggerWithBox
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import cv2
from datetime import datetime
from scipy.spatial.transform import Rotation as R

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
    super().__init__('node_anygrasp')
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

    self.create_service(TriggerWithBox, "/triggerwithbox", callback=self.cb_trigger, callback_group=self.callback_group_A)
    self.create_subscription(Image, "/D455/color/image_raw", callback=self.cb_img_raw, callback_group=self.callback_group_B, qos_profile=10)
    self.create_subscription(Image, "/D455/aligned_depth_to_color/image_raw", callback=self.cb_img_dep, callback_group=self.callback_group_B, qos_profile=10)

    self.anygrasp = AnyGrasp(cfgs)
    self.anygrasp.load_net()

    xmin, xmax = -1.0, 1.0
    ymin, ymax = -1.0, 1.0
    zmin, zmax = 0.0, 2.0
    self.lims = [xmin, xmax, ymin, ymax, zmin, zmax]

  def cb_trigger(self, req, response):
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
    coords = map(self._get_uv_from_grasp, gg)
    bbox = req.bbox
    corner0 = bbox.corners[0]
    corner1 = bbox.corners[1]
    x1, y1 = corner0.x, corner0.y
    x2, y2 = corner1.x, corner1.y
    assert x2 > x1, f"x2 must be greater than x1: {x1} vs {x2}"
    assert y2 > y1, f"y2 must be greater than y1: {y1} vs {y2}"
    for idx, each in enumerate(coords):
      u, v = each
      if x1 <= u <= x2 and y1 <= v <= y2:
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = f"{self.camera.cam_name}_color_optical_frame"
        target_F =Myframe.from_posit_only(gg[idx].translation)
        target_F.R = R.from_matrix(gg[idx].rotation_matrix)
        ps.pose = target_F.as_geom_pose()
        response.pose_stamped = ps
        response.width_mm = int(gg[idx].width * 1000)
        response.score = gg[idx].score
        self.logger.info(f"Width: {gg[idx].width*1000:.0f}mm\tScore: {gg[idx].score:.2f}\tHt: {gg[idx].height*1000:.0f}mm")
        response.successcode = ConsPercept.FOUND.value
        break
    else:
      response.successcode = ConsPercept.NOTFOUND.value
      self.logger.warn(f"Out of {len(gg)} total grasps, none was within bbox")

    # visualization
    if cfgs.debug:
      grippers = gg.to_open3d_geometry_list()

      self.vis = o3d.visualization.Visualizer()
      self.vis.create_window()      
      self.vis.add_geometry(cloud)

      if response.successcode == ConsPercept.NOTFOUND.value:
        for ea in grippers:
          self.vis.add_geometry(ea)
      else:
        grippers[idx].paint_uniform_color([1, 0, 0])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(gg[idx].translation)
        sphere.paint_uniform_color([1, 0, 0])     
        self.vis.add_geometry(grippers[idx])
        self.vis.add_geometry(sphere)
      
      ctr = self.vis.get_view_control()
      ctr.set_front([-0.039, 0.63, -0.78])
      ctr.set_lookat([0.01, -0.02, 0.65])
      ctr.set_up([0.01, -0.78, -0.63])
      ctr.set_zoom(0.3)

      self.vis.run()
      self.vis.destroy_window()
    
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
