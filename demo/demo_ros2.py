# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


class Mask2FormerNode(Node):
    def __init__(self):
        super().__init__('mask2former_node')

        # 宣告 ROS 參數
        self.declare_parameter("config_file", "../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
        self.declare_parameter("weights_file", "../ckpt/coco/model_final_f07440.pkl")
        self.declare_parameter("input_topic", "/usb_cam/image_raw")  # 訂閱影像的 topic
        self.declare_parameter("output_topic", "segmented_image")  # 發佈處理後影像的 topic
        self.declare_parameter("confidence_threshold", 0.5)

        # 讀取參數
        self.config_file = self.get_parameter("config_file").get_parameter_value().string_value
        self.weights_file = self.get_parameter("weights_file").get_parameter_value().string_value
        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value

        # 設定 Mask2Former 模型
        self.cfg = self.setup_cfg()
        self.demo = VisualizationDemo(self.cfg)

        # 初始化 ROS 影像訂閱與發佈
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, self.input_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(f"Subscribed to {self.input_topic}, publishing to {self.output_topic}")

    def setup_cfg(self):
        """設定 Mask2Former 的 Detectron2 配置"""
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(self.config_file)

        # 設定模型權重檔案
        if self.weights_file:
            cfg.MODEL.WEIGHTS = self.weights_file
            self.get_logger().info(f"Using model weights: {self.weights_file}")
        else:
            self.get_logger().warn("No weights file specified. Using default from config.")

        cfg.freeze()
        return cfg

    def image_callback(self, msg):
        """當接收到影像時執行推論，並發佈結果"""
        self.get_logger().info("Received an image, processing...")

        # 轉換 ROS Image 訊息到 OpenCV 格式
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 執行 Mask2Former 推論
        start_time = time.time()
        predictions, visualized_output = self.demo.run_on_image(img)
        elapsed_time = time.time() - start_time
        self.get_logger().info(f"Inference completed in {elapsed_time:.2f} seconds.")

        # 轉換回 ROS Image 訊息並發佈
        output_image = visualized_output.get_image()[:, :, ::-1]  # RGB to BGR
        output_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
        self.publisher.publish(output_msg)

        self.get_logger().info("Published segmented image.")

def main():
    rclpy.init()
    node = Mask2FormerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()