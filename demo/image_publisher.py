import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import glob
 
class ImageFolderPublisher(Node):
    def __init__(self):
        super().__init__('image_folder_publisher')

        # 設定參數（可以透過 launch 或 CLI 指定）
        self.declare_parameter("image_folder", "../input")  # 設定要發佈的資料夾
        self.declare_parameter("publish_rate", 10.0)  # 設定發佈頻率 (Hz)
        self.declare_parameter("output_topic", "image_raw")  # 設定發佈的 topic 名稱

        # 讀取參數值
        self.image_folder = self.get_parameter("image_folder").get_parameter_value().string_value
        self.publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        # 建立影像發佈者
        self.publisher = self.create_publisher(Image, self.output_topic, 10)
        self.bridge = CvBridge()

        # 讀取資料夾內所有圖片
        self.image_files = sorted(glob.glob(os.path.join(self.image_folder, "*.*")))
        self.image_files = [f for f in self.image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not self.image_files:
            self.get_logger().error(f"No images found in {self.image_folder}")
            return
        
        self.get_logger().info(f"Found {len(self.image_files)} images. Publishing to {self.output_topic} at {self.publish_rate} Hz")

        # 設定計時器，以固定頻率發佈圖片
        self.index = 0
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_image)

    def publish_image(self):
        if not self.image_files:
            return

        img_path = self.image_files[self.index]
        self.get_logger().info(f"Publishing image: {img_path}")

        image = cv2.imread(img_path)
        if image is None:
            self.get_logger().error(f"Failed to load image: {img_path}")
            return

        # 轉換 OpenCV 影像為 ROS 影像訊息
        msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        self.publisher.publish(msg)

        # 更新索引（如果到最後一張則重新開始）
        self.index = (self.index + 1) % len(self.image_files)

def main():
    rclpy.init()
    node = ImageFolderPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
