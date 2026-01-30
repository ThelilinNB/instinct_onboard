import numpy as np
import time
import sys
import cv2  # 需要 pip install opencv-python

# 引入 SDK
try:
    from pyorbbecsdk import *
except ImportError:
    print("❌ 错误: 未找到 pyorbbecsdk2")
    print("请运行: pip3 install pyorbbecsdk2")
    sys.exit(1)

# --- 这里复制了你修改后的核心驱动类 ---
class OrbbecCamera:
    def __init__(self, resolution: tuple[int, int], fps: int):
        self.resolution = resolution
        self.fps = fps
        self.depth_scale = 0.001

        self.pipeline = Pipeline()
        self.config = Config()

        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                # 尝试获取 Y16 格式
                depth_profile = profile_list.get_video_stream_profile(
                    self.resolution[0], 
                    self.resolution[1], 
                    OBFormat.Y16, 
                    self.fps
                )
                print(f"✅ 成功匹配配置: {self.resolution} @ {self.fps}FPS")
            except Exception:
                print(f"⚠️ 警告: 不支持 {self.resolution}，尝试使用默认配置...")
                depth_profile = profile_list.get_default_video_stream_profile()
                self.resolution = (depth_profile.get_width(), depth_profile.get_height())
                print(f"➡️ 使用默认配置: {self.resolution} @ {depth_profile.get_fps()}FPS")

            self.config.enable_stream(depth_profile)
            self.pipeline.start(self.config)
            
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            sys.exit(1)

        print("⏳ 相机预热中...")
        for _ in range(10):
            self.pipeline.wait_for_frames(100)

    def get_camera_data(self):
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return None
        
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        
        # 提取数据
        data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        data = data.reshape((height, width))
        
        # 转为米 (float32)
        return data.astype(np.float32) * self.depth_scale
    
    def close(self):
        self.pipeline.stop()

# --- 主程序入口 ---
if __name__ == "__main__":
    print("🚀 开始启动奥比中光 Gemini 2...")
    
    # 常用分辨率：640x400 (Gemini 2) 或 640x480
    cam = OrbbecCamera((640, 400), 30)
    
    try:
        while True:
            # 1. 获取深度数据 (米)
            depth_map = cam.get_camera_data()
            
            if depth_map is not None:
                # 2. 为了显示，把米转回 0-255 的图像
                # 将 0米-2米 的范围映射到 0-255，超过2米的都算最远
                display_img = np.clip(depth_map, 0, 2.0) / 2.0 * 255
                display_img = display_img.astype(np.uint8)
                
                # 上色
                display_img = cv2.applyColorMap(display_img, cv2.COLORMAP_JET)
                
                cv2.imshow("Orbbec Camera Test", display_img)
            
            # 按 Q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    finally:
        cam.close()
        cv2.destroyAllWindows()
        print("✅ 相机已关闭")