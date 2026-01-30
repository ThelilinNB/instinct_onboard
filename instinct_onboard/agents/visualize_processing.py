import numpy as np
import cv2
import time
import sys

# --- 导入 SDK (兼容处理) ---
try:
    from pyorbbecsdk import *
except ImportError:
    try:
        from pyorbbecsdk2 import *
    except ImportError:
        print("❌ 错误: 未找到 pyorbbecsdk。请安装: pip3 install pyorbbecsdk2")
        sys.exit(1)

# ================= 配置参数 (模拟 Agent 的 config) =================
# 1. 硬件参数
RS_RESOLUTION = (640, 400)  # Gemini 2 原始分辨率
RS_FPS = 30

# 2. 模型输入参数 (必须与训练时的 config 保持一致)
# 假设训练时的分辨率是 58x87 (Legged Gym 常用) 或者 64x64
# 你需要根据你的 agent.yaml 修改这里！
OUTPUT_RESOLUTION = (84, 56)  # (width, height)

# 3. 深度范围 (米)
DEPTH_RANGE = [0.0, 3.0]  # 小于0的归一化为0，大于3的归一化为1

# 4. 预处理开关
ENABLE_INPAINT = True      # 修复空洞
ENABLE_BLIND_SPOT = True   # 盲区遮挡
ENABLE_BLUR = True         # 高斯模糊

# 5. 盲区裁剪 (模拟狗头遮挡)
# [上, 下, 左, 右] 像素数 (注意这是基于 OUTPUT_RESOLUTION 的)
BLIND_SPOT_CROP = [0, 10, 0, 0]  # 假设底部有10个像素是自己身体

# ================================================================

class OrbbecCamera:
    """精简版相机驱动"""
    def __init__(self, resolution, fps):
        self.pipeline = Pipeline()
        config = Config()
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                profile = profile_list.get_video_stream_profile(resolution[0], resolution[1], OBFormat.Y16, fps)
            except:
                profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(profile)
            self.pipeline.start(config)
        except Exception as e:
            print(f"相机启动失败: {e}")
            sys.exit(1)
        self.depth_scale = 0.001

    def get_data(self):
        frames = self.pipeline.wait_for_frames(100)
        if not frames: return None
        depth_frame = frames.get_depth_frame()
        if not depth_frame: return None
        
        w, h = depth_frame.get_width(), depth_frame.get_height()
        data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
        return data.astype(np.float32) * self.depth_scale

    def close(self):
        self.pipeline.stop()

def process_depth_image(depth_image_np):
    """
    模拟 ParkourAgent.refresh_depth_frame 的核心逻辑
    """
    # 1. Resize (最近邻插值，保持硬边缘)
    depth_image = cv2.resize(depth_image_np, OUTPUT_RESOLUTION, interpolation=cv2.INTER_NEAREST)

    # 2. Inpaint (修复 < 0.2m 的黑洞，通常认为是噪声)
    if ENABLE_INPAINT:
        mask = (depth_image < 0.2).astype(np.uint8)
        # 注意：OpenCV inpaint 比较慢，实机部署有时会跳过这一步或者用更快的算法
        depth_image = cv2.inpaint(depth_image, mask, 3, cv2.INPAINT_NS)

    # 3. Blind Spot (自身遮挡置零)
    if ENABLE_BLIND_SPOT:
        h, w = depth_image.shape
        x1, x2, y1, y2 = BLIND_SPOT_CROP
        # 注意：这里的逻辑参考了你的 Agent 代码
        # x1:上, x2:下, y1:左, y2:右
        if x1 > 0: depth_image[:x1, :] = 0
        if x2 > 0: depth_image[h - x2:, :] = 0
        if y1 > 0: depth_image[:, :y1] = 0
        if y2 > 0: depth_image[:, w - y2:] = 0

    # 4. Gaussian Blur (平滑噪声)
    if ENABLE_BLUR:
        depth_image = cv2.GaussianBlur(depth_image, (3, 3), 0.5, 0.5)

    # 5. Clip & Normalize (归一化到 [0, 1])
    # 小于 min 的变 0，大于 max 的变 1
    filt_m = np.clip(depth_image, DEPTH_RANGE[0], DEPTH_RANGE[1])
    filt_norm = (filt_m - DEPTH_RANGE[0]) / (DEPTH_RANGE[1] - DEPTH_RANGE[0])

    return filt_norm

def main():
    print("🚀 启动相机可视化...")
    cam = OrbbecCamera(RS_RESOLUTION, RS_FPS)
    
    try:
        while True:
            # 1. 获取原始数据 (米)
            raw_depth_m = cam.get_data()
            if raw_depth_m is None: continue

            # 2. 执行 Agent 处理流程
            start_t = time.time()
            processed_norm = process_depth_image(raw_depth_m)
            proc_time = (time.time() - start_t) * 1000

            # --- 可视化渲染 ---
            
            # A. 原始图 (为了显示，归一化到 0-255 并上色)
            vis_raw = np.clip(raw_depth_m, 0, 3.0) / 3.0 * 255
            vis_raw = cv2.applyColorMap(vis_raw.astype(np.uint8), cv2.COLORMAP_JET)
            # 在图上写字
            cv2.putText(vis_raw, f"Original: {RS_RESOLUTION[0]}x{RS_RESOLUTION[1]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # B. 处理后的图 (它是 0-1 的 float，放大到 0-255)
            # 神经网络看到的通常是灰度图，这里我们用灰度显示，更能反映真实输入
            vis_proc = (processed_norm * 255).astype(np.uint8)
            # 放大回原始尺寸以便并排显示
            vis_proc_large = cv2.resize(vis_proc, RS_RESOLUTION, interpolation=cv2.INTER_NEAREST)
            # 转成3通道以便和彩色图拼接
            vis_proc_large = cv2.cvtColor(vis_proc_large, cv2.COLOR_GRAY2BGR)
            
            cv2.putText(vis_proc_large, f"Agent Obs: {OUTPUT_RESOLUTION[0]}x{OUTPUT_RESOLUTION[1]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_proc_large, f"Proc Time: {proc_time:.1f}ms", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # C. 拼接并显示
            combined = np.hstack((vis_raw, vis_proc_large))
            cv2.imshow("Orbbec Processing Debug (Left: Raw, Right: Network Input)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()