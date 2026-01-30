import sys
import numpy as np
import time

# --- 尝试导入 SDK ---
try:
    # 通常 pip install pyorbbecsdk2 后，导入名依然是 pyorbbecsdk
    from pyorbbecsdk import *
except ImportError:
    try:
        from pyorbbecsdk2 import *
    except ImportError:
        print("❌ 无法导入 pyorbbecsdk 或 pyorbbecsdk2")
        print("请检查安装：pip3 list | grep pyorbbecsdk")
        sys.exit(1)

def main():
    pipeline = Pipeline()
    config = Config()

    # --- 1. 配置相机 ---
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        # 尝试 640x400 @ 30fps (Gemini 2 常用配置)
        try:
            profile = profile_list.get_video_stream_profile(640, 400, OBFormat.Y16, 30)
        except OBError:
            # 如果找不到，就用默认的
            profile = profile_list.get_default_video_stream_profile()
        
        config.enable_stream(profile)
        pipeline.start(config)
        print(f"✅ 相机启动成功！分辨率: {profile.get_width()}x{profile.get_height()}")
        
    except Exception as e:
        print(f"❌ 相机启动失败: {e}")
        return

    # --- 2. 循环获取数据并打印 ---
    try:
        while True:
            # 等待 100ms
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            # --- 数据解析 ---
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            
            # 1. 转换数据 (uint16, 单位 mm)
            # data 是一个一维数组，需要 reshape 成二维图片矩阵
            data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            data = data.reshape((height, width))
            
            # 2. 获取中心点的坐标
            center_x = width // 2
            center_y = height // 2
            
            # 3. 读取中心点的深度值
            center_distance_mm = data[center_y, center_x]
            
            # 4. 计算整个画面的统计信息 (可选)
            # 过滤掉 0 (无效值) 后计算平均距离
            valid_pixels = data[data > 0]
            if valid_pixels.size > 0:
                min_dist = np.min(valid_pixels)
                max_dist = np.max(valid_pixels)
                avg_dist = np.mean(valid_pixels)
            else:
                min_dist = max_dist = avg_dist = 0

            # --- 打印输出 ---
            # 为了不刷屏太快，我们把光标移回行首 (用 \r) 或者简单地 print
            print(f"📍 中心点距离: {center_distance_mm:4d} mm ({center_distance_mm/1000:.2f} m) | "
                  f"范围: {min_dist}-{max_dist} mm | "
                  f"平均: {avg_dist:.1f} mm")
            
            # 稍微睡一下，防止刷屏太快看不清
            # time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 程序已停止")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()