from __future__ import annotations

import ctypes
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import os
import time
import sys
from typing import Literal

import numpy as np

# --- [修改点 1] 替换 SDK 导入 ---
try:
    from pyorbbecsdk import *
except ImportError:
    print("错误: 未找到 pyorbbecsdk2。请确保已安装: pip3 install pyorbbecsdk2")
    sys.exit(1)

# 假设这些是你项目中的原有依赖，保留不动
from instinct_onboard.utils import _depth_to_ros_pointcloud_msg
from .unitree import UnitreeNode

REALSENSE_PROCESS_FREQUENCY_CHECK_INTERVAL = 500


class MpSharedHeader(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_double),  # bytes: 8
        ("writer_status", ctypes.c_uint32),  # bytes: 4, 0: idle, 1: writing
        ("writer_termination_signal", ctypes.c_uint32),  # bytes: 4, 0: alive, 1: should terminate
        ("_pad", ctypes.c_uint32 * 4),  # bytes: 16, pad to 32 bytes
    ]


SIZE_OF_MP_SHARED_HEADER = ctypes.sizeof(MpSharedHeader)  # bytes: 32
assert SIZE_OF_MP_SHARED_HEADER == 32


# --- [修改点 2] 重写相机驱动类 ---
class OrbbecCamera:
    def __init__(self, resolution: tuple[int, int], fps: int):
        self.resolution = resolution  # (width, height)
        self.fps = fps
        # 奥比中光深度图单位通常是 1mm，转换为米需要乘以 0.001
        self.depth_scale = 0.001

        # 初始化 Pipeline
        self.pipeline = Pipeline()
        self.config = Config()

        # 配置深度流
        try:
            # 1. 获取深度传感器的所有流配置
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            
            # 2. 尝试找到匹配分辨率、帧率和格式 (Y16) 的 Profile
            try:
                depth_profile = profile_list.get_video_stream_profile(
                    self.resolution[0], 
                    self.resolution[1], 
                    OBFormat.Y16, 
                    self.fps
                )
            except Exception:
                # 如果找不到指定配置，尝试回退到默认
                print(f"[OrbbecCamera] 警告: 不支持 {self.resolution} @ {self.fps}FPS，尝试使用默认配置...")
                depth_profile = profile_list.get_default_video_stream_profile()
                print(f"[OrbbecCamera] 使用默认 Profile: {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()}FPS")
                # 更新分辨率以匹配实际获取到的
                self.resolution = (depth_profile.get_width(), depth_profile.get_height())

            # 3. 启用配置
            self.config.enable_stream(depth_profile)
            
        except Exception as e:
            print(f"[OrbbecCamera] 配置流失败: {e}")
            raise

        # 启动 Pipeline
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"[OrbbecCamera] 启动失败: {e}")
            raise

        # 预热：等待几帧以稳定自动曝光
        # 注意：wait_for_frames 参数是超时(ms)
        for _ in range(20):
            self.pipeline.wait_for_frames(100)

    def get_frame(self):
        # 阻塞等待帧，超时 100ms
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None
        return frames.get_depth_frame()

    def get_camera_data(self) -> np.ndarray or None:
        depth_frame = self.get_frame()
        if depth_frame is None:
            return None
        
        # --- [修改点 3] 数据格式转换 ---
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        
        # 从 buffer 读取原始数据 (uint16, mm)
        data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        
        # 校验数据完整性
        if data.size != width * height:
            return None
            
        data = data.reshape((height, width))
        
        # 转换为 float32 (米)
        # 这里的计算是在子进程做的，不会阻塞主进程
        depth_data = data.astype(np.float32) * self.depth_scale
        
        return depth_data

    def close(self):
        if self.pipeline:
            self.pipeline.stop()


# --- [修改点 4] 进程函数使用 OrbbecCamera ---
def camera_process_func(
    resolution: tuple[int, int],
    fps: int,
    shm_name: str,
    camera_process_affinity: set[int] | None,
) -> None:
    if camera_process_affinity is not None:
        os.sched_setaffinity(os.getpid(), camera_process_affinity)
    
    # 实例化新的奥比中光驱动类
    camera = OrbbecCamera(resolution, fps)
    
    shared_memory = mp.shared_memory.SharedMemory(name=shm_name)
    header = MpSharedHeader.from_buffer(shared_memory.buf)
    
    # 注意：如果 Orbbec 使用默认 Profile 导致分辨率改变，这里 buffer 大小可能会有风险
    # 但在 __init__ 里我们是按传入的 resolution 申请的内存
    image_buffer = np.ndarray(
        resolution[::-1], dtype=np.float32, buffer=shared_memory.buf, offset=SIZE_OF_MP_SHARED_HEADER
    )
    
    camera_process_start_time = time.time()
    camera_process_counter = 0
    
    try:
        while True:
            camera_data = camera.get_camera_data()
            
            # 只有当获取到数据且形状匹配时才写入
            if camera_data is not None and camera_data.shape == image_buffer.shape:
                # mark in header to start writing
                header.writer_status = 1
                # write the camera data to the shared memory
                image_buffer[:] = camera_data
                header.timestamp = time.time()
                # mark in header to stop writing
                header.writer_status = 0
            
            # check if the writer termination signal is set
            if header.writer_termination_signal == 1:
                print("Writer termination signal set, exiting camera process.")
                header = None
                image_buffer = None
                break
            
            camera_process_counter += 1
            if camera_process_counter % REALSENSE_PROCESS_FREQUENCY_CHECK_INTERVAL == 0:
                hz = camera_process_counter / (time.time() - camera_process_start_time)
                print(f"Orbbec camera process running at {hz:.4f} Hz.")
                camera_process_counter = 0
                camera_process_start_time = time.time()
                
    except Exception as e:
        print(f"Camera process error: {e}")
    finally:
        camera.close()
        shared_memory.close()  # unlink in the main process


class RsCameraNodeMixin:
    """
    Mixin for camera sensor or processing nodes.
    Extend this class when implementing a ROS2 node related to camera sensing or image streams.
    """

    def __init__(
        self,
        *args,
        rs_resolution: tuple[int, int] = (640, 400),  # [建议] Gemini 2 常用分辨率
        rs_fps: int = 30,                             # [建议] Gemini 2 常用帧率
        rs_vfov_deg: float = 58.0,
        camera_individual_process: bool = False,
        camera_dead_behavior: Literal["restart", "raise_error", "none"] = "restart",
        main_process_affinity: set[int] | None = None,
        camera_process_affinity: set[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Add any depth-specific initialization here
        # 注意：这里虽然变量名还叫 rs_ (RealSense)，但实际上用于 Orbbec
        self.rs_resolution = rs_resolution
        self.rs_fps = rs_fps
        self.rs_vfov_deg = rs_vfov_deg
        self.camera_individual_process = camera_individual_process
        self.camera_dead_behavior = camera_dead_behavior
        self.main_process_affinity = main_process_affinity
        self.camera_process_affinity = camera_process_affinity
        self.camera = None
        self.camera_process = None
        self.request_queue = None
        self.result_queue = None
        self.initialize_camera()

    def initialize_camera(self):
        """Initialize the Camera (Orbbec) with the specified configuration."""
        if self.camera_individual_process:
            # self.rs_rgb_data = None # Todo: add rgb data support
            self.rs_depth_data = np.zeros(self.rs_resolution[::-1], dtype=np.float32)
            shm_size = (
                SIZE_OF_MP_SHARED_HEADER
                + np.prod(self.rs_resolution[::-1]) * np.dtype(self.rs_depth_data.dtype).itemsize
            )
            self.rs_shared_memory = mp_shm.SharedMemory(create=True, size=shm_size)
            self.rs_shared_header = MpSharedHeader.from_buffer(self.rs_shared_memory.buf)
            self.rs_image_buffer = np.ndarray(
                self.rs_resolution[::-1],
                dtype=np.float32,
                buffer=self.rs_shared_memory.buf,
                offset=SIZE_OF_MP_SHARED_HEADER,
            )
            self.rs_data_fresh_counter = 0
            self.camera_process = mp.Process(
                target=camera_process_func,
                args=(
                    self.rs_resolution,
                    self.rs_fps,
                    self.rs_shared_memory.name,
                    self.camera_process_affinity,
                ),
                daemon=True,
            )
            self.camera_process.start()
            if self.main_process_affinity is not None:
                os.sched_setaffinity(os.getpid(), self.main_process_affinity)
            
            # Get data for the first time
            self.refresh_rs_data() 
        else:
            # 单进程模式直接实例化
            self.camera = OrbbecCamera(
                resolution=self.rs_resolution,
                fps=self.rs_fps,
            )

    def restart_camera(self):
        """Restart the camera (process), but reusing the resources as much as possible."""
        self.get_logger().info("Restarting Orbbec camera.")
        if self.camera_individual_process:
            # Only restart the camera process while reusing the shared memory buffer.
            self.camera_process = mp.Process(
                target=camera_process_func,
                args=(
                    self.rs_resolution,
                    self.rs_fps,
                    self.rs_shared_memory.name,
                    self.camera_process_affinity,
                ),
                daemon=True,
            )
            self.camera_process.start()
        else:
            self.initialize_camera()

    def depth_image_to_pointcloud_msg(self, depth: np.ndarray):
        return _depth_to_ros_pointcloud_msg(
            depth=depth,
            frame_id="orbbec_depth_link",  # [修改] 建议修改 frame_id
            vfov_deg=self.rs_vfov_deg,
            stamp=self.get_clock().now().to_msg(),
        )

    def handle_camera_dead_behavior(self):
        if self.camera_dead_behavior == "restart":
            self.get_logger().error("Camera process is not alive. Restarting one.")
            self.restart_camera()
        elif self.camera_dead_behavior == "raise_error":
            raise RuntimeError("Camera process is not alive. Exiting.")
        elif self.camera_dead_behavior == "none":
            self.get_logger().warn("Camera process is not alive. User chose to do nothing")
        else:
            raise ValueError(f"Invalid camera process dead behavior: {self.camera_dead_behavior}")

    def refresh_rs_data(self) -> bool:
        """Currently refresh the depth data only."""
        refreshed = False
        if self.camera_individual_process:
            if self.camera_process is None or not self.camera_process.is_alive():
                self.handle_camera_dead_behavior()
            # Dump queue and get latest
            if self.rs_shared_header.writer_status == 0:
                rs_timestamp = self.rs_shared_header.timestamp
                self.rs_depth_data[:] = self.rs_image_buffer
                # [可选] 日志频率可以根据需要调整
                # self.get_logger().info(
                #     f"Orbbec depth data delayed: {(time.time() - rs_timestamp):.4f} s.", throttle_duration_sec=5.0
                # )
                refreshed = True
            self.rs_data_fresh_counter += 1
        else:
            if self.camera is None:
                self.handle_camera_dead_behavior()
            self.rs_depth_data = self.camera.get_camera_data()  # (height, width)
            refreshed = True
        return refreshed

    def destroy_node(self):
        if self.camera_individual_process and self.camera_process:
            self.rs_shared_header.writer_termination_signal = 1
            self.camera_process.join(timeout=1.0)
            if self.camera_process.is_alive():
                self.get_logger().warn("Camera process is still alive after timeout. Terminating and joining.")
                self.camera_process.terminate()
                self.camera_process.join()
            self.rs_image_buffer = None
            self.rs_shared_header = None
            self.camera_process = None
            self.rs_shared_memory.close()
            self.rs_shared_memory.unlink()
            self.rs_shared_memory = None
        super().destroy_node()


class UnitreeRsCameraNode(RsCameraNodeMixin, UnitreeNode):
    pass