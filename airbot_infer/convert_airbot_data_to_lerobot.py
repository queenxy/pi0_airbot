"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
from pathlib import Path
import cv2
import json
import numpy as np

def main():
    # Clean up any existing dataset in the output directory
    # output_path = LEROBOT_HOME / REPO_NAME
    # if output_path.exists():
    #     shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id="airbot_tranfer_3x3",
        robot_type="airbot",
        fps=20,
        features={
            "image": {
                "dtype": "image",
                "shape": (640, 480, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (640, 480, 3),
                "names": ["height", "width", "channel"],
            },
            "pure_drawn_image": {
                "dtype": "image",
                "shape": (640, 480, 3),
                "names": ["height", "width", "channel"],
            },
            "ref_trajectory_image": {
                "dtype": "image",
                "shape": (640, 480, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 定义原始数据路径
    raw_data_dir = Path("data/3x3_transfer_1")
    
    # 遍历所有子目录
    for sub_dir in raw_data_dir.glob("*/"):
        # 检查是否存在两个摄像头视频文件
        cam1_path = sub_dir / "observation.images.cam1.mp4"
        cam2_path = sub_dir / "observation.images.cam2.mp4"
        lowdim_path = sub_dir / "low_dim.json"
        meta_path = sub_dir / "meta.json"
        pure_drawn_path = sub_dir / "pure_drawn.png"
        ref_trajectory_path = sub_dir / "ref_with_trajectory.jpg"

        if not pure_drawn_path.exists():
            raise FileNotFoundError(f"纯绘图图像文件不存在: {pure_drawn_path}")
        if not ref_trajectory_path.exists():
            raise FileNotFoundError(f"参考轨迹图像文件不存在: {ref_trajectory_path}")
        pure_drawn_img = cv2.imread(str(pure_drawn_path))
        ref_trajectory_img = cv2.imread(str(ref_trajectory_path))

        if not (cam1_path.exists() and cam2_path.exists() and lowdim_path.exists()):
            print(f"警告: 在目录 {sub_dir} 中缺少必要的视频文件或元数据文件")
            continue
            
        # 读取元数据
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
            length = meta_data["length"]

        # 获取关节位置数据
        with open(lowdim_path, "r") as f:
            lowdim_data = json.load(f)
            act_joint_positions = lowdim_data.get("action/arm/joint_position", [])
            act_eef_positions = lowdim_data.get("action/eef/joint_position", [])
            obs_joint_positions = lowdim_data.get("observation/arm/joint_position", [])
            obs_eef_positions = lowdim_data.get("observation/eef/joint_position", [])
        
        # 检查所有数据长度是否一致
        if (len(act_joint_positions) != length or 
            len(act_eef_positions) != length or
            len(obs_joint_positions) != length or
            len(obs_eef_positions) != length):
            print(f"警告: 在目录 {sub_dir} 中数据长度不一致 - 动作关节: {len(act_joint_positions)}, 动作末端: {len(act_eef_positions)}, 观测关节: {len(obs_joint_positions)}, 观测末端: {len(obs_eef_positions)}, 预期: {length}")
            continue

            
        # 为每个摄像头创建帧目录
        cam1_frames = sub_dir / "cam1_frames"
        cam2_frames = sub_dir / "cam2_frames"
        cam1_frames.mkdir(exist_ok=True)
        cam2_frames.mkdir(exist_ok=True)
        
        # 解码cam1视频，按20Hz频率提取帧
        cap1 = cv2.VideoCapture(str(cam1_path))
        fps = cap1.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps / 20))  # 计算帧间隔
        frame_count1 = 0
        saved_count1 = 0
        
        cam1_frames_list = []
        while True:
            ret, frame = cap1.read()
            if not ret:
                break
            # 每隔frame_interval帧保存到list
            if frame_count1 % frame_interval == 0:
                # 将RGB转为BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cam1_frames_list.append(frame_bgr)
                saved_count1 += 1
            frame_count1 += 1
        cap1.release()
        
        # 解码cam2视频，按20Hz频率提取帧到list
        cap2 = cv2.VideoCapture(str(cam2_path))
        fps = cap2.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps / 20))
        frame_count2 = 0
        saved_count2 = 0
        cam2_frames_list = []
        
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            # 每隔frame_interval帧保存到list
            if frame_count2 % frame_interval == 0:
                # 将RGB转为BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cam2_frames_list.append(frame_bgr)
                saved_count2 += 1
            frame_count2 += 1
        cap2.release()

        # 检查提取的帧数是否等于预期长度
        if saved_count1 != length or saved_count2 != length:
            print(f"警告: 在目录 {sub_dir} 中摄像头帧数不等 - cam1: {saved_count1}, cam2: {saved_count2}, 预期: {length}")

        truncate_idx = length
        for i in range(100, length):
            # 检查从当前帧到最后一帧的关节位置和末端执行器位置变化是否都小于阈值
            all_small = True
            check_length = length-i
            for j in range(check_length):
                joint_diff = np.abs(np.array(act_joint_positions[i+j]) - np.array(act_joint_positions[i+j-1]))
                eef_diff = np.abs(np.array(act_eef_positions[i+j]) - np.array(act_eef_positions[i+j-1]))
                if not (np.all(joint_diff < 1e-2) and np.all(eef_diff < 1e-2)):
                    all_small = False
                    break
            if all_small:
                truncate_idx = i
                break
        
        # 截断所有数据到相同长度
        if truncate_idx < length:
            # 截断joint position
            act_joint_positions = act_joint_positions[:truncate_idx]  
            act_eef_positions = act_eef_positions[:truncate_idx]
            obs_joint_positions = obs_joint_positions[:truncate_idx]
            obs_eef_positions = obs_eef_positions[:truncate_idx]
            # 截断图像帧
            cam1_frames_list = cam1_frames_list[:truncate_idx]
            cam2_frames_list = cam2_frames_list[:truncate_idx]
            

        for step in range(truncate_idx):
            dataset.add_frame(
                {
                    "image": cam1_frames_list[step],
                    "wrist_image": cam2_frames_list[step],
                    "ref_image": ref_trajectory_img,
                    "pure_ref_image": pure_drawn_img,
                    "state": np.concatenate([obs_joint_positions[step],obs_eef_positions[step]]),
                    "actions": np.concatenate([act_joint_positions[step],act_eef_positions[step]]),
                }
            )
        dataset.save_episode(task="put the block into the cup according to refence image.")

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

if __name__ == "__main__":
    tyro.cli(main)