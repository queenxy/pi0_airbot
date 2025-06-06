import dataclasses
import logging
import time

import sys
import os

# 获取脚本所在目录的父目录（即项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro
from pynput import keyboard
from PIL import Image

from airbot_infer.robots.airbots.airbot_play.airbot_play_2 import AIRBOTPlay, AIRBOTPlayConfig
from airbot_infer.envs.airbot_play_real_env import make_env
from annotate.sketch_via_web.sketch_server import ImageSketchServer

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    cams: list[int] = dataclasses.field(default_factory=lambda:[2,4])
    arms: list[int] = dataclasses.field(default_factory=lambda:[0])
    right: bool = True
    mit: bool = False
    
    max_episodes = 200
    ctrl_hz = 20

    arm_init: list[float] = dataclasses.field(
        default_factory=lambda:[0.057,-0.27,0.56,1.48,-1.2,-1.35,0]
    )
    
import threading
import queue
import collections
import math

# 使用双端队列作为动作缓冲区
action_deque = collections.deque()  # 只保留最新的动作块
obs_queue = queue.Queue(maxsize=1)  # 观测队列

# 共享状态变量
class SharedState:
    def __init__(self):
        self.paused = False
        self.reset = False
        self.shutdown = False
        # 添加动作队列锁确保线程安全
        self.action_lock = threading.Lock()

shared_state = SharedState()

def blend_actions(old_action, new_action, alpha=0.7):
    """平滑融合两个动作，减少抖动"""
    # 线性插值融合
    blended_action = (1 - alpha) * old_action + alpha * new_action
    return blended_action

def inference_thread(policy, args):
    first_infer = True
    """推理线程：生成动作块并替换队列"""
    while not shared_state.shutdown:
        try:
            # 获取最新观测（阻塞式）
            obs = obs_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        # 执行推理
        t = time.time()
        result = policy.infer(obs)
        infer_t = time.time() - t
        n = math.ceil(infer_t / (1 /args.ctrl_hz))
        if first_infer:
            action_buffer = parse_action(result["actions"])
            first_infer = False
        else:
            action_buffer = parse_action(result["actions"])[n:]
            
        with shared_state.action_lock:
            if action_deque and len(action_buffer) > 0:
                # 计算需要融合的动作数量（取两个序列的较小值）
                blend_count = min(len(action_deque),len(action_buffer))
                for i in range(blend_count):
                    alpha = (i+1)/blend_count  #权重逐渐增大
                    blended = blend_actions(action_deque[i], action_buffer[i], alpha)
                    action_deque[i] = blended
                
                # 逐步融合动作
                # blend_count = min(len(action_deque), 15)
                # for i in range(min(len(action_deque),len(action_buffer))):
                #     if i < blend_count:
                #         # 随时间增加融合权重
                #         alpha = (i+1)/blend_count  #权重逐渐增大
                #         blended = blend_actions(action_deque[i], action_buffer[i], alpha)
                #         action_deque[i] = blended
                #     else:
                #         action_deque[i] = action_buffer[i]
                
                # 添加新chunk的剩余部分
                if len(action_buffer) > len(action_deque):
                    action_deque.extend(action_buffer[len(action_deque):])
            else:
                # 没有需要融合的动作，直接替换整个队列
                action_deque.clear()
                if len(action_buffer) > 0:
                    action_deque.extend(action_buffer)
            
            print(f"Inference: Added {len(action_buffer)} actions, total queue size: {len(action_deque)}")
            
            
def execution_thread(env, args, ref_img):
    """执行线程：从队列弹出并执行动作"""
    ts = env.reset(sleep_time=1)
    obs = parse_obs(ts, ref_img)
    obs_queue.put(obs)  # 初始观测
    t = time.time()
    
    while not shared_state.shutdown:
        # 处理暂停状态
        while shared_state.paused and not shared_state.shutdown:
            time.sleep(0.1)
        
        action = None
        with shared_state.action_lock:
            if action_deque:
                action = action_deque.popleft()
        
        # 如果没有可用动作，等待推理
        if action is None:
            time.sleep(0.01)
            continue
            
                # 执行环境动作
        ts = env.step(action=action, get_obs=True)
        delta_t = time.time() - t
        if 1/args.ctrl_hz - delta_t > 0:
            time.sleep(1/args.ctrl_hz - delta_t)
        
        # 更新观测数据
        new_obs = parse_obs(ts, ref_img)
        try:
            # 清空队列中所有旧观测
            while True:
                obs_queue.get_nowait()
        except queue.Empty:
            pass  # 队列已空是正常情况
        finally:
            # 放入最新观测
            obs_queue.put(new_obs)
        t = time.time()
        
        # 再次检查暂停
        while shared_state.paused and not shared_state.shutdown:
            time.sleep(0.1)

        # 处理重置请求
        if shared_state.reset:
            print("Starting new episode.")
            shared_state.reset = False
            ts = env.reset(sleep_time=1)
            # 更新初始观测
            obs = parse_obs(ts, ref_img)
            try:
                # 清空队列中所有旧观测
                while True:
                    obs_queue.get_nowait()
            except queue.Empty:
                pass  # 队列已空是正常情况
            finally:
                # 放入最新观测
                obs_queue.put(new_obs)
            
            # 清空动作队列
            with shared_state.action_lock:
                action_deque.clear()
            continue
        
        
def main(args: Args) -> None:
    right_cfg = AIRBOTPlayConfig(can_bus=f"can{args.arms[0]}", eef_mode="gripper") # mit=args.mit
    right_robot = AIRBOTPlay(right_cfg)
    robots = [right_robot]

    cameras = (
        {
            "0": args.cams[0],
            "1": args.cams[1],
        }
    )
    
    env = make_env(
        record_images=True,
        robot_instance=robots,
        cameras=cameras,
    )

    env.set_reset_position(args.arm_init) # 多余的reset_position不会被使用
    ts = env.reset(sleep_time=1)    # raw image in ts is 480*640*3, 0-255 unit8, BGR mode
    
    server = ImageSketchServer()
    server.start()

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )  # 输入输出都是右-左顺序
    logging.info(f"Server metadata: {policy.get_server_metadata()}")
    
    # Make sure camera 0 is base image
    base_img = ts.observation["images"]["0"][..., [2, 1, 0]]
    print(base_img.shape)
    base_img = Image.fromarray(np.uint8(base_img), mode="RGB")
    base_img.save("base.jpg")
    
    obs = parse_obs(ts, ts.observation["images"]["0"])
    policy.infer(obs)

    def on_press(key):
        try:
            if key.char == "p":
                shared_state.paused = not shared_state.paused
            elif key.char == "r":
                shared_state.reset = True
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    logging.info("Waiting for drawing...")
    # transfer the obs to PIL Image to draw on picture
    img = ts.observation["images"]["0"][..., [2, 1, 0]]
    img = Image.fromarray(np.uint8(img), mode="RGB")
    img.save("before_draw.jpg")
    ref_img, _, _ = server.draw_img(img)       # draw on picture on website
    ref_img = ref_img.resize((640,480)).convert('RGB')     # convert the result to input shape
    ref_img.save("draw.png")
    ref_img = np.asarray(ref_img)   # ref_img is 480*640*3, 0-255 uint8, RGB mode
    logging.info("Get refence.")
    
    input("Policy ready. Press Enter to inference.")
    # ... (previous code remains the same until the main loop)
    
    inf_thread = threading.Thread(
        target=inference_thread, 
        args=(policy, args),
        daemon=True
    )
    exec_thread = threading.Thread(
        target=execution_thread,
        args=(env, args, ref_img),
        daemon=True
    )
    
    inf_thread.start()
    exec_thread.start()


def parse_obs(ts, ref_img) -> dict:
    raw_obs = ts.observation

    images = {}
    for cam_name in raw_obs["images"]:
        # resize will be automatically down in src/openpi/models/model.py
        # img = image_tools.resize_with_pad(raw_obs["images"][cam_name], 224, 224)
        images[cam_name] = raw_obs["images"][cam_name][..., [2, 1, 0]] # convert bgr to rgb

    state = raw_obs["qpos"]
    return {
        "observation/state": state,
        "observation/image": images["0"],
        "observation/wrist_image": images["1"],
        "observation/ref_image": ref_img,
        "prompt": "You are tasked with placing specific building blocks onto designated positions on a shelf. \
        You will be provided with three images: the first two are real-time observations from cameras, \
        and the third is a reference image illustrating the task. The reference image uses lines to indicate \
        which building blocks need to be moved and where they should be placed. Based on the reference image,\
        first predict the 2D visual trajectory coordinates on the image for the movement, and then output the corresponding actions for the robotic arm.",
    }


def parse_action(result):
    action = np.array(result)
    action = action[...,:7]
    return action



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main(tyro.cli(Args))
    except KeyboardInterrupt:
        pass
