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
    
    # pause handler
    paused = False
    reset = False

    def on_press(key):
        nonlocal paused
        nonlocal reset
        try:
            if key.char == "p":
                paused = not paused
            elif key.char == "r":
                reset = True
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

    for _ in range(args.max_episodes):
        obs = parse_obs(ts, ref_img)
        result = policy.infer(obs)
        action_buffer = parse_action(result["actions"])
        
        for i in range(action_buffer.shape[0]):
            # Check if paused before executing any actions
            while paused:
                time.sleep(0.1)  # Reduce CPU usage while paused
            
            ts = env.step(action=action_buffer[i], get_obs=True)
            
            # Check again for pause after each step
            while paused:
                time.sleep(0.1)
                
            time.sleep(1/args.ctrl_hz)

            if reset:
                print("Starting new episode.")
                reset = False
                ts = env.reset(sleep_time=1)
                break


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
