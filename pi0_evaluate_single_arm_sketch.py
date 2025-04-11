import dataclasses
import logging
import time

import einops
from functools import partial
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro
from pynput import keyboard
from PIL import Image

from robots.airbots.airbot_play.airbot_play_2 import AIRBOTPlay, AIRBOTPlayConfig
from envs.airbot_play_real_env import make_env
from annotate.sketch_via_web.sketch_server import ImageSketchServer


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    cams: list[int] = dataclasses.field(default_factory=lambda:[0,4])
    arms: list[int] = dataclasses.field(default_factory=lambda:[1])
    right: bool = True
    mit: bool = False
    
    max_episodes = 200
    ctrl_hz = 20

    arm_init: list[float] = dataclasses.field(
        default_factory=lambda:[
            -0.05664911866188049,
            -0.26874953508377075,
            0.5613412857055664,
            1.483367681503296,
            -1.1999313831329346,
            -1.3498512506484985,
            0,
        ]
    )


def main(args: Args) -> None:
    right_cfg = AIRBOTPlayConfig(can_bus=f"can{args.arms[0]}", eef_mode="gripper") # mit=args.mit
    right_robot = AIRBOTPlay(right_cfg)
    robots = [right_robot]

    cameras = (
        {
            "0": args.cams[1],# 4
            "1": args.cams[0],# 0
        }
    )
    
    env = make_env(
        record_images=True,
        robot_instance=robots,
        cameras=cameras,
    )

    env.set_reset_position(args.arm_init) # 多余的reset_position不会被使用
    ts = env.reset(sleep_time=1)
    
    server = ImageSketchServer()
    server.start()

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )  # 输入输出都是右-左顺序
    logging.info(f"Server metadata: {policy.get_server_metadata()}")

    obs = parse_obs(ts, ts.observation["images"]["1"])
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

    # logging.info("Waiting for drawing...")
    # # transfer the obs to PIL Image to draw on picture
    # img = ts.observation["images"]["1"]
    # img = img[:, :, ::-1]
    # img = Image.fromarray(np.uint8(img))
    # ref_img, _ = server.draw_img(img)       # draw on picture on website
    # ref_img = ref_img.resize((640,480))     # convert the result to input shape
    # ref_img.save("draw.png")
    # ref_img = np.asarray(ref_img)[:,:,0:3].transpose(1,0,2)
    # # ref_img = ref_img[:, :, ::-1]         # RGB to BGR
    # print(ref_img.shape)
    # logging.info("Get refence.")
    
    import cv2 
    img = cv2.imread("ref_train.jpg")
    img = np.array(img)
    print(img.shape)
    print(img)  
    img.transpose(1,0,2)
    ref_img = img
    input("Policy ready. Press Enter to inference.")
    for _ in range(args.max_episodes):
        obs = parse_obs(ts, ref_img)
        result = policy.infer(obs)
        action_buffer = parse_action(result["actions"])
        for i in range(action_buffer.shape[0]):
            ts = env.step(action=action_buffer[i], get_obs=True)
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
        img = image_tools.resize_with_pad(raw_obs["images"][cam_name], 224, 224)
        img = img[:,:,::-1]
        images[cam_name] = img

    state = raw_obs["qpos"]
    return {
        "observation/state": state,
        "observation/image": images["1"],
        "observation/wrist_image": images["0"],
        "observation/ref_image": ref_img,
        "prompt": "Transfer object according to the reference line.",
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
