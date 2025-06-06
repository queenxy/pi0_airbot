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

from airbot_infer.robots.airbots.airbot_play.airbot_play_2 import AIRBOTPlay, AIRBOTPlayConfig
from airbot_infer.envs.airbot_play_real_env import make_env


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
    ts = env.reset(sleep_time=1)

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )  # 输入输出都是右-左顺序
    logging.info(f"Server metadata: {policy.get_server_metadata()}")

    obs = parse_obs(ts)
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

    input("Policy ready. Press Enter to inference.")
    for _ in range(args.max_episodes):     
        obs = parse_obs(ts)
        t = time.time()
        result = policy.infer(obs)
        print(time.time()-t)
        action_buffer = parse_action(result["actions"])
        for i in range(action_buffer.shape[0]):
            ts = env.step(action=action_buffer[i], get_obs=True)
            time.sleep(1/args.ctrl_hz)

            if reset:
                print("Starting new episode.")
                reset = False
                ts = env.reset(sleep_time=1)
                break


def parse_obs(ts) -> dict:
    raw_obs = ts.observation

    images = {}
    for cam_name in raw_obs["images"]:
        img = image_tools.resize_with_pad(raw_obs["images"][cam_name], 224, 224)
        images[cam_name] = img

    state = raw_obs["qpos"]
    return {
        "observation/state": state,
        "observation/image": images["1"],
        "observation/wrist_image": images["0"],
        "prompt": "put the cuboid into the bowl",
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
