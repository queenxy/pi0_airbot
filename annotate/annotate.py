import os
import argparse
import requests
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sketch_via_web.sketch_server import ImageSketchServer

RAW_PATH = '/dysData/kairui/workspace/Imitate-All/data/raw'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='transfer_block')
    parser.add_argument("--save_root", type=str, default='/dysData/kairui/workspace/Imitate-All/data/output')
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    server = ImageSketchServer()
    server.start()

    name = os.path.join(RAW_PATH, args.name)
    episodes_dir = os.listdir(name)
    # episodes_dir = sorted([x for x in episodes_dir if x.isdigit()])
    save_dir = os.path.join(args.save_root, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for episode in tqdm(episodes_dir, desc=f"Annotating Task {name}"):
        if args.start and int(episode) < args.start:
            continue
        episode_dir = f"{name}/{episode}"
        # wrist_view_dir = f"{episode_dir}/observation.images.cam1.mp4"
        # side_view_dir = f"{episode_dir}/observation.images.cam2.mp4"
        wrist_view_dir = f"{episode_dir}/observation.images.cam2.mp4"
        side_view_dir = f"{episode_dir}/observation.images.cam1.mp4"
        # read video frames via cv2
        assert os.path.exists(side_view_dir) and os.path.exists(wrist_view_dir)

        raw_dirs = [side_view_dir]

        cap = cv2.VideoCapture(raw_dirs[0])
        _, frame = cap.read()
        frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_pil)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ref_imgs = []
        n = 9
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // n * i)
            _, ref_frame = cap.read()
            ref_frame_pil = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
            ref_img = Image.fromarray(ref_frame_pil)
            ref_imgs.append(ref_img)

        res, points = server.draw_img(img, ref_imgs)
        res = res.convert('RGB')
        if res.size == (1280, 960):
            res = res.resize((640, 480), Image.LANCZOS)
        assert res.size == (640, 480)

        res.save(f'{save_dir}/cond_ep{episode}.jpg')

        start_time = points[0]['time']
        end_time = points[-1]['time']
        line_duration = end_time - start_time

        # save annotated points
        parsed_points = []
        for i in range(len(points)):
            x = points[i]['x']
            y = points[i]['y']
            t = (points[i]['time'] - start_time) / line_duration
            parsed_points.append([x, y, t])
        parsed_points = np.array(parsed_points)
        np.save(f"{save_dir}/cond_ep{episode}.npy", parsed_points)
    

if __name__ == "__main__":
    main()