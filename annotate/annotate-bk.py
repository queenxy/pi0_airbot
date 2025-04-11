import os
import argparse
import requests
import numpy as np
from PIL import Image
from sketch_via_web.sketch_server import ImageSketchServer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    args = parser.parse_args()

    server = ImageSketchServer()
    server.start_server_in_background()

    name = args.name
    episodes_dir = os.listdir(args.name)
    episodes_dir = sorted([x for x in episodes_dir if x.isdigit()])
    for episode in episodes_dir:
        episode_dir = f"{name}/{episode}"
        imgs_dirs = os.listdir(episode_dir)
        for imgs_dir in imgs_dirs:
            if "cam1" in imgs_dir:
                img = Image.open(f"{episode_dir}/{imgs_dir}/frame_000000.jpg")
                res, points = server.draw_img(img)
                res = res.convert('RGB')
                assert res.size == (640, 480) or res.size == (1280, 960)
                res = res.resize((640, 480))
                res.save(f"{episode_dir}/condition.jpg")
                print(f"saved {episode_dir}/condition.jpg")

                start_time = points[0]['time']
                end_time = points[-1]['time']
                line_duration = end_time - start_time
                scale_factor = 2 if res.size == (1280, 960) else 1
                parsed_points = []
                for i in range(len(points)):
                    x = points[i]['x'] / scale_factor
                    y = points[i]['y'] / scale_factor
                    t = (points[i]['time'] - start_time) / line_duration
                    parsed_points.append([x, y, t])
                parsed_points = np.array(parsed_points)
                np.save(f"{episode_dir}/condition.npy", parsed_points)
            break
    
    requests.request('POST', "http://localhost:5000/shutdown")

if __name__ == "__main__":
    main()