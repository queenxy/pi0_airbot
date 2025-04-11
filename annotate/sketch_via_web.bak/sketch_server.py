import io
import base64
import time
import logging
from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
import threading
from functools import wraps

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class ImageSketchServer:
    def __init__(self):
        self.app = app
        self.drawn_image = None  # 要绘制的图像
        self.result_image = None  # 用于存储绘制后的图像
        self.drawing_points = None
        self.image_lock = threading.Lock()  # 用于处理多张图像的同步问题
        self.setup_routes()
        print(f"visit http://127.0.0.1:5000")

    def route_wrapper(self, route_func):
        """装饰器用于包装路由函数，使其能够访问类实例"""
        @wraps(route_func)
        def wrapper(*args, **kwargs):
            return route_func(self, *args, **kwargs)
        return wrapper

    def setup_routes(self):
        @app.route('/')
        @self.route_wrapper
        def index(self):
            return render_template('index.html')

        @app.route('/get_image')
        @self.route_wrapper
        def get_image(self):
            if self.drawn_image:
                # 将图片转为base64编码
                buffered = io.BytesIO()
                self.drawn_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return jsonify({"image": img_str})
            return jsonify({"status": "No image available"}), 404

        @app.route('/draw', methods=['POST'])
        @self.route_wrapper
        def draw_image(self):
            drawing_data = request.json.get('drawing')  # 获取前端绘制的数据
            drawing_points = request.json.get('drawingPoints')
            if drawing_data:
                # 解码前端传来的base64图像
                img_data = base64.b64decode(drawing_data.split(',')[1])  # 去掉 base64 前缀
                self.drawn_image = Image.open(BytesIO(img_data))
                self.result_image = self.drawn_image  # 保存绘制后的图像
                self.drawing_points = drawing_points
                return jsonify({"status": "Image drawn successfully"}), 200
            return jsonify({"status": "Failed to draw image"}), 400
        
        @app.route('/shutdown', methods=['POST'])
        @self.route_wrapper
        def shutdown(self):
            # 从 request.environ 中获取 werkzeug 的 shutdown 方法
            shutdown_func = request.environ.get('werkzeug.server.shutdown')
            if shutdown_func is None:
                raise RuntimeError("Not running with the Werkzeug Server")
            shutdown_func()
            return 'Server shutting down...'

    def run(self, debug=False):
        app.run(debug=debug, use_reloader=False)

    def start_server_in_background(self):
        """在后台启动Flask服务器"""
        self.server_thread = threading.Thread(target=self.run, args=(False,))
        self.server_thread.daemon = True
        self.server_thread.start()

    def draw_img(self, img: Image.Image):
        """外部程序调用此函数，传入图像，并等待返回绘制后的图像"""
        with self.image_lock:
            self.drawn_image = img

            # 确保服务器已启动
            if not self.server_thread or not self.server_thread.is_alive():
                self.start_server_in_background()

            while self.result_image is None:
                time.sleep(0.01)  # 阻塞，直到图像被绘制并返回

            result = self.result_image
            points = self.drawing_points
            self.result_image = None
            self.drawn_image = None
            self.drawing_points = None

            # 返回绘制后的图像
            return result, points


if __name__ == '__main__':
    import requests
    import matplotlib.pyplot as plt

    image_paths = ['winxp.jpeg', 'win7.jpg', 'win10.jpeg']

    server = ImageSketchServer()
    server.start_server_in_background()

    for image_path in image_paths:
        img = Image.open(image_path)

        # 传递图像给服务器并等待返回绘制后的图像
        res, _ = server.draw_img(img)
        # res的尺寸可能为(1280, 960)或(640, 480)

        plt.imshow(res)
        plt.title(str(res.size))
        plt.show()

    requests.request('POST', "http://localhost:5000/shutdown")