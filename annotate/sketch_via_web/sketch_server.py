import io
import math
import base64
import time
import logging
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw
from io import BytesIO
import threading
from functools import wraps

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class ImageSketchServer:
    def __init__(self):
        self.app = app
        app.config['SERVER_NAME'] = None
        self.drawn_image = None  # 要绘制的图像
        self.result_image = None  # 用于存储绘制后的图像
        self.reference_image = None  # 用于存储参考图像
        self.drawing_points = None
        self.image_lock = threading.Lock()  # 用于处理多张图像的同步问题
        self.setup_routes()
        print(f"visit http://127.0.0.1:5555")

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
            response = {}
            
            if self.drawn_image:
                # 将图片转为base64编码
                buffered = io.BytesIO()
                self.drawn_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                response["image"] = img_str
            
            if self.reference_image:
                # 将参考图片转为base64编码
                buffered = io.BytesIO()
                self.reference_image.save(buffered, format="JPEG")
                ref_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                response["reference_image"] = ref_img_str
            
            if response:
                return jsonify(response)
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

    def run(self, debug=False):
        app.run(debug=debug, use_reloader=False, threaded=True, host='127.0.0.1', port=5555)

    def start(self):
        """在后台启动Flask服务器"""
        self.server_thread = threading.Thread(target=self.run, args=(False,))
        self.server_thread.daemon = True
        self.server_thread.start()

    def draw_img(self, img: Image.Image, reference_img=None):
        """外部程序调用此函数，传入图像，并等待返回绘制后的图像
        
        Args:
            img: 要绘制的图像
            reference_img: 参考图像
        """
        with self.image_lock:
            self.drawn_image = img
            
            if isinstance(reference_img, list):
                reference_img_grid = Image.new('RGB', (640, 480))
                n = math.ceil(math.sqrt(len(reference_img)))
                w = 640 // n
                h = 480 // n
                for i, img in enumerate(reference_img):
                    img = img.resize((w, h))
                    img = img.convert('RGB')
                    reference_img_grid.paste(img, ((i%n)*w, (i//n)*h))
                self.reference_image = reference_img_grid
            elif isinstance(reference_img, Image.Image):
                self.reference_image = reference_img

            # 确保服务器已启动
            if not self.server_thread or not self.server_thread.is_alive():
                self.start()

            while self.result_image is None:
                time.sleep(0.01)  # 阻塞，直到图像被绘制并返回

            result = self.result_image
            points = self.drawing_points
            self.result_image = None
            self.drawn_image = None
            self.reference_image = None
            self.drawing_points = None

            # 返回绘制后的图像
            return result, points


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image_paths = ['winxp.jpeg', 'win7.jpg', 'win10.jpeg']

    server = ImageSketchServer()
    server.start()

    # 创建参考图像
    ref = Image.new('RGB', (640, 480))
    draw = ImageDraw.Draw(ref)
    draw.line([(0, 0), (640, 480)], fill='red', width=2)
    draw.line([(0, 480), (640, 0)], fill='blue', width=2)

    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            # 传递图像和参考图像给服务器并等待返回绘制后的图像
            res, _ = server.draw_img(img, ref)
            # res的尺寸可能为(1280, 960)或(640, 480)

            plt.imshow(res)
            plt.title(str(res.size))
            plt.show()
        except FileNotFoundError:
            print(f"找不到图像文件: {image_path}")
            continue
