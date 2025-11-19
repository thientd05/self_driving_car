import argparse
import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from tensorflow.keras.models import load_model # type: ignore
import tensorflow.keras as keras # type: ignore
import math

# SỬ DỤNG MÔ HÌNH TỐT NHẤT: save_at21.keras
sio = socketio.Server()
app = Flask(__name__)
model = None

def send_control(steering_angle, throttle): # hàm gửi dữ liệu đến trình mô phỏng
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

@sio.on('connect') # sự kiện connect
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 1)

@sio.on('telemetry') # sự kiện telemetry để nhận dữ liệu từ trình mô phỏng và xử lý
def telemetry(sid, data):
    if data:
        speed = data["speed"]
        imgString = data["image"]
        image_array = np.asarray(Image.open(BytesIO(base64.b64decode(imgString))))

        prediction = model.predict(image_array[None, :, :, :], batch_size=1, verbose=1) # dự đoán dữ liệu
        steering_angle = float(prediction[0, 0])  # lấy dữ liệu dự đoán góc lái
        throttle = (float(prediction[0, 1]) + 1.0) / 2.0 # lấy dữ liệu dự đoán ga
        throttle = throttle if math.fabs(float(steering_angle)) < 0.2 or float(speed) < 20 else -0.08*throttle

        print(f"Steering angle:{steering_angle},Throttle:{throttle}")
        send_control(steering_angle, throttle)  # gửi lệnh điều khiển
    else:
        sio.emit('manual', data={}, skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model .keras file. Model should be on the same path.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), app)