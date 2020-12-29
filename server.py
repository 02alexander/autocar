#!/usr/bin/env python3
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
"""
from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)"""
from threading import Thread, Lock
import threading
from multiprocessing import Process, Pipe
import keras
import trainer
import numpy as np
import argparse

from recorder import Recorder
from car import Car
import cv2

lock = Lock()
deg = 8
motor_status = False

car = Car()

img_lock = Lock()
cur_img = None

def camera_reader():
    global img_lock, cur_img
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        img_lock.acquire()
        cur_img = frame
        img_lock.release()


def autonomous_driver_server(conn, model_file_name=None):
    if model_file_name is None:
        return
    model = keras.models.load_model(model_file_name)
    while True:
        print("waiting for image")
        img = conn.recv()
        print("image received")
        if img is None:
            conn.send(8)
            continue
        
        img = np.array(img)
        input_shape = model.layers[0].input_shape
        rows = input_shape[1]

        processed_img = trainer.proc_img(img, rows_removed=30-rows, break_point=0.5)
        m = np.array(processed_img)
        raw_prediction = model.predict(m.reshape((1, rows, 30, 1)))
        prediction = None
        (_,c) = raw_prediction.shape
        if c == 1:
            prediction = np.round(raw_prediction, 0)
        else:
            prediction = np.argmax(raw_prediction,axis=1)[0]
        prediction += 1

        conn.send(prediction)


def car_controller(predictor_conn, alternating_autonomous=False, record_dir="replays/test"):
    global img_lock, lock, car
    
    # how many seconds it takes before it switches between being controlled by model and human.
    seconds_between_switch = 0.5
    # when it is controlled by the model it should not record any images.

    # how often an image and degree is to be stored.
    seconds_between_capture = 0.2

    rec = Recorder(record_dir)
    tlast_switch = time.time()
    tlast_capture = time.time()
    currently_autonomous = False
    
    while True:
        time.sleep(0.01)
        if (time.time()-tlast_switch) > seconds_between_switch and alternating_autonomous:
            print("switch")
            tlast_switch = time.time()
            currently_autonomous = not currently_autonomous

        if not currently_autonomous:
            lock.acquire()
            local_deg = deg
            print(deg)
            car.turn(deg)
            car.motor(motor_status)
            lock.release()

            img_lock.acquire()

            if cur_img is None:
                img_lock.release()
                print("continue")
                continue

            gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray, (30, 30))
            img_lock.release()

            if (time.time()-tlast_capture) > seconds_between_capture:
                tlast_capture = time.time()
                rec.store(img, deg=local_deg)

        else:
            img_lock.acquire()

            if cur_img is None:
                img_lock.release()
                continue

            gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray, (30, 30))
            predictor_conn.send(img)
            pred = predictor_conn.recv()
            img_lock.release()

            lock.acquire()
            car.turn(pred)
            lock.release()



class Server(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        path = self.path
        path = path[1:]
        if path == '' or path=='favicon.ico':
            path = "index.html"
        path = "templates/"+path
        f = open(path, "r")
        data = f.read()
        self.wfile.write(bytes(data, "utf8"))

    def do_POST(self):
        global lock, deg, motor_status
        
        #print()
        #print(self.path)
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        print(data)

        lock.acquire()
        if self.path == '/servo':
            f = float(data)
            deg = round(f*14.0)+1
        elif self.path == '/motor':
            if data == b'false':
                motor_status = False
            else:
                motor_status = True
        lock.release()


def server_thread():
    hostname = "192.168.10.223"
    port = 5000
    server = HTTPServer((hostname, port), Server)

    print("Server started http://%s:%s" % (hostname, port))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    server.server_close()
    print("Server stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dagger')
    parser.add_argument('--savedir', '-d')
    args = parser.parse_args()

    parent_conn, child_conn = Pipe()
    predictor = Process(target=autonomous_driver_server, args=(child_conn, args.dagger,))
    predictor.start()
    s = Thread(target=server_thread)
    s.start()
    cr = Thread(target=camera_reader)
    cr.start()
    car_controller(parent_conn, alternating_autonomous=args.dagger, record_dir=args.savedir)

    #controller = Thread(target=car_controller, args=(parent_conn,))
    #controller.start()
    #print(controller.ident)
    #predictor.terminate()
    #controller.join()


"""
def SERVER():
    app = Flask(__name__)

    @app.route('/')
    def hello():
        return render_template('index.html')

    @app.route('/script.js')
    def post():
        return render_template('script.js')

    @app.route('/servo', methods=['POST'])
    def pos():
        global deg, lock
        f = float(request.data)
        f = f*14.0
        p = round(f)+1
        print(p)
        lock.acquire()
        #car.turn(p)
        deg = p
        lock.release()
        return ''

    @app.route('/motor', methods=['POST'])
    def motor():
        global motor_status, lock
        data = request.data
        status = False
        if data == b'false':
            status = False
        elif data == b'true':
            status = True
        
        lock.acquire()
        #car.motor(status)
        motor_status = status
        lock.release()

        return ''

    app.run(host='0.0.0.0')
"""
