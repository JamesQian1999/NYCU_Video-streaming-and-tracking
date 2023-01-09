from flask import Flask, Response, render_template, redirect, request, session
import cv2, sys, os
import argparse
import os
import time
from loguru import logger
import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from predect import Predictor, image_demo


#Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
camera = cv2.VideoCapture(0)
curent = None

def gen_frames(res = '1080p', cls = [], conf = 0.6):

    exp = get_exp(None, 'yolox-s')
    model = exp.get_model()
    model.eval()

    ckpt_file = 'YOLOX/yolox_s.pth'
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    predictor = Predictor( model, exp, COCO_CLASSES, None, None, 'cpu', False, False)
    
    while True:
        success, frame = camera.read()  # read the camera frame
       
        if not success:
            break
        else:
            # print(session['resolution'], file=sys.stderr)
            if   res == '720p':
                frame = cv2.resize(frame, (1280, 720))
            elif res == '480p':
                frame = cv2.resize(frame, (854, 480))
            elif res == '360p':
                frame = cv2.resize(frame, (640, 360))
            elif res == '240p':
                frame = cv2.resize(frame, (426, 240))
            
            frame = image_demo(predictor, frame, conf = conf, cls = cls)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            global current
            current =  (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            yield current        

# Get video
@app.route('/video_feed')
def video_feed():
    try:
        session['resolution']
    except:
        session['resolution'] = '1080p'

    try:
        session['cls']
    except:
        session['cls'] = []
    
    try:
        session['conf']
    except:
        session['conf'] = 0.6

    current = gen_frames(res = session['resolution'], cls = session['cls'], conf = session['conf'])

    return Response(current, mimetype='multipart/x-mixed-replace; boundary=frame')


#start/stop
@app.route('/')
def index():
    previous_data = {"selected_item":"1080p"}
    return render_template('index.html',previous_data=previous_data)

@app.route('/start', methods=['GET', 'POST'])
def start():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Start':
            return redirect('/')
        elif request.form['submit_button'] == 'Pause':
            try:
                previous_data = {"selected_item":session['resolution'] }  
            except:
                previous_data = {"selected_item":"1080p" }  
            return render_template('freeze.html', previous_data=previous_data)


@app.route('/freeze')
def freeze():
    return Response(current, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/resolution', methods=['GET', 'POST'])
def resolution():
    if request.method == 'POST':
        if request.form['operator'] == '1080p':
            session['resolution'] = '1080p'
        elif request.form['operator'] == '720p':
            session['resolution'] = '720p'
        elif request.form['operator'] == '480p':
            session['resolution'] = '480p'
        elif request.form['operator'] == '360p':
            session['resolution'] = '360p'
        elif request.form['operator'] == '240p':
            session['resolution'] = '240p'
        
        previous_data = {"selected_item":session['resolution'] }      
        return render_template('index.html',previous_data=previous_data)


@app.route('/detect', methods=['POST'])
def my_form_post():
    text = request.form['text']
    text = str(text.lower())

    try:
        session['cls']
    except:
        session['cls'] = []

    c = session['cls']
    if request.form['submit_button'] == 'Add':
        print('Add:', text, file=sys.stderr)
        if text not in session['cls']:
            c.append(text)
            session['cls'] = c
        
            
    elif request.form['submit_button'] == 'Remove':
        print('Reomve:', text, file=sys.stderr)
        if text in session['cls']:
            c.remove(text)
            session['cls'] = c

    elif request.form['submit_button'] == 'submit':
        session['conf'] = float(text)
        print("session['conf']:", session['conf'], file=sys.stderr)

    print("session['cls']:", session['cls'], file=sys.stderr)
    
    
    try:
        previous_data = {"selected_item":session['resolution'] }  
    except:
        previous_data = {"selected_item":"1080p" }  
    return render_template('index.html', previous_data=previous_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0',  port='8000', debug=True, threaded=True)


'''
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
'''