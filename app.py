import sys

if 'ext/mrcnn/' not in sys.path:
    sys.path.append('ext/mrcnn/')

import tensorflow as tf

print()
print(tf.__version__)
print()

import cv2
import numpy as np
from PIL import Image
from src.image_processing import ImageProcessing
import os
from flask import Flask, render_template, request, make_response
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
from src import visualizations as viz

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html", file_path="img/image_here.jpg")


@app.route("/about")
@nocache
def about():
    return render_template('about.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


pipiline = ImageProcessing()

@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_now.jpg")
    copyfile("static/img/img_now.jpg", "static/img/img_normal.jpg")

    inp_img = cv2.imread("static/img/img_now.jpg")
    inp = {'img': inp_img}

    # Health
    preds = pipiline.apply(inp)
    viz.seg_visualization(preds['img'], preds['seg_preds'])
    viz.health_visualization(preds['health_preds'])
    viz.phase_visualization(preds['phase_preds'])

    return render_template("uploaded.html", file_path="img/img_now.jpg", preds=preds)

if __name__ == '__main__':
    app.run(debug=False, threaded=False, host="0.0.0.0")
    # app.run(debug=True, host="0.0.0.0")
