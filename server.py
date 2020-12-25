#!/usr/bin/env python3
import os

from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from recorder import Recorder

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/script.js')
def post():
    return render_template('script.js')

@app.route('/servo', methods=['POST'])
def pos():
    #if request.method=='POST':
    #for key, value in request:
    #    print(str(key)+" "+str(value))
    f = float(request.data)
    print(f)
    return ''

@app.route('/motor', methods=['POST'])
def motor():
    data = request.data
    status = False
    if data == b'false':
        status = False
    elif data == b'true':
        status = True



    return ''

app.run(host='0.0.0.0')