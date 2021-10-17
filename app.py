import glob
import time

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from videoProcessing import processor
import os
from multiprocessing import Process

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/output')
def output():
    filenames = [i for i in os.listdir() if i.endswith(".jpg")]
    print(filenames)
    return render_template('output.html', user_images = filenames)

@app.route('/', methods=['POST'])
def upload_file():
    removeFiles()
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        # Video processing script.
        task_cb = Process(target=processor, args=(uploaded_file.filename,))
        task_cb.start()
        # processor(uploaded_file.filename)
        
    return redirect(url_for('output'))

def removeFiles():
    filesList = glob.glob("*.jpg")+glob.glob("*.mp4")
    for file in filesList:
        os.remove(file)

@app.route('/image/<filename>')
def upload(filename):
    return send_from_directory(".", filename)

def main():
    app.run(debug=True, port=8000, host='0.0.0.0')

if __name__ == "__main__":
    main()

