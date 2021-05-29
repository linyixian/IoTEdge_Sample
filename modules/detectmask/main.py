from flask import Flask, render_template, Response
import os
import sys
import onnxruntime
import onnx
import numpy as np
from PIL import Image, ImageDraw
from object_detection import ObjectDetection
import tempfile
import cv2

MODEL_FILENAME = 'model.onnx'
LABELS_FILENAME = 'labels.txt'

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self,od_model,fontType):

        success, frame = self.video.read()

        image=Image.fromarray(frame)
        predictions = od_model.predict_image(image)

        for prediction in predictions:

            if prediction['probability']>=0.5:
                height=image.height
                width=image.width
                x1=int(prediction['boundingBox']['left']*width)
                y1=int(prediction['boundingBox']['top']*height)
                x2=int(x1+width*prediction['boundingBox']['width'])
                y2=int(y1+height*prediction['boundingBox']['height'])

                frame=cv2.rectangle(frame,(x1,y1),(x2,y2),color=(0,0,255),thickness=2)
                cv2.putText(frame,prediction['tagName'],(x1,y1-15),fontType,2,(0,0,255),1,cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels):
        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        model = onnx.load(model_filename)
        with tempfile.TemporaryDirectory() as dirpath:
            temp = os.path.join(dirpath, os.path.basename(MODEL_FILENAME))
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
            onnx.save(model, temp)
            self.session = onnxruntime.InferenceSession(temp)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        if self.is_fp16:
            inputs = inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):

    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = ONNXRuntimeObjectDetection(MODEL_FILENAME, labels)

    cap=cv2.VideoCapture(0)

    fontType = cv2.FONT_HERSHEY_COMPLEX

    while True:
        frame = camera.get_frame(od_model,fontType)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

