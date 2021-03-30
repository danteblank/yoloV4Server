import os
import flask
import logging
import io
from inference.inference import DataDetect
from flask import request, jsonify, send_file

from utils.utils import resp_json
from utils.exceptions import InvalidUsage

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

print('Initializing neural network')
yolo_v4 = DataDetect()

app = flask.Flask(__name__)


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.errorhandler(404)
def not_found(error):
    response = {"message": "404 - Not Found Error"}
    return resp_json(404, response)


@app.route('/api/classify_small', methods=['POST'])
def classify_small():
    try:
        data = request.data
        image = io.BytesIO(data)
        finish = yolo_v4.classify_image(image=image)
        return resp_json(200, finish)
    except Exception:
        raise InvalidUsage('400 - Bad Request', status_code=400)


@app.route('/api/classify_full', methods=['POST'])
def classify_full():
    try:
        image = request.data
        finish = yolo_v4.classify_full_image(image=image)
        return resp_json(200, finish)
    except Exception:
        raise InvalidUsage('400 - Bad Request', status_code=400)


@app.route('/api/draw_image', methods=['POST'])
def draw_image():
    try:
        image = request.data
        finish = yolo_v4.return_boxes_on_image(image=image)
        return send_file(finish), os.remove(finish)
    except Exception:
        raise InvalidUsage('400 - Bad Request', status_code=400)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=9999, use_reloader=False)
