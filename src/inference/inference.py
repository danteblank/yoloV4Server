from yolov4 import Detector
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import random
import os

from utils.image_saver import save_image
from utils.convert_points import convert_points


class DataDetect:
    def __init__(self):
        self.detector = Detector(gpu_id=0, lib_darknet_path='cfg/libdarknet.so', weights_path='cfg/yolov4.weights')

    def classify_image(self, image):
        out_dict = []
        img = Image.open(image)
        img_arr = np.array(img.resize((self.detector.network_width(), self.detector.network_height())))
        start = time.time()
        detections = self.detector.perform_detect(image_path_or_buf=img_arr)
        finish = time.time() - start
        for detection in detections:
            box_x, box_y, box_w, box_h = detection.left_x, detection.top_y, detection.width, detection.height
            detected = {"class": detection.class_name,
                        "confidence": f"{detection.class_confidence * 100:.1f}%",
                        "bbox": {"x": box_x,
                                 "y": box_y,
                                 "width": box_w,
                                 "height": box_h},
                        "networkConfig": {
                            "inputWidth": self.detector.network_width(),
                            "inputHeight": self.detector.network_height()
                        },
                        "detectionTime": finish}
            out_dict.append(detected)
        return out_dict

    def classify_full_image(self, image):
        out_dict = []
        saved_image = save_image(date=datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                                 path='data/tmp/', image=image)
        # noinspection PyBroadException
        try:
            im_width, im_height = Image.open(saved_image).size
        except Exception:
            os.remove(saved_image)
        start = time.time()
        detections = self.detector.detect(image=saved_image)
        finish = time.time() - start
        os.remove(saved_image)
        for detection in detections:
            class_name, class_confidence, bbox = detection
            x, y, w, h = bbox
            x_top, y_top, x_bot, y_bot = convert_points(im_width=im_width, im_height=im_height, x=x, y=y, w=w, h=h)
            detected = {"class": class_name,
                        "confidence": f"{class_confidence * 100:.1f}%",
                        "bbox": {"x_top": x_top,
                                 "y_top": y_top,
                                 "x_bot": x_bot,
                                 "y_bot": y_bot},
                        "centroid": {"x": x,
                                     "y": y,
                                     "width": w,
                                     "height": h},
                        "imageSize": {"width": im_width,
                                      "height": im_height},
                        "detectionTime": finish}
            out_dict.append(detected)
        return out_dict

    def return_boxes_on_image(self, image):
        saved_image = save_image(date=datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                                 path='data/tmp/', image=image)
        # noinspection PyBroadException
        try:
            im_width, im_height = Image.open(saved_image).size
        except Exception:
            os.remove(saved_image)
        detections = self.detector.detect(image=saved_image)
        font = ImageFont.truetype('data/font/Ubuntu-Medium.ttf', 14)
        im = Image.open(saved_image)
        img_format = im.format
        draw = ImageDraw.Draw(im)
        os.remove(saved_image)
        for detection in detections:
            col1 = random.choice([0, 50, 100, 150, 200, 255])
            col2 = random.choice([170, 200, 255])
            col3 = random.choice([0, 50, 100, 150, 200, 255])
            class_name, class_confidence, bbox = detection
            x, y, w, h = bbox
            x_top, y_top, x_bot, y_bot = convert_points(im_width=im_width, im_height=im_height, x=x, y=y, w=w, h=h)
            draw.rectangle((x_top, y_top, x_bot, y_bot), width=5, outline=(col1, col2, col3))
            draw.rectangle(xy=[(x_top + 5, y_top + 5), (x_top + 100, y_top + 25)], fill=(0, 0, 0))
            draw.text((x_top + 5, y_top + 5), f'{class_name}, {class_confidence * 100:.1f}%', fill=(255, 255, 255),
                      font=font, anchor='la')
        file_path = f'{saved_image}.{img_format}'
        im.save(f'{saved_image}.{img_format}')
        return file_path
