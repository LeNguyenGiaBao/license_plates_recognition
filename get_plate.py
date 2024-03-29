import numpy as np 
import cv2
import math
import time
import glob
import os 
from datetime import date, datetime
from openvino.runtime import Core
from utils import * 

DETECT_THRESHOLD = 0.56 # IMPORTANT PARAM!!!
MIN_PLATE_DIAGONAL = 90 # IMPORTANT PARAM!!! 
IS_SAVE_PLATE = True

if IS_SAVE_PLATE:
    today = date.today()
    today = '20' + today.strftime('%y%m%d')
    dir_path = r'D:\ParkingAppData\{}\plate_image'.format(today)

    if not os.path.exists(dir_path): 
        try:
            os.mkdir(dir_path)
        except:
            pass


def get_plate(image, compiled_model, input_layer_ir):
    processed_image, resized_image = preprocess_image(image)

    # Create inference request
    request = compiled_model.create_infer_request()
    request.infer({input_layer_ir.any_name: processed_image})
    model_output = request.get_output_tensor().data # Đầu ra của WPOD-NET  

    # Remove các chiều =1 của model_output
    model_output = np.squeeze(model_output)

    # Postprocessing
    lp_threshold=DETECT_THRESHOLD
    predict_label, plate_image, plate_type = reconstruct(image, resized_image, model_output, lp_threshold) # plate_type: (1: dài: 2 vuông)

    if (len(plate_image)):
        plate = plate_image[0]
        plate_h, plate_w = plate.shape[:2]
        plate_diagonal = math.sqrt(plate_h**2+plate_w**2)
        if plate_diagonal < MIN_PLATE_DIAGONAL:
            return None 
            
        plate = cv2.cvtColor(plate_image[0],cv2.COLOR_RGB2BGR)

        if IS_SAVE_PLATE:
            now = datetime.now() 
            now = now.strftime("%H_%M_%S_%f")
            path = os.path.join(dir_path, now+'.jpg')
            cv2.imwrite(path, plate)
            
        return plate 

    return None

if __name__=="__main__":
    # Load the model 
    ie = Core()
    model = ie.read_model(model="./plate_detection_model/saved_model.xml")
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    input_layer_ir = next(iter(compiled_model.inputs))

    dataset = glob.glob('./private_data/*.png')
    start_time = time.time()
    for i in dataset:
        img = cv2.imread(i)
        plate = get_plate(img, compiled_model, input_layer_ir)
        if plate is not None:
            print(plate.shape)
    end_time = time.time()
    print(end_time - start_time)