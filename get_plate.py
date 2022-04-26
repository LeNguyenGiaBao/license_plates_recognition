import numpy as np 
import cv2
import time
import glob
from openvino.runtime import Core
from utils import * 

# Load the model 
ie = Core()
model = ie.read_model(model="./plate_detection_model/saved_model.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer_ir = next(iter(compiled_model.inputs))


def get_plate(image):
    processed_image, resized_image = preprocess_image(image)

    # Create inference request
    request = compiled_model.create_infer_request()
    request.infer({input_layer_ir.any_name: processed_image})
    model_output = request.get_output_tensor().data # Đầu ra của WPOD-NET  

    # Remove các chiều =1 của model_output
    model_output = np.squeeze(model_output)

    # Postprocessing
    lp_threshold=0.5
    predict_label, plate_image, plate_type = reconstruct(image, resized_image, model_output, lp_threshold) # plate_type: (1: dài: 2 vuông)

    if (len(plate_image)):
        plate = cv2.cvtColor(plate_image[0],cv2.COLOR_RGB2BGR)

        return plate 

    return None

if __name__=="__main__":
    dataset = glob.glob('./data/*.png')
    start_time = time.time()
    for i in dataset:
        img = cv2.imread(i)
        plate = get_plate(img)
        if plate is not None:
            print(plate.shape)
    end_time = time.time()
    print(end_time - start_time)