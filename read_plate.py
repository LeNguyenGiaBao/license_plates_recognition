import cv2 
import time
import glob 
from openvino.runtime import Core
from paddleocr import PaddleOCR
from get_plate import get_plate


def read_plate(plate, ocr_model):
    # https://github.com/PaddlePaddle/PaddleOCR/blob/95c670faf6cf4551c841764cde43a4f4d9d5e634/paddleocr.py#L345
    output = ocr_model.ocr(plate, cls=True)
    result = ''

    for line in output:
        bbox = line[0]
        plate_text, plate_score = line[1]
        result += " " + plate_text

    return result[1:]

if __name__=="__main__":
    # Load the model 
    ie = Core()
    model = ie.read_model(model="./plate_detection_model/saved_model.xml")
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    input_layer_ir = next(iter(compiled_model.inputs))

    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
    # to switch the language model in order.
    ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

    img = cv2.imread('./private_data/0028802154_134931_PLATE_6.png')
    plate = get_plate(img, compiled_model, input_layer_ir)
    if plate is not None:
        plate_text = read_plate(plate, ocr)
        print(plate_text)




    # dataset = glob.glob('./private_data/*.png')
    # start_time = time.time()
    # for i in dataset:
    #     img = cv2.imread(i)
    #     plate = get_plate(img)
    #     if plate is not None:
    #         plate_text = read_plate(plate)
    #         print("plate_text", type(plate_text))

    # end_time = time.time()
    # print('time:', end_time - start_time)  # 2.55 seconds for 9 images




