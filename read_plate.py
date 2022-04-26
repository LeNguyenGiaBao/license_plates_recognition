import cv2 
import time
import glob 
from paddleocr import PaddleOCR
from plate_detect import get_plate

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

def read_plate(plate):
    # https://github.com/PaddlePaddle/PaddleOCR/blob/95c670faf6cf4551c841764cde43a4f4d9d5e634/paddleocr.py#L345
    result = ocr.ocr(plate, cls=True)
    
    return result 

if __name__=="__main__":
    dataset = glob.glob('./data/*.png')
    start_time = time.time()
    for i in dataset:
        img = cv2.imread(i)
        plate = get_plate(img)
        if plate is not None:
            plate_text = read_plate(plate)
            print("plate_text", type(plate_text))

    end_time = time.time()
    print('time:', end_time - start_time)  # 2.55 seconds for 9 images




