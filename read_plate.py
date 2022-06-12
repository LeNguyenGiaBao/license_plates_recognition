import cv2 
import time
import glob 
from openvino.runtime import Core
from paddleocr import PaddleOCR
from get_plate import get_plate

def read_plate(plate, ocr_model):
    # https://github.com/PaddlePaddle/PaddleOCR/blob/95c670faf6cf4551c841764cde43a4f4d9d5e634/paddleocr.py#L345
    # output = ocr_model.ocr(plate, cls=True, det=False)
    result = ''

    # Split plate_image into 2 parts (to skip paddle detector) 
    h, w, _ = plate.shape
    cropped_plate = plate[:h//2,:,:]
    # cv2.imwrite('plate_above.jpg', cropped_plate)
    plate_text = ocr_model.ocr(cropped_plate, cls=False, det=False)[0][0]
    plate_text = ''.join(e for e in plate_text if e.isalnum())
    result += plate_text

    cropped_plate = plate[h//2:,:,:]
    # cv2.imwrite('plate_below.jpg', cropped_plate)
    plate_text = ocr_model.ocr(cropped_plate, cls=False, det=False)[0][0]
    plate_text = ''.join(e for e in plate_text if e.isalnum())
    result += plate_text

    return result


if __name__=="__main__":
    # command for run OCR only: python3 tools/infer/predict_rec.py --image_dir="./" --rec_model_dir="./ch_PP-OCRv2_rec_infer/" --use_gpu=True
    # time: 0.0483s

    # Load the model 
    ie = Core()
    model = ie.read_model(model="./plate_detection_model/saved_model.xml")
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    input_layer_ir = next(iter(compiled_model.inputs))

    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
    # to switch the language model in order.
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det=False) # need to run only once to download and load model into memory
    print(ocr)
    time_0 = time.time()
    # for i in glob.glob('./private_data/*.png'):
    img = cv2.imread('./private_data/0028802154_134931_PLATE_7.png')
    # img = cv2.imread(i)
    plate = get_plate(img, compiled_model, input_layer_ir)
    if plate is not None:
        start_time = time.time()
        plate_text = read_plate(plate, ocr)
        end_time = time.time()  
        print(end_time - start_time) # 0.0659s, > 0.02s for load model from command
        print(plate_text)

    print(time.time() - time_0)





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




