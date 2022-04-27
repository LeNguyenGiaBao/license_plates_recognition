import cv2 
from openvino.runtime import Core
from paddleocr import PaddleOCR
from get_plate import get_plate
from read_plate import read_plate

ie = Core()
model = ie.read_model(model="./plate_detection_model/saved_model.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer_ir = next(iter(compiled_model.inputs))
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log = False) # need to run only once to download and load model into memory


video_path = './private_data/00000000044000201.mp4'
video = cv2.VideoCapture(video_path)

while video.isOpened():
    ret, frame = video.read()
    # index += 1
    # if index % 2 != 0:
    #     continue 
    # print(frame.shape)
    frame = frame[300:800, 700:1200]
    if ret:
        plate = get_plate(frame, compiled_model, input_layer_ir)
        if plate is not None:
            cv2.imwrite('test.jpg', plate)
            plate_text = read_plate(plate, ocr)
            cv2.putText(frame, plate_text, (100,100), fontFace=0, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('1', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cv2.destroyAllWindows()