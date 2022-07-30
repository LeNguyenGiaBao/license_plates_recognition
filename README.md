# license_plates_recognition

#### Update 22_07_17
- test with text detection:
    - no detection: 0.0580617875394309 s/img
    - has detection: 0.07414934519306778 s/img 
    - ->deviant: 0.016089 s/img
    - -> USE DETECTION
#### Update 22_05_07
- Change port to localhost:8400
- Test with only recognition model [paddle](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_en/inference_ppocr_en.md#1-lightweight-chinese-recognition-model-inference). 
- Inference time with 1 plate: 
    - Only recognition model: ~0.05s
    - Use PaddleOCR interface: ~0.07s


#### Update 22_04_29
- crop plate into 2 parts -> ocr time decreases 0.25s -> 0.05~0.1s (because ocr model doesn't detect text)
- OCR Time: 0.05 ~ 0.1s
- Total time in one image: ~0.2s
- Time by call API to server: ~0.25s (client and server in the same device , i5 8250, run on cpu)

#### Update 22_04_28
- add api, default run on localhost:8100
- add function to test on video. Change path of video and see the result

#### Update 22_04_27
- Plate detection from `openvino`, inference time: 0.05 second/ image
- Plate OCR from `PaddleOCR`, inference time (including plate detection): 0.28 second/ image

### Fail case:
- B -> 3: because glare

### API for C#
Use: ```python app.py```  
Url: ```http://127.0.0.1:8400/plate/```  
```
var client = new RestClient("http://0.0.0.0:8400/plate/");
client.Timeout = -1;
var request = new RestRequest(Method.POST);
request.AddParameter("name_cam", "");
request.AddFile("image", "path_of_image");
IRestResponse response = client.Execute(request);
Console.WriteLine(response.Content);
```
Input:
- name_cam: str
- image: file


### Result Format
- Success 
  ```
  {
    "code": 200,
    "plate_text": "12-B3 456.78",
    "msg": "success"
  }
  ```
  
- Fail
  ```
  {
    "code": 201,   # error
    "error_code": 0,
    "msg": "error message"
  }
  ```
