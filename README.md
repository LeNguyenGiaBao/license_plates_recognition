# license_plates_recognition

#### Update 22_04_28
- add api, default run on localhost:8100
- add function to test on video. Change path of video and see the result

#### Update 22_04_27
- Plate detection from `openvino`, inference time: 0.05 second/ image
- Plate OCR from `PaddleOCR`, inference time (including plate detection): 0.28 second/ image

### Fail case:
- B -> 3: because glare


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
