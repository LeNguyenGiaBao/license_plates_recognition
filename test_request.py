import requests
import time 
import glob
import os

url = "http://127.0.0.1:8400/plate_file/"
t0 = time.time()

for i in glob.glob(r'D:\ParkingAppData\20220710\CAM_BIEN_VAO\*.jpg'):
    file_name = os.path.split(i)[1]
    payload={'name_cam': ''}
    files=[
    ('image',(file_name,open(i,'rb'),'image/png'))
    ]
    headers = {}

    t1 = time.time()
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    t2 = time.time() - t1
    print(response.text, t2)
    
total_time = time.time() - t0
print(str(total_time) + 's')
print(len(glob.glob(r'D:\ParkingAppData\20220710\CAM_BIEN_VAO\*.jpg')))
print(str(total_time/len(glob.glob(r'D:\ParkingAppData\20220710\CAM_BIEN_VAO\*.jpg'))) + 's/img')