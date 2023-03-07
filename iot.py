import os
import numpy
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_sync
import RPi.GPIO as GPIO
import time


#SOURCE = 'data/images/bus.jpg'
WEIGHTS = '../smoking_r2.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False
PIR_PIN = 7
servoV_PIN = 16
servoH_PIN = 18
servoVPosition = 90  # 수직
servoHPosition = 90  # 수평

SERVO_MAX_DUTY = 12   # 서보의 최대(180도) 위치의 주기
SERVO_MIN_DUTY = 3    # 서보의 최소(0도) 위치의 주기
SPICLK = 11
SPIMISO = 9
SPIMOSI = 10
SPICS = 8
mq2_dpin = 26
mq2_apin = 0
allow_error = 10  # 오차


def servoMotor(pin, degree, t):
  GPIO.setmode(GPIO.BOARD)
  GPIO.setup(pin, GPIO.OUT)
  pwm = GPIO.PWM(pin, 50)
  pwm.start(3)
  time.sleep(t)
  pwm.ChangeDutyCycle(degree)
  time.sleep(t)
  pwm.stop()
  GPIO.cleanup(pin)

#port init


def init():
    GPIO.setwarnings(False)
    GPIO.cleanup()  # clean up at the end of your script
    GPIO.setmode(GPIO.BCM)  # to specify whilch pin numbering system
    # set up the SPI interface pins
    GPIO.setup(SPIMOSI, GPIO.OUT)
    GPIO.setup(SPIMISO, GPIO.IN)
    GPIO.setup(SPICLK, GPIO.OUT)
    GPIO.setup(SPICS, GPIO.OUT)
    GPIO.setup(mq2_dpin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

#read SPI data from MCP3008(or MCP3204) chip,8 possible adc's (0 thru 7)


def readadc(adcnum, clockpin, mosipin, misopin, cspin):
        if ((adcnum > 7) or (adcnum < 0)):
                return -1
        GPIO.output(cspin, True)

        GPIO.output(clockpin, False)  # start clock low
        GPIO.output(cspin, False)     # bring CS low

        commandout = adcnum
        commandout |= 0x18  # start bit + single-ended bit
        commandout <<= 3    # we only need to send 5 bits here
        for i in range(5):
            if (commandout & 0x80):
                  GPIO.output(mosipin, True)
            else:
                  GPIO.output(mosipin, False)
            commandout <<= 1
            GPIO.output(clockpin, True)
            GPIO.output(clockpin, False)

        adcout = 0
        # read in one empty bit, one null bit and 10 ADC bits
        for i in range(12):
            GPIO.output(clockpin, True)
            GPIO.output(clockpin, False)
            adcout <<= 1
            if (GPIO.input(misopin)):
                  adcout |= 0x1

        GPIO.output(cspin, True)

        adcout >>= 1       # first bit is 'null' so drop it
        return adcout


def detect(img):
    source, weights, imgsz = img, WEIGHTS, IMG_SIZE

    # Initialize
    device = select_device(DEVICE)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print('device:', device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors 
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once

    # Load image
    img0 = img
    assert img0 is not None, 'Image Not Found ' + source #except error

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3: #3차원이면  [3,10,200]
        img = img.unsqueeze(0) #0인 차원 하나 생성 [3,0,10,200]

    # Inference
    t0 = time_sync()
    pred = model(img, augment=AUGMENT)[0]
    print('pred shape:', pred.shape)

    # Apply NMS
    pred = non_max_suppression(
        pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

    # Process detections
    det = pred[0]
    print('det shape:', det.shape)

    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string

    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(
            img.shape[2:], det[:, :4], img0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)},"  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, #좌표 설정
                         color=colors[int(cls)], line_thickness=3)

        print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')
    # Stream results

    return s


def image():
    servoVPosition = 90  # 수직
    servoHPosition = 90  # 수평
    SERVO_MAX_DUTY = 12   # 서보의 최대(180도) 위치의 주기
    SERVO_MIN_DUTY = 3    # 서보의 최소(0도) 위치의 주기

    init()
    print("please wait...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture("http://192.168.0.6:8090/?action=stream")


    while True:
        ret, frame = cap.read()
        detected, _ = hog.detectMultiScale(frame)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        length = len(detected)
        #COlevel = readadc(mq7_apin, SPICLK, SPIMOSI, SPIMISO, SPICS)
        if length < 1:
            print("no")

        else:
            for (x, y, w, h) in detected:
                m_w = int(x+(w/2))  # 중앙점
                m_h = int(y+(h/2))  # 중앙점
                cv2.circle(frame, (m_w, m_h), 50, (255, 0, 0), 5)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255,
                                                          255), thickness=2)  # (161,14),(246,183)
                if m_h < height/2:  # 수직
                    if servoVPosition >= 5:
                        servoVPosition -= 1
                        degree = servoVPosition
                        duty = SERVO_MIN_DUTY + \
                            (degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
                        # 수직담당 서보모터 핀코드와 degree를 입력
                        servoMotor(servoV_PIN, duty, 1)

                elif m_h > (height/2-allow_error):
                    if servoVPosition <= 175:
                        servoVPosition += 1
                        degree = servoVPosition
                        duty = SERVO_MIN_DUTY + \
                            (degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
                        servoMotor(servoV_PIN, duty, 1)

                if m_w < width/2:  # 수
                    if servoHPosition >= 5:
                        servoHPosition -= 1
                        degree = servoHPosition
                        duty = SERVO_MIN_DUTY + \
                            (degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
                        servoMotor(servoH_PIN, duty, 1)

                elif m_w > (width/2-allow_error):
                    if servoHPosition <= 175:
                        servoHPosition += 1
                        degree = servoHPosition
                        duty = SERVO_MIN_DUTY + \
                            (degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
                        servoMotor(servoH_PIN, duty, 1)

                key = detect(frame)
                print("key : "+key)
            COlevel = readadc(mq2_apin, SPICLK, SPIMOSI, SPIMISO, SPICS)
            if "cigarette" and "smoke" in key or "cigarette" in key:  # cigarette이랑 smoke거나 cigarette이랑 실제 연기
                if GPIO.input(mq2_dpin):
                    print("No")
                else:
                    print("Current Gas AD vaule = " + str() + "%.2f" %
                          ((COlevel/1024.)*3.3)+" V")
                    print("Yes!!! this is TTS")
            else:
                print("I'm Sorry")

            cv2.imshow('walk image', frame)
            if cv2.waitKey(10) == 27:  # esc
                break


if __name__ == '__main__':
    image()
    GPIO.cleanup()
