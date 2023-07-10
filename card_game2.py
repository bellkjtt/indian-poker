# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html 
# Usage example:  python3 object_detection_yolo.py --video=run.mp4 
#python3 object_detection_yolo.py --image=bird.jpg 

import cv2 as cv
import sys 
import numpy as np 
import os.path
import RPi.GPIO as GPIO
import time
import random
import threading
from bluetooth import *

GPIO.setmode(GPIO.BCM)
GPIO.setup(27,GPIO.OUT)
GPIO.setup(18,GPIO.OUT)
GPIO.setup(17,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(15,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(14,GPIO.IN,pull_up_down=GPIO.PUD_UP)
 
# Initialize the parameters 
confThreshold = 0.5  #Confidence threshold 
nmsThreshold = 0.4   #Non-maximum suppression threshold 
inpWidth = 64       #Width of network's input image 
inpHeight = 64      #Height of network's input image 

 
# Load names of classes 
classesFile = "cards.names" 
classes = None 
with open(classesFile, 'rt') as f: 
    classes = f.read().rstrip('\n').split('\n') 

 
# Give the configuration and weight files for the model and load the network using them. 
modelConfiguration = "yolocards.cfg" 
modelWeights = "yolocards_608.weights"

 
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights) 
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) 
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) 

 
# Get the names of the output layers 
def getOutputsNames(net): 
# Get the names of all the layers in the network 
    layersNames = net.getLayerNames() 
# Get the names of the output layers, i.e. the layers with unconnected outputs 
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()] 

 
# Draw the predicted bounding box 
def drawPred(classId, conf, left, top, right, bottom):
     global label
     # Draw a bounding box. 
     cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3) 
      
     label = '%.2f' % conf
          
    # Get the label for the class name and its confidence 
     if classes:
         assert(classId < len(classes)) 
         label = '%s:%s' % (classes[classId], label)
 
 
     #Display the label at the top of the bounding box 
     labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
     top = max(top, labelSize[1])
     cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED) 
     cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
 
 
     #Display the label at the top of the bounding box 
 
  # Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
     frameHeight = frame.shape[0] 
     frameWidth = frame.shape[1]

     # Scan through all the bounding boxes output from the network and keep only the 
     # ones with high confidence scores. Assign the box's class label as the class with the highest score. 
     classIds = [] 
     confidences = [] 
     boxes = [] 
     for out in outs: 
         for detection in out: 
             scores = detection[5:] 
             classId = np.argmax(scores) 
             confidence = scores[classId] 
             if confidence > confThreshold:
                 center_x = int(detection[0] * frameWidth)
                 center_y = int(detection[1] * frameHeight) 
                 width = int(detection[2] * frameWidth) 
                 height = int(detection[3] * frameHeight) 
                 left = int(center_x - width / 2)
                 top = int(center_y - height / 2)
                 classIds.append(classId) 
                 confidences.append(float(confidence)) 
                 boxes.append([left, top, width, height]) 
 
 
     # Perform non maximum suppression to eliminate redundant overlapping boxes with 
     # lower confidences. 
     indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold) 
     for i in indices: 
         i = i[0] 
         box = boxes[i] 
         left = box[0] 
         top = box[1] 
         width = box[2] 
         height = box[3]
         drawPred(classIds[i], confidences[i], left, top, left + width, top + height) 
            
def calculate(label3, count):
    global give_up,Token,Token_have,Token_count
    Token_count +=1
    count = Player_Token - Token_Throw
    give_up = 0
    if count == 0:
        give_up = 3
    elif any(label3[0] in s for s in c[0:4]):
        Token = random.choice([count+1,count+1,count+1,count+2,count+3,count+3,Token_have[Token_count]])
        Token_have.append(Token_have[Token_count] - Token)
        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
        print('Token ={}'.format(Token))
        print('have ={}'.format(Token_have[Token_count+1]))
        if Token_have[Token_count+1] <= 0:
            Token = Token_have[Token_count]
            print('Token ={}'.format(Token))
            give_up = 2
    elif any(label3[0] in s for s in c[4:8]):
            Token = random.choice([count+1,count+1,count+1,count+2,count+3,count+3,Token_have[Token_count]])
            if Token_have[Token_count] <= count+1:
                    Token = Token_have[Token_count]
                    Token_have.append(Token_have[Token_count] - Token)
                    print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                    print('Token ={}'.format(Token))
                    print('have ={}'.format(Token_have[Token_count+1]))
                    if Token_have[Token_count+1] <= 0:
                        Token = Token_have[Token_count]
                        print('Token ={}'.format(Token))
                        give_up = 2
            elif Token_have[Token_count] > count+1:
                    give_up = random.choice([0,0,0,0,1,0,0,0,0,2,2])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0:
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
                    elif give_up == 2:
                        Token = count
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                            
    elif any(label3[0] in s for s in c[8:12]):
            Token = random.choice([count+1,count+1,count+1,count+1,count+1,count+2,Token_have[Token_count]])
            if Token_have[Token_count] <= count+1:
                    Token = Token_have[Token_count]
                    Token_have.append(Token_have[Token_count] - Token)
                    print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                    print('Token ={}'.format(Token))
                    print('have ={}'.format(Token_have[Token_count+1]))
                    if Token_have[Token_count+1] <= 0:
                        Token = Token_have[Token_count]
                        print('Token ={}'.format(Token))
                        give_up = 2
            elif Token_have[Token_count] > count+1:
                    give_up = random.choice([0,0,0,0,0,1,0,2,2,2,2])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0:
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
                    elif give_up == 2:
                        Token = count
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
    elif any(label3[0] in s for s in c[12:16]):
            Token = random.choice([count+1,count+1,count+1,count+1,count+1,count+1,count+2,count+2,count+2,count+3])
            if Token_have[Token_count] <= count+1:
                    give_up = random.choice([0,0,0,0,0,0,0,1,1,2,2])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0 or 2:
                        Token = Token_have[Token_count]
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
            elif Token_have[Token_count] > count+1:
                    give_up = random.choice([0,0,0,0,1,0,0,0,2,2,2])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0:
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
                    elif give_up == 2:
                        Token = count
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
    elif any(label3[0] in s for s in c[16:20]):
            Token = random.choice([count+1,count+1,count+1,count+1,count+2,count+2,count+2,count+3,count+3])
            if Token_have[Token_count] <= count+1:
                    give_up = random.choice([0,0,0,0,1,1,1,2,0,2,2])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0 or 2:
                        Token = Token_have[Token_count]
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
            elif Token_have[Token_count] > count+1:
                    give_up = random.choice([0,0,0,0,0,2,2,2,1,2,2,0])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0:
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
                    elif give_up == 2:
                        Token = count
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
    elif any(label3[0] in s for s in c[20:24]):
            Token = random.choice([count+1,count+1,count+1,count+1,count+1,count+2,count+2,count+3])
            if Token_have[Token_count] <= count+1:
                    give_up = random.choice([0,0,0,0,1,1,1,1,0,1,2,2])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0 or 2:
                        Token = Token_have[Token_count]
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
            elif Token_have[Token_count] > count+1:
                    give_up = random.choice([0,0,0,0,1,1,1,0,2,1,2,2])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0:
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
                    elif give_up == 2:
                        Token = count
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
    elif any(label3[0] in s for s in c[24:28]):
            Token = random.choice([count+1,count+1,count+1,count+1,count+1,count+2,count+2,count+3])
            if Token_have[Token_count] <= count+1:
                    give_up = random.choice([0,0,1,1,1,1,1,1,1,1,1,1])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0 or 2:
                        Token = Token_have[Token_count]
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
            elif Token_have[Token_count] > count+1:
                    give_up = random.choice([0,0,1,1,1,1,1,2,2,2,2,2])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0:
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
                    elif give_up == 2:
                        Token = count
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        
    elif any(label3[0] in s for s in c[28:32]):
            Token = random.choice([count+1,count+1,count+1,count+1,count+1,count+2,count+2,count+3])
            if Token_have[Token_count] <= count+1:
                    give_up = random.choice([0,1,0,0,1,1,1,1,1,1,1,1])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0 or 2:
                        Token = Token_have[Token_count]
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
            elif Token_have[Token_count] > count+1:
                    give_up = random.choice([0,1,1,1,1,1,1,1,2,2,2,2])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0:
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
                    elif give_up == 2:
                        Token = count
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
    elif any(label3[0] in s for s in c[32:36]):
            Token = random.choice([count+1,count+1,count+1,count+1,count+1,count+2,count+2,count+3])
            if Token_have[Token_count] <= count+1:
                    give_up = random.choice([0,1,1,1,1,1,1,1,0,0,1,2])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0 or 2:
                        Token = Token_have[Token_count]
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
            elif Token_have[Token_count] > count+1:
                    give_up = random.choice([0,2,1,2,1,1,1,1,2,1,2,2])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0:
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
                    elif give_up == 2:
                        Token = count
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
    elif any(label3[0] in s for s in c[36:40]):
            Token = random.choice([count+1,Token_have[Token_count],Token_have[Token_count]])
            if Token_have[Token_count] <= count+1:
                    give_up = random.choice([1,1,1,1,1,1,1,1,1,1,1,0])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0 or 2:
                        Token = Token_have[Token_count]
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            give_up = 2
            elif Token_have[Token_count] > count+1:
                    give_up = random.choice([1,1,1,1,1,1,1,0,0,0,0,0])
                    if give_up == 1:
                        print('give_up')
                    elif give_up == 0:
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
                        if Token_have[Token_count+1] <= 0:
                            Token = Token_have[Token_count]
                            print('Token ={}'.format(Token))
                            give_up = 2
                    elif give_up == 2:
                        Token = count
                        Token_have.append(Token_have[Token_count] - Token)
                        print('Token calculate = {} - {}'.format((Token_have[Token_count]),Token))
                        print('Token ={}'.format(Token))
                        print('have ={}'.format(Token_have[Token_count+1]))
 
 # Process inputs 
 
cap = cv.VideoCapture(0)
cap.set(3,960)
cap.set(4,640)
counter = 0
label6 = []
label8 = []
GPIO.output(18,GPIO.LOW)
GPIO.output(27,GPIO.LOW)

target_name = "H-C-2010-06-01"   # target device name
target_address = '00:18:E4:34:D1:8A'
port = 1         # RFCOMM port
count2 =0
frame_rate = 90
prev = 0
a = ['A','2','3','4','5','6','7','8','9','10']
b = ['h','d','j','c']
c = [x+y for x in a for y in b]
Token_have = [int(input('Enter the number of Token robot have: '))-1]
Token_count =-1
Token_Throw = 0
Player_Token = 0

try:
    sock=BluetoothSocket( RFCOMM )
    sock.connect((target_address, port))
    sock.send(str(1).encode())

 
# Get the video writer initialized to save the output video 
    while cv.waitKey(1) < 0:
             time_elapsed = time.time() - prev
             # get frame from the video 
             hasFrame, frame = cap.read()
             frame = cv.GaussianBlur(frame,(9,9),0)

             
             # Stop the program if reached end of video 
             if not hasFrame: 
                 print("Done processing !!!") 
                 cv.waitKey(3000) 
                 # Release device 
                 cap.release() 
                 break
                
             try: 
                    if classes:
                     global label3
                     label2 = label[:3]
                     label3 = label2.split(':')
                     label4 = label[3:]
                     label5 = label4.split(':')
                     print(label5[-1])
                     GPIO.output(18,GPIO.HIGH)
                     time.sleep(5)
                     GPIO.output(18,GPIO.LOW)
                     break

             except:
                pass
         
         
             # Create a 4D blob from a frame. 
             blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False) 

         
             # Sets the input to the network 
             net.setInput(blob) 
         
         
             # Runs the forward pass to get output of the output layers 
             outs = net.forward(getOutputsNames(net)) 
         
         
             # Remove the bounding boxes with low confidence 
             postprocess(frame, outs)
             # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    
    print('Now count is {}.'.format(count2))
    while True:
        if not GPIO.input(17):
            count2+=1
            if count2 > 10:
                count2 = 10
            print('Now count is {}.'.format(count2))
            time.sleep(0.5)
        elif not GPIO.input(15):
            count2-=1
            if count2 <0:
                count2 = 0
            print('Now count is {}.'.format(count2))
            time.sleep(0.5)
        elif not GPIO.input(14):
            print('player Final Token is {}.'.format(count2))
            Player_Token = Player_Token + count2
            if count2 == 0:
                    print(label3[0])
                    GPIO.output(18,GPIO.HIGH)
                    time.sleep(0.3)
                    GPIO.output(18,GPIO.LOW)
                    time.sleep(0.3)
                    GPIO.output(27,GPIO.HIGH)
                    time.sleep(0.5)
                    GPIO.output(27,GPIO.LOW)
                    time.sleep(0.3)
                    GPIO.output(18,GPIO.HIGH)
                    time.sleep(0.3)
                    GPIO.output(18,GPIO.LOW)
                    time.sleep(0.3)
                    GPIO.output(27,GPIO.HIGH)
                    time.sleep(0.3)
                    GPIO.output(27,GPIO.LOW)
                    time.sleep(0.3)
                    GPIO.output(18,GPIO.HIGH)
                    time.sleep(0.3)
                    GPIO.output(18,GPIO.LOW)
                    time.sleep(0.3)
                    GPIO.output(27,GPIO.HIGH)
                    time.sleep(0.5)
                    GPIO.output(27,GPIO.LOW)
                    time.sleep(0.3)
                    GPIO.output(18,GPIO.HIGH)
                    time.sleep(0.3)
                    GPIO.output(18,GPIO.LOW)
                    time.sleep(0.3)
                    GPIO.output(27,GPIO.HIGH)
                    time.sleep(0.3)
                    GPIO.output(27,GPIO.LOW)
                    time.sleep(0.3)
                    GPIO.cleanup()
                    break
            calculate(label3[0],count2)
            
            try:
                if give_up == 1:
                    GPIO.output(27,GPIO.HIGH)
                    time.sleep(5)
                    GPIO.output(27,GPIO.LOW)
                    GPIO.cleanup()
                    print(label3[0])
                    break
                elif give_up ==2:
                    print('call')
                    sock.send(str(Token).encode())
                    print(str(Token).encode())
                    GPIO.output(18,GPIO.HIGH)
                    time.sleep(5)
                    GPIO.output(18,GPIO.LOW)
                    GPIO.cleanup()
                    break
                elif give_up ==3:
                    print('call')
                    GPIO.output(18,GPIO.HIGH)
                    time.sleep(5)
                    GPIO.output(18,GPIO.LOW)
                    GPIO.cleanup()
                    break
                elif give_up == 0:
                    sock.send(str(Token).encode())
                    print(str(Token).encode())
                    print('Now count is {}.'.format(count2))
                    Token_Throw = Token_Throw+Token
                    time.sleep(3)
                    continue

            except KeyboardInterrupt:
                print("disconnected")
                sock.close()
                print("all done")
                GPIO.cleanup()

                
except btcommon.BluetoothError as err:
    print('An error occurred : %s ' % err)
    print(label)
    pass

