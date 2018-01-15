'''
处理鼠标事件；
从而获得手写数字！
'''
import cv2;
import numpy as np
import time
import os

# 创建一个空帧，定义(700, 700, 3)画图区域
frame = np.zeros((600, 600, 3), np.uint8) 

last_measurement = current_measurement = np.array((0, 0), np.float32)

def OnMouseMove(event, x, y, flag, userdata):
    global frame, current_measurement, last_measurement
    if event == cv2.EVENT_LBUTTONDOWN:
        #last_measurement = np.array([[np.float32(x)], [np.float32(y)]]) # 当前测量
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]]) # 当前测量
        #print('鼠标左键点击事件！')
        #print('x:%d,y:%d'%(x,y),mousedown)
        #cv2.line(frame, (0, 0), (100, 100), (255, 0, 0)) # 蓝色线为测量值     
        
    if flag == cv2.EVENT_FLAG_LBUTTON: 
        #print('鼠标移动事件！')
        #print('x:%d,y:%d'%(x,y))
        last_measurement = current_measurement # 把当前测量存储为上一次测量
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]]) # 当前测量
        lmx, lmy = last_measurement[0], last_measurement[1] # 上一次测量坐标
        cmx, cmy = current_measurement[0], current_measurement[1] # 当前测量坐标
        #print('lmx:%f.1,lmy:%f.1,cmx:%f.1'%(lmx,lmy,cmx))
        cv2.line(frame, (lmx, lmy), (cmx, cmy), (255, 255, 255), thickness = 8) #输入数字    
        #print(str(event))
#print('start!')
# 窗口初始化
cv2.namedWindow("Input Number:")
#opencv采用setMouseCallback函数处理鼠标事件，具体事件必须由回调（事件）函数的第一个参数来处理，该参数确定触发事件的类型（点击、移动等）
cv2.setMouseCallback("Input Number:", OnMouseMove)
timestack = str(int(time.time()))[-6:-1]
key = 0
while key != ord('q'):
    cv2.imshow("Input Number:", frame)
    key = cv2.waitKey(1) & 0xFF
cv2.imwrite('number'+timestack + '.jpg',frame)
#cv2.destroyWindow('Input Number:')
print('number image has been stored and named "number' + timestack + '.jpg"')
cv2.destroyAllWindows()

if os.path.exists('./HandWritingFileName.txt'):
    shutil.rmtree('./HandWritingFileName.txt')
filename = open('./HandWritingFileName.txt', 'w')
filename.write('number'+timestack + '.jpg')
filename.close()
print('the image name has been stored in HnadWritingFileName.txt ')