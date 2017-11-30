'''
边缘检测，裁剪
'''

#  -*- coding: utf-8 -*
import cv2
import numpy as np
import os

#import the image
img = cv2.imread('number.jpg',1)
#cv2.imshow('img',img)
#转化为灰度图
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
#高斯平滑除噪
#img_blur = cv2.GaussianBlur(img,(5,5),0)
img_blur = cv2.medianBlur(img_gray,5)
#cv2.imshow('Gaussian',img_blur)
#canny算子 边缘检测
img_canny = cv2.Canny(img_blur,200,100)
#cv2.imshow('canny',img_canny)
#二值化处理
_, img_bin = cv2.threshold(img_canny, 80, 255, cv2.THRESH_BINARY )
#cv2.imshow('bin',img_bin)
#积分运算
imgI = cv2.integral(img_bin)

#######分块
#定义分块的尺寸
(xh,yw) = img_gray.shape
p = int(xh / 10) #高度方向上等分的快数
q = int(yw / 10) #宽度方向上等分的快数
#分块矩阵
sat = np.arange(p*q).reshape(p,q)
#获取原图的尺寸

#计算每块的宽和高；
w = int(yw / q)
h = int(xh / p)
if w <= 5:
    print('the image is too small to split!')
    os._exit(0)
#print(w,h)
#合并块
sated = np.ones((p,q))

p = range(p)
q = range(q)

# 计算各块的能量
sat[0][0] = imgI[h-1][w-1]
for n in q[1:]: #先计算第0行的能量
    sat[0][n] = imgI[h-1][w * (n+1) -1] - imgI[h-1][w * n -1]
for m in p[1:]: #计算第0列的能量
    sat[m][0] = imgI[h * (m+1) - 1][w-1] - imgI[h * m - 1][w-1]
for m in p[1:]: #计算其余的能量
    for n in q[1:]:
        sat[m][n] = imgI[h * (m+1) - 1][w * (n+1) -1]  - imgI[h * (m+1) - 1][w * n -1] -imgI[h * m - 1][w * (n+1) -1] + imgI[h * m - 1][w * n -1] 

#print(sat)
#计算各块的能量密度
sat = sat / (w * h) 

#选出能量密度较高的块
print('to draw:')
threshold1 = 10

##8邻域搜索算法合并区域
def eightSearch(sated, m, n,mleft,ntop,mright,nbottom):
    #sat_search = [sat[m-1][n-1],sat[m-1][n],sat[m-1][n+1],sat[m][n-1],sat[m][n+1],sat[m+1][n-1],sat[m+1][n],sat[m+1][n+1]]
    sat_search = [(m-1,n-1),(m-1,n),(m-1,n+1),(m,n-1),(m,n+1),(m+1,n-1),(m+1,n),(m+1,n+1)]
    for sati in sat_search:
        s0 = sati[0]
        s1 = sati[1]
        
        if sated[s0][s1] != -1:
            left = s1 * w -1
            top = s0 * h -1
            right = s1 * w + w -1
            bottom = s0 * h + h -1
            sated[s0][s1] = -1
            if sat[s0][s1] > 10:
                #记录边界
                if left < mleft:
                    mleft = left
                elif right > mright:
                    mright = right
                if top < ntop:
                    ntop = top
                elif bottom > nbottom:
                    nbottom = bottom
                #循环
                sated,mleft,ntop,mright,nbottom = eightSearch(sated,s0,s1,mleft,ntop,mright,nbottom)
    return sated,mleft,ntop,mright,nbottom

##保存框选出的区域
#@param img 原图
#@param left, right ,top,bottom 裁切区域的上下左右坐标
#@param pad 是否添加边（添加20%的黑边）
#@param name 保存图片的名字,默认为None,则不保存
def saveRect(img, mleft, ntop, mright, nbottom, pad = True, name = None):
    subimg = img[ntop:nbottom, mleft:mright]
    if pad is False:
        if name is not None:
            cv2.imwrite(name,subimg)
        return subimg
    else:
        subimgshape = subimg.shape
        addpad =(2*int(0.1 * subimgshape[0]),2*int(0.1 * subimgshape[1]))
        frame = np.zeros((addpad[0] + subimgshape[0], addpad[1] + subimgshape[1], 3), np.uint8)
        frame[int(addpad[0]/2):(int(addpad[0]/2)+subimgshape[0]),int(addpad[1]/2):(int(addpad[1]/2)+subimgshape[1])] = subimg
        if name is not None:
            cv2.imwrite(name,frame)
        return frame #返回裁切结果
        
count = 0;#图中数字计数 

for m in p[1:-2]: #因为8邻域，所以排除
    for n in q[1:-2]:
        if sated[m][n] != -1 and sat[m][n]> 1:
            print('has number!')
            sated[m][n] = -1
            sated,mleft,ntop,mright,nbottom = eightSearch(sated,m,n,n*w-1,m*h-1,n*w-1+w,m*h-1+h)
            #cv2.rectangle(img,(mleft,ntop),(mright,nbottom),(0,0,255),1)
            saimg = saveRect(img,mleft,ntop,mright,nbottom)
            cv2.imshow('subimg' + str(count), saimg)
            res = cv2.resize(saimg,(28, 28), interpolation = cv2.INTER_AREA )
            cv2.imwrite('./img/num1647' + str(count) + '.jpg',res)
            count = count + 1

cv2.waitKey()
cv2.destroyAllWindows()
