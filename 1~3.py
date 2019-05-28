#!/usr/bin/env python
# coding: utf-8

# In[29]:


import math
import numpy as np
import cv2


# In[30]:


def gauss(size,sigma2):
    """生成任意尺寸的模板,第一位为尺寸，第二位为方差"""
    k = size//2#尺寸的一半
    row = []
    for i in range(size):
        x = pow(i-k,2)
        line = []
        for j in range(size):
            y = pow(j-k,2)
            gaosi = (math.exp(-(x + y)/(2 * sigma2)))/(2 * math.pi * sigma2)
            line.append(gaosi)
        row.append(line)
    s = np.sum(np.sum(row,axis = 1),axis = 0)#求和
    row = np.array(row)
    row /= s
    return row


# In[31]:


def convolution(size,gauss_,img):
    """尺寸，卷积核，图像路径"""
    #img = cv2.imread(path)
    #print(img)
    #img_gray = cv2.imread(path,0)
    
    x,y = img.shape#获取图像信息
    gauss_ = np.rot90(gauss_, 2)#旋转180度
    new_image = np.zeros((x-2*(size//2),y-2*(size//2)),dtype=int)
    new =[]
    for pos_x in range(size//2,x-size//2):
        new_line = []
        for pos_y in range(size//2,y-size//2):
            new0 = int((img[pos_x-size//2:pos_x+size//2+1,pos_y-size//2:pos_y+size//2+1]*gauss_).sum())
            new_line.append(new0)
            #print(int(s),round(s))
        new.append(new_line)
    new_image = np.array(new)
    #print(new_image)
    return new_image


# In[32]:


def sobel():
    """计算梯度强度及其方向"""
    #sobel算子
    S_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    S_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
    
    G_x = convolution(3,S_x,new_image)
    G_y = convolution(3,S_y,new_image)
    #print(G_x,G_y)
    G = np.sqrt(G_x**2 + G_y**2)
    theta =[]
    G_temp = G_x/G_y
    #print(G_temp[0])
    for line in G_temp:
        theta.append([math.atan(row)/math.pi*180 for row in line])#化成角度制
    #print(theta[0])
    cv2.imwrite(r'G.jpg', G)
    return G,G_temp,theta
#G,tan_G,theta = sobel()


# In[33]:


def where(d):
    """计算方向,分为0，1，2，3四个区域其中0为90～45，1为45～0，2为0～-45，3为-45～-90"""
    if d <= 90 and d > 45:
        return 0
    elif d >= 0 and d < 45:   
        return 1
    elif d >= -45 and d < 0:
        return 2
    elif d >= -90 and d < -45:
        return 3
    
def direction(img,Theta,tan_G):
    """非最大化抑制,theta为角度，tan_G为tan值"""
    (x,y) = img.shape
    new_img = np.zeros((x-2,y-2),dtype=int)
    for img_x in range(1,x-1):
        for img_y in range(1,y-1):#定义上下左右等所表示的像素点的位置
            up = img[img_x-1,img_y]
            down = img[img_x+1,img_y]
            left = img[img_x,img_y-1]
            right = img[img_x,img_y+1]
            up_left = img[img_x-1,img_y-1]
            up_right = img[img_x-1,img_y+1]
            down_left = img[img_x+1,img_y-1]
            down_right = img[img_x+1,img_y+1]
            tan = tan_G[img_x,img_y]
            tan = abs(tan)
            #合成同一方向上的临近两点
            if where(Theta[img_x,img_y]) == 0:
                tan = 1/tan
                G1 = up*(1-tan) + up_right*tan
                G2 = down*(1-tan) + down_left*tan
            elif where(Theta[img_x,img_y]) == 1:
                G1 = right*(1-tan) + up_right*tan
                G2 = left*(1-tan) + down_left*tan
            elif where(Theta[img_x,img_y]) == 2:
                G1 = right*(1-tan) + down_right*tan
                G2 = left*(1-tan) + up_left*tan
            elif where(Theta[img_x,img_y]) == 3:
                tan = 1/tan
                G1 = down*(1-tan) + down_right*tan
                G2 = up*(1-tan) + up_left*tan
            #非最大化抑制
            if img[img_x,img_y] >= G1 and img[img_x,img_y] >= G2:
                #new_img[img_x-1,img_y-1] = img[img_x,img_y]
                if img[img_x,img_y] >= highThreshould:#双阈值检测
                    new_img[img_x-1,img_y-1] = 255
                elif img[img_x,img_y] >= lowThreshould:
                    new_img[img_x-1,img_y-1] = lowThreshould
                else:
                    new_img[img_x-1,img_y-1] = 0
            else:
                new_img[img_x-1,img_y-1] = 0
    cv2.imwrite(r"G2.jpg", new_img)
    return new_img


# In[34]:


def is_Strong(x,y,img_is_S):
    """判断周围是否有强边缘点"""
    up = img_is_S[x-1,y]
    down = img_is_S[x+1,y]
    left = img_is_S[x,y-1]
    right = img_is_S[x,y+1]
    up_left = img_is_S[x-1,y-1]
    up_right = img_is_S[x-1,y+1]
    down_left = img_is_S[x+1,y-1]
    down_right = img_is_S[x+1,y+1]
    if up==255 or down==255 or left==255 or right==255 or up_left==255 or up_right==255 or down_left==255 or down_right==255:
        return 1
    else:
        return 0
    
def Suppression(img_S):
    """抑制孤立弱边缘"""
    for x in range(1,len(img_S)-1):
        for y in range(1,len(img_S[x])-1):
            if img_S[x,y] == lowThreshould:
                if is_Strong(x,y,img_S)==1:
                    img_S[x,y]=255
                else:
                    img_S[x,y]=0
    cv2.imwrite(r"G3.jpg", img_S)
#Suppression(img_d)


# In[35]:


def main():
    img1 = cv2.imread(r'1.jpg',0)#获取灰度图
    new_image = convolution(3,a,img1)
    cv2.imwrite(r"2.jpg", new_image)
   # new_image = cv2.imread(r'1.jpg',0)
    print(new_image)
    G,tan_G,theta = sobel()
    lowThreshould = 50
    highThreshould = 120
    theta = np.array(theta)
    img_d = direction(G,theta,tan_G)
    Suppression(img_d)
    
if __name__ == '__main__':
    main()

