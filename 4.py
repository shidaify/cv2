#!/usr/bin/env python
# coding: utf-8

# In[9]:


import math
import numpy as np
import cv2


# In[10]:


def gauss(size,sigma2):
    """生成任意尺寸的模板,第一位为尺寸，第二位为方差"""
    k = size//2
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


# In[11]:


def convolution(size,gauss_,img):
    """尺寸，卷积核，图像路径，"""
    x,y = img.shape[:2]
    gauss_ = np.rot90(gauss_, 2)#旋转180度
    new_image = np.zeros((x-2*(size//2),y-2*(size//2)),dtype=int)
    new =[]
    for pos_x in range(size//2,x-size//2):
        new_line = []
        for pos_y in range(size//2,y-size//2):
                new0 = (img[pos_x-size//2:pos_x+size//2+1,pos_y-size//2:pos_y+size//2+1]*gauss_).sum()
                new_line.append(new0)
        new.append(new_line)
    new_image = np.array(new)
    return new_image


# In[12]:


def harris_CRF(jiao,size,k):
    """计算角点响应值"""
    S_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    S_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
    I_x = convolution(3,S_x,jiao)
    cv2.imwrite(r"I_x.jpg", I_x)
    I_y = convolution(3,S_y,jiao)
    cv2.imwrite(r"I_y.jpg", I_y)
    ga = gauss(size,1)
    m1 = convolution(size,ga,I_x**2)
    cv2.imwrite(r"I_x2.jpg", m1)
    m2 = convolution(size,ga,I_x*I_y)
    cv2.imwrite(r"I_xy.jpg", m2)
    m4 = convolution(size,ga,I_y**2)
    CRF = (m1*m4 - m2**2)- k*((m1+m4)**2)
    return CRF


# In[13]:


def max_down(img,size):
    """非最大值抑制"""
    x,y = img.shape
    #print(img.max())
    find=[]
    for img_x in range(size//2,x-size//2):
        for img_y in range(size//2,y-size//2):
            flag = 0
            for i in range(-size//2,size//2+1):
                for j in range(-size//2,size//2+1):
                    if img[img_x+i,img_y+j] > img[img_x,img_y] or img[img_x,img_y] < img.max()*0.01:
                        flag = 1
                        img[img_x,img_y] = 0
                        break
                else:#跳出两个for循环
                    continue
                break
            if flag == 0:
                find.append([img_y,img_x])
    cv2.imwrite(r"crf2.jpg", img)
    return find


# In[21]:


def paint(img,find):
    """img为原图""" 
    for i in find:
        i = [p+4 for p in i]
        new_i = tuple(i)
        #print(new_i)
        cv2.circle(img,new_i,10,(0, 0, 255),0)
    cv2.imwrite(r"paint.jpg", img)    


# In[22]:


def main():
    jiao = cv2.imread(r'jiao2.png',0)
    crf = harris_CRF(jiao,5,0.04)
    find=max_down(crf,7)
    #print(find)
    yuantu = cv2.imread(r'jiao2.png')
    paint(yuantu,find)
    
if __name__ == '__main__':
    main()

