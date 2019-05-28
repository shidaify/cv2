#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[ ]:


img = cv2.imread('1.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)#找到关键点
img=cv2.drawKeypoints(gray,kp,img)#绘制关键点

cv2.imwrite(r"sift.jpg",img)

