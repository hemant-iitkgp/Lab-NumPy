import numpy as np
import sys
import time
file="book1.csv"
arr=np.genfromtxt(file,dtype=float,delimiter=None,skip_header=1)
mx=arr.max()
mn=arr.min ()
print("Maximum value element:",mx)
print("Minimum value element:",mn)


##########################

arr=arr.flatten()
arr.sort()
print(arr)

#################
arr=arr[::-1]
print(arr)

###########################
file2="book2.csv"
file3="book3.csv"

#############################

arr2=np.genfromtxt(file2,dtype=float,delimiter=None,skip_header=1)
arr3=np.genfromtxt(file3,dtype=float,delimiter=None,skip_header=1)

mean_arr=np.array([arr.mean(),arr2.mean(),arr3.mean()])
print(mean_arr)

#########################################

import cv2
from matplotlib import pyplot as plt

image=cv2.imread('a.png')
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
arr=np.array(image_rgb)

plt.imshow(arr)
plt.axis('off')
plt.show()

###############

gray_img=np.mean(image,axis=2,dtype=np.uint8)

arr=np.array(gray_img)
plt.imshow(arr,cmap='gray')
plt.axis('off')
plt.show()

##################################

tran_arr=arr.T
Z=np.dot(arr,tran_arr)

#####################################

gray_list=gray_img.tolist()

start_time=time.time()
transposed=np.array(gray_list).T
result=np.zeros_like(transposed)

# for row in range(len(gray_list)):
#     for col in range(len(transposed[0])):
#         for k in range(len(transposed)):
#             result[row][col]+=gray_list[row][k]*transposed[k][col]

elapse_time=time.time()-start_time
start_time_numpy=time.time()
result_numpy=np.dot(gray_img,gray_img.T)
elapse_numpy=time.time()-start_time_numpy

# print("elapsed time for NumPy:",elapse_numpy,"seconds")
# print("Elapsed time for simple approach:",elapse_time,"seconds")


##############################

histogram, bins=np.histogram(arr.flatten(),bins=256,range=(0,256))

plt.figure(figsize=(8,6))
plt.bar(bins[:-1],histogram,width=1,color='gray')
plt.title('Pixel Intensity Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

##################################


modified_image=gray_img.copy()
top_left=(100,40)
bottom_right=(200,70)
cv2.rectangle(modified_image,top_left,bottom_right,color=0,thickness=-1)

plt.imshow(modified_image,cmap='gray')
plt.axis('off')
plt.show()

################################

thresholds=[50,70,100,150]
binarized_images={}
for threshold in thresholds:
    binarized_image=np.where(arr>threshold,1,0)
    binarized_images[f'Z{threshold}']=binarized_image

for threshold, image in binarized_images.items():
    print(f'Binarized image with threshold {threshold}:')
    print(image)


##################################
import numpy as np
image=cv2.imread('a.png')
filter=np.array([[-1,-1,-1],
                 [0,0,0],
                 [1,1,1]])
filtered_image=cv2.filter2D(image,-1,filter)

plt.imshow(cv2.cvtColor(filtered_image,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()