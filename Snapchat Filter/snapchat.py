import cv2
import numpy as np 
import pandas as pd
import random
import os


folder_name = "/Users/ayman/Desktop/AI_mafia/Snapchat Assignment/Modified_Images/"
if not os.path.exists(folder_name):
   os.makedirs(folder_name)


image_url = '/Users/ayman/Desktop/AI_mafia/Snapchat Assignment/Train/Jamie_Before.jpg'
# image_url = '/Users/ayman/Desktop/AI_mafia/Snapchat Assignment/Test/Before.png'
image = cv2.imread(image_url)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


eye_cascade = cv2.CascadeClassifier("/Users/ayman/Desktop/AI_mafia/Snapchat Assignment/Train/third-party/frontalEyes35x16.xml")  
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))                                  
sunglasses = cv2.imread('/Users/ayman/Desktop/AI_mafia/Snapchat Assignment/Train/glasses.png', cv2.IMREAD_UNCHANGED)
for (x, y, w, h) in eyes:
  resized_sunglasses = cv2.resize(sunglasses, (w, h))
  x_offset = x
  y_offset = y
  x_end = x_offset + resized_sunglasses.shape[1]
  y_end = y_offset + resized_sunglasses.shape[0]

  alpha_s = resized_sunglasses[:, :, 3] / 255.0
  alpha_l = 1.0 - alpha_s
  for c in range(0, 3):
    image[y_offset:y_end, x_offset:x_end, c] = (alpha_s * resized_sunglasses[:, :, c] + alpha_l * image[y_offset:y_end, x_offset:x_end, c])


nose_cascade = cv2.CascadeClassifier("/Users/ayman/Desktop/AI_mafia/Snapchat Assignment/Train/third-party/Nose18x15.xml")  
nose  = nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=13, minSize=(30, 30))
mustache = cv2.imread("/Users/ayman/Desktop/AI_mafia/Snapchat Assignment/Train/mustache.png", cv2.IMREAD_UNCHANGED)
for (x, y, w, h) in nose:
    resized_mustache = cv2.resize(mustache, (int(w * 1.3), int(h * 1)))
    x_offset = x - int(w * 0.1)
    y_offset = y + int(h * 0.55)
    x_end = x_offset + resized_mustache.shape[1]
    y_end = y_offset + resized_mustache.shape[0]

    alpha_m = resized_mustache[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_m
    for c in range(0, 3):
        image[y_offset:y_end, x_offset:x_end, c] = (alpha_m * resized_mustache[:, :, c] + alpha_l * image[y_offset:y_end, x_offset:x_end, c])


flat_image = image.reshape(-1,3)
df = pd.DataFrame(flat_image, columns=["R","G","B"])
csv_file_path = os.path.join(folder_name, f"modified_image{random.randrange(0,1000)}.csv")
df.to_csv(csv_file_path)
print(f"CSV file saved at: {csv_file_path}")


cv2.imshow('Modified Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
