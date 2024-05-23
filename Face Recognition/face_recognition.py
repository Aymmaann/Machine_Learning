import cv2 
import numpy as np 
import os

# KNN Algorithm
def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test,ix)
        dist.append([d,iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:,-1]

    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]


# Initialize webcam
cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier("/Users/ayman/Desktop/AI_mafia/Face Recognition/haarcascade_frontalface_alt.xml")

# Data Preperation
class_id = 0
names = {}
dataset_path = "/Users/ayman/Desktop/AI_mafia/Face Recognition/data/"
# dataset_path = "./data/"
face_data = []
labels = []
for fx in os.listdir(dataset_path):
    if fx.endswith(".npy"):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1,1))

train_set = np.concatenate((face_dataset, face_labels), axis=1)
print(train_set.shape)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue

    for (x,y,w,h) in faces:
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))
        out = knn(train_set, face_section.flatten())
        pred_name = names[int(out)]
        cv2.putText(gray_frame, pred_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray_frame, (x,y), (x+w,y+h), (0,255,0), 3)
        print(len(face_section))

    # cv2.imshow("Frame", frame)
    cv2.imshow("Gray frame", gray_frame)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()