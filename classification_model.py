import cv2
import numpy as np
import os
import shutil
import pywt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import json

face_cascade = cv2.CascadeClassifier('./haarCascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarCascade/haarcascade_eye.xml')

def get_cropped_img(image_path):
    img = cv2.imread(image_path,0)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces :
        roi_gray = img[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >=2:
            return roi_color

path_to = './dataset/'
path_to_cr = './dataset/cropped/'

img_dirs = []
for entry in  os.scandir(path_to):
    if entry.is_dir():
        img_dirs.append(entry.path)

if os.path.exists(path_to_cr):
    shutil.rmtree(path_to_cr)
os.mkdir(path_to_cr)

cropped_img_dir = []
celebrity_file = {}

for img_dir in img_dirs:
    celeb_name = img_dir.split('/')[-1]

    count = 1
    celebrity_file[celeb_name] = []

    for entry in os.scandir(img_dir):
        roi_color = get_cropped_img(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_cr + celeb_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_img_dir.append(cropped_folder)
                
            crp_file_name = celeb_name + str(count) + ".png"
            crp_file_path = cropped_folder + "/" + crp_file_name

            cv2.imwrite(crp_file_path,roi_color)
            celebrity_file[celeb_name].append(crp_file_path)
            count = count+1

def w2d(img, mode='haar',level=1):
    imarr = img #datatype conversion
    imarr = cv2.cvtColor(imarr,cv2.COLOR_BGR2GRAY) # convert to grayscale
    imarr = np.float32(imarr) #convert to float
    imarr/= 255
    coeff = pywt.wavedec2(imarr,mode,level=level) #computing coefficients

    coeff_h = list(coeff)
    coeff[0]*=0

    imarr_h = pywt.waverec2(coeff_h,mode)
    imarr_h*=255
    imarr_h = np.uint8(imarr_h)

    return imarr_h

celebrity_file= {}
for img_dir in cropped_img_dir:
    celeb_name = img_dir.split('/')[-1]
    print(celeb_name)
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    celebrity_file[celeb_name]=file_list

class_dict = {}
count=0
for celeb_name in celebrity_file.keys():
    class_dict[celeb_name] = count
    count = count + 1


x,y=[],[]
for celeb_name, train_file in celebrity_file.items():
    for train_img in train_file:
        img = cv2.imread(train_img)
        if img is None:
            continue
        sc_raw_img = cv2.resize(img,(32,32))
        img_har=w2d(img,'db1',5)
        sc_img_har = cv2.resize(img_har,(32,32))
        comb_img = np.vstack((sc_raw_img.reshape(32*32*3,1),sc_img_har.reshape(32*32,1)))
        x.append(comb_img)
        y.append(class_dict[celeb_name])

X = np.array(x).reshape(len(x),4096).astype(float)
        
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=0)
pipe = Pipeline([('scaler',StandardScaler()),('svc',SVC(kernel='linear',C=1))])
pipe.fit(X_train,y_train)

joblib.dump(pipe,'img_classification.pkl')
with open("class_dicte.json","w") as f:
    f.write(json.dumps(class_dict))