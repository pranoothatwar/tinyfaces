import dlib
from skimage import io
import numpy as np
import pandas as pd
import glob
import os
import time
import pickle
import svm_classifier

face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
predictor_path = './shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def get_vect(frame):
    # frame='/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/tinyfaces/images/23-friends-cover-story-lede.w750.h560.2x.jpg'
    img = io.imread(frame)
    dets = detector(img, 1)
    b = []

    
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        a = np.array(face_descriptor)
        b.append(a)

        # print(a)
    c = pd.DataFrame(b)
    # print(c)
    svm_classifier.predict(frame, c)
    # save_json( a, frame, '/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/tinyfaces/video/gray_feat/')

# def save_json(embedding, image_name, json_dir):
#     image_name = os.path.basename(image_name)
#     # print image_name
#     # k={}
#     # print type(embedding)
#     # k={'sadasd' : np.array(embedding)}
#     # k['uparwala'] = [str(x) for x in embedding]
#     # print k
#     with open(json_dir + image_name.split('.')[0]  +'.pkl', 'wb') as f:
#         pickle.dump(np.array(embedding), f)

for i in glob.glob('/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/tinyfaces/test/*'):
    start = time.time()
    get_vect(i)
    print(i, time.time() - start)