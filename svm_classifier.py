import time
# import keras
import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import glob

svc = svm.SVC(probability=True)

# param_grid = {
#     "kernel": ['linear', 'rbf'],
#     "gamma": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1],                     ## 'kernel': 'linear', 'gamma': 1e-6, 'C': 0.1
#     "C": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]}



# Training Part
# train_data = pd.read_csv('/home/ayush/Downloads/rootTrainingData.csv')
# train_data_X =  train_data.iloc[:,2:130]    # feature Vector's
# train_data_Y = train_data.iloc[:,130]       # Output data


# fitting/training a model
# CV_svc = GridSearchCV(estimator=svc, param_grid=param_grid, cv=10)
# CV_svc.fit(train_data_X, train_data_Y)
# print(CV_svc.best_params_)

# clf = svm.SVC(kernel='rbf', C = 2, gamma = 10, probability=True)
# clf.fit(train_data_X, train_data_Y)
# joblib.dump(CV_svc, 'SVMMODEL.pkl')



# Testing Part
# test_data_dir = '/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/tinyfaces/video/gray_feat/'

# test_data_dir = glob.glob(test_data_dir+'*')
clf = joblib.load('/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/tinyfaces/SVMMODEL_new.pkl')
# for test_data in test_data_dir:
# 	with open(test_data, 'rb') as f:
# 		test_data_feature = pickle.load(f)

	# print type(test_data_feature)

	# test_data_feature = test_data.iloc[:,2:130]


		

	#predicting the unique model
def predict(img, test_data_feature):	
	predict_uniq_test = clf.predict(test_data_feature)
	print(img)
	print(predict_uniq_test)


	# calculating the probability score for each classes
	predict_prob = clf.predict_proba(test_data_feature)
	print(predict_prob)

	# test_data['Tag'] = predict_uniq_test

	predict_prob_df = pd.DataFrame(predict_prob)
	print('\n')
	# test_data['confidence'] =  predict_prob_df.max(axis=1)



	# test_data.to_csv('outcome'+''+'.csv', index = False)
