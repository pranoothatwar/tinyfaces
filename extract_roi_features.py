import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
import cv2
from scipy import misc
import time
import json
import glob
# paths, labels = facenet.get_image_paths_and_labels(dataset)
            
#             print('Number of classes: %d' % len(dataset))
#             print('Number of images: %d' % len(paths))


def create_graph(model):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
			print('Loading feature extraction model')
			facenet.load_model(model)

			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			# embedding_size = embeddings.get_shape()[1]

			# print('graph is : ')
			# op = sess.graph.get_operations()
			# name_ops = [m.values() for m in op]
			# for name in name_ops:
			# 	print name

			# Run forward pass to calculate embeddings
			print('Calculating features for images')
			# nrof_images = len(paths)
			# nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
			# emb_array = np.zeros((nrof_images, embedding_size))
			# for i in range(nrof_batches_per_epoch):
			# start_index = i*args.batch_size
			# end_index = min((i+1)*args.batch_size, nrof_images)
			# paths_batch = paths[start_index:end_index]
			# images = facenet.load_data(paths_batch, False, False, args.image_size)
			return sess, images_placeholder, embeddings, phase_train_placeholder


def get_emb(sess, image, images_placeholder, embeddings, phase_train_placeholder):
	image = misc.imread(image)
	# print image.shape
	# image = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))
	image = np.reshape(image,(1,image.shape[0],image.shape[1],1))
	# image = misc.resize(image,(60,60))
	print image.shape
	feed_dict = { images_placeholder:image, phase_train_placeholder:False }
	image_embedding = sess.run(embeddings, feed_dict=feed_dict)

	# print type(image_embedding), image_embedding.shape
	mean = np.mean(image_embedding, axis =0)
	print type(mean), mean.shape, image_embedding.shape
	return np.mean(image_embedding)


def save_json(embedding, image_name, json_dir):
	image_name = os.path.basename(image_name)
	# print image_name
	# k={}
	# print type(embedding)
	# k={'sadasd' : np.array(embedding)}
	# k['uparwala'] = [str(x) for x in embedding]
	# print k
	with open(json_dir + image_name.split('.')[0]  +'.pkl', 'wb') as f:
		# f.write(embedding)
		pickle.dump(np.array(embedding), f)
	# with open(json_dir + image_name.split('.')[0]  +'.pkl', 'rb') as f:
		# print pickle.load(f)

def directory_exist(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def main(video_file_path, model):
	dirname = os.path.dirname(os.path.realpath(video_file_path))
	roi_dir = dirname +'/gray_roi/'
	features_dir = dirname + '/gray_feat/'
	directory_exist(features_dir)


	all_roi = glob.glob(roi_dir+'*')
	print all_roi
	sess, images_placeholder, embeddings, phase_train_placeholder =create_graph(model)
	for img in all_roi:
		emb = get_emb(sess, img, images_placeholder, embeddings, phase_train_placeholder)
		save_json(emb, img, features_dir)

# if __name__ == '__main__':
# 	model = '/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/tinyfaces/20170512-110547/20170512-110547.pb'
# 	image = '/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/tinyfaces/video/roi/4.jpeg'
# 	json_dir = '/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/tinyfaces/video/json/'
# 	print('hi')
# 	start_ = time.time()
# 	embeddings = main('/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/tinyfaces/video/output.mp4',model)


# 	print('time required is : ', (time.time() - start_))
          