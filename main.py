import tiny_face_eval 
import extract_roi_features 
import video_prepro
import time

if __name__ == '__main__' :
	video_path = '/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/tinyfaces/video/output2.mp4'
	weight_file_tiny_face = '/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/old/resnet_tf.pkl'
	embedding_model = '/home/pranoot/Desktop/Tiny_Faces_in_Tensorflow-master/tinyfaces/20170512-110547/20170512-110547.pb'


	start = time.time()
	print 'started!'
	video_prepro.main(video_path)
	print 'frames generated in ', time.time() - start 

	start = time.time()
	tiny_face_eval.main(video_path, weight_file_tiny_face)
	print 'roi extracted in ',  time.time() - start 


	start = time.time()
	extract_roi_features.main(video_path, embedding_model)
	print 'features of roi extracted in ',  time.time() - start 



