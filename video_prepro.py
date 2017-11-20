import sys
import cv2
import os
import numpy as np
import skimage
import tensorflow as tf
import time
from argparse import ArgumentParser

def preprocess_frame(image, target_height=0, target_width=0):
    #function to resize frames then crop
    # if len(image.shape) == 2:
    #     image = np.tile(image[:,:,None], 3)
    # elif len(image.shape) == 4:
    #     image = image[:,:,:,0]

    # image = skimage.img_as_float(image).astype(np.float32)
    # height, width, rgb = image.shape
    # if width == height:
    #     resized_image = cv2.resize(image, (target_width,target_height))

    # elif height < width:
    #     #cv2.resize(src, dim) , where dim=(width, height)
    #     #image.shape[0] returns height, image.shape[1] returns width, image.shape[2] reutrns 3 (3 RGB channels)
    #     resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_height))
    #     cropping_length = int((resized_image.shape[1] - target_width) / 2)
    #     resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    # else:
    #     resized_image = cv2.resize(image, (target_width, int(height * float(target_width) / width)))
    #     cropping_length = int((resized_image.shape[0] - target_height) / 2)
    #     resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    # print resized_image
    # return cv2.resize(resized_image, (target_width, target_height))
    # print image.shape
    if image.shape[0] >1000:
        HEIGHT = image.shape[0]/2
        WIDTH = image.shape[1]/2
    else :
        HEIGHT = image.shape[0]
        WIDTH = image.shape[1]

    return cv2.resize(image, (WIDTH, HEIGHT))

def save_frame(frame_list, frame_dir):
    for i, frame in enumerate(frame_list):
        # print type(frame), frame.shape
        cv2.imwrite(frame_dir+'/'+str(i)+'.jpeg', frame)

def get_frame_list(frame_num):
    start = 0.0
    i = 0
    end = 0.0
    frame_list = []
    # if frame_num >FRAME_THRESHOLD:
    #     frame_num = FRAME_THRESHOLD
    while end < frame_num:
        start = 10.0*i
        end = start + 10.0
        i += 1
        if end > frame_num:
            end = frame_num
        frame_list.append(start)
    return frame_list


def directory_exist(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def make_frames(video_path, frame_path):

    
    # print idx, video,
    time_start = time.time()
    # if os.path.exists( os.path.join(video_save_path, video) ):
    #     print "Already processed ... "
    #     continue

    # video_fullpath = os.path.join(video_path, video)
    video_fullpath = video_path

    try:
        cap  = cv2.VideoCapture(video_fullpath)
    except:
        print 'fail : ', video_fullpath
        pass

    total_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    frame_index = get_frame_list(total_frames)
    frame_count = 0
    frame_list = []
    while True:
        #extract frames from the video, where each frame is an array (height*width*3)
        ret, frame = cap.read()
        if ret is False:
            break

        frame_list.append(frame)
        frame_count += 1


    # frame_list = np.array(frame_list)
    # print frame_list

    frame_list_reduced = np.array([frame_list[int(i)] for i in frame_index])

    # if frame_count > num_frames:
    #     #select 80 frames if frame_cout is >80
    #     frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
    #     frame_list = frame_list[frame_indices]

    reshaped_list = np.asarray(map(lambda x: preprocess_frame(x), frame_list_reduced))
    save_frame(reshaped_list, frame_path)

def main(video_path):
    # argparse = ArgumentParser()
    # argparse.add_argument('--video_path', type=str, help='path to video ', default="/path/to/mat2tf.mp4")
    # argparse.add_argument('--frame_dir', type=str, help='frame output directory.', default="/path/to/input_image_directory")

    # args = argparse.parse_args()
    dirname = os.path.dirname(os.path.realpath(__file__))
    # video_path = dirname + '/'+args.video_path
    # frame_dir = dirname + '/' + args.frame_dir
    frame_dir =  os.path.dirname(os.path.realpath(video_path)) + '/frames'
    directory_exist(video_path)
    directory_exist(frame_dir)

    make_frames(video_path = video_path, frame_path = frame_dir)



# if __name__ == '__main__':
#     main()

