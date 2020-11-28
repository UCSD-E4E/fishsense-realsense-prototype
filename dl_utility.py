# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:42:43 2018

@author: xuwe421
"""

from collections import namedtuple
import io
import shutil 
import argparse
import getpass
import os, sys
import datetime
import time
import json
import csv
import subprocess
import os
import sys
import cv2
import pandas as pd
from sys import getsizeof
import numpy as np
from lxml import etree as ET

dir_eval = 'C:\\2016\\59_detect_fish\\videofish-master@d055b20a072\\evaluation\\'
dir_data = 'C:\\2016\\59_detect_fish\\3_data\\'

dir_input = 'C:\\2016\\59_detect_fish\\3_data\\orpc_iputs\\'
dir_labels = 'C:\\2016\\59_detect_fish\\3_data\\orpc_labels\\'
# add the folder of evaluation to system path
sys.path.append(dir_eval)

from annotator import *
from eyetest import *
#read one video
def load_video(video_name):
    print("Loading video file: " + video_name)
    img_all = []
    cap = cv2.VideoCapture(dir_data + video_name)
    length = 0
    ret, img = cap.read()
    print(getsizeof(img))
    img_int = img.astype(int)
    print(getsizeof(img_int))
    img_all.append(img)
    if (not ret):
        print("Error: could not read video frames")
        sys.exit()
    while (ret):
        length = length + 1
        ret, img = cap.read()
        img_all.append(img)
    cap.release()
    return length, img_all

def write_xml(full_name, img_size, detections):
    # read in the name of the file, the size of the file, detections in the format of a list of dictionary
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = name + '.png'
    ET.SubElement(annotation, "path").text = 'Unkown'
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = 'Unknown'
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_size[1])
    ET.SubElement(size, "height").text = str(img_size[0])
    ET.SubElement(size, "depth").text = str(img_size[2])
    ET.SubElement(annotation, "segmented").text =str(0)
    for i in range(len(detections)):
        myobject = ET.SubElement(annotation, "object",name="detection"+str(i))
        ET.SubElement(myobject, "name").text = 'fish'
        ET.SubElement(myobject, "pose").text = 'unspecified'
        ET.SubElement(myobject, "truncated").text = str(1)
        ET.SubElement(myobject, "difficult").text = str(0)
        bndbox = ET.SubElement(myobject, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(min(detections[i]['x1'], detections[i]['x2']))
        ET.SubElement(bndbox, "ymin").text = str(min(detections[i]['y1'], detections[i]['y2']))
        ET.SubElement(bndbox, "xmax").text = str(max(detections[i]['x1'], detections[i]['x2']))
        ET.SubElement(bndbox, "ymax").text = str(max(detections[i]['y1'], detections[i]['y2']))
    tree = ET.ElementTree(annotation)
    tree.write(full_name+".xml",  pretty_print=True)
    return None

video_list = pd.read_csv(u'C:\\2016\\59_detect_fish\\videofish-master@d055b20a072\\evaluation\\video_list_orpc.csv')
for ivideo in range(len(video_list)):
    #test
    video_name= video_list.video[ivideo]
    print(video_name)
    getTotalFrames(dir_data + video_name)
    length, img_all = load_video(video_name)
    
    anno_name = video_list.annotation[ivideo]
    print(anno_name)
    framesList = extractBoundingBoxesFromJSON(dir_eval+'annotations\\'+anno_name)
    
    #if the frames in video and annotation are the same, we write the files to folders
    
    if len(framesList) == len(img_all)-1:
        print('outputing ' + video_name + '...')
        for frame in range(len(framesList)):
            # name each frames
            name = video_name[0:-4] + '_fm_' + str(frame+1)
            name = name.split('\\')[-1]
            # save each frame to image
            cv2.imwrite(dir_input + name +'.png', np.uint8(img_all[frame]))
            #save each annotation into the format for training
            img_size = img_all[frame].shape
            write_xml(dir_labels +name, img_size = img_size, detections=framesList[frame])
    else:
        print('not match...')


# read one Jason file of a specific frame

# compile x and y data

#split train and test data



