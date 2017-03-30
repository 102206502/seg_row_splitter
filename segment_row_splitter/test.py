import json
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy.misc
import shutil
from pylab import *

def strokes_to_image(file_input_name):
	strokes_data = read_strokes(file_input_name)
	n_data = normalize_x_y(strokes_data)
	im = draw_storke_in_line(n_data)
	return im


def read_strokes(file_name):
	'''
		read the stroke_NN.json, and copy it to testOpenStrokes.txt for c# program
		file_name: stroke_NN.json
	'''
	with open(file_name) as fout:
		data = json.load(fout)
	shutil.copy(file_name, 'testOpenStrokes.txt')
	return data
 
def normalize_x_y(strokes_data):
	'''
		combine feild x, fx, and y, fy => 
		new x = x*100 + fx, 
		new y = y*100 + fy
		for opencv
	'''
	data = strokes_data
	n_data=[]
	listdata=[]
	tmp=[]
	for i in range(len(data['strokes'])):
		for j in range(len(data['strokes'][i]['stroke'])):
			x = data['strokes'][i]['stroke'][j]['x']*100+data['strokes'][i]['stroke'][j]['fx']
			y = data['strokes'][i]['stroke'][j]['y']*100+data['strokes'][i]['stroke'][j]['fy']
			tmp.append(y)
			tmp.append(x)
			listdata.append(tmp)
			tmp=[]
		n_data.append(listdata)
		listdata=[]

	return n_data

def find_boundary(n_data):
	boundary = {'i_left': 100000, 'i_top': 100000, 'i_right': -1, 'i_down': -1}
	'''
		find the four point boundary of the handwritting document input
		input: the x,y of strokes
	'''
	for i in range(len(n_data)):
		for j in range(len(n_data[i])):
			if boundary['i_left']>n_data[i][j][0]:
				boundary['i_left'] = n_data[i][j][0]
			if boundary['i_top']>n_data[i][j][1]:
				boundary['i_top'] = n_data[i][j][1]
			if boundary['i_right']<n_data[i][j][0]:
				boundary['i_right'] = n_data[i][j][0]
			if boundary['i_down']<n_data[i][j][1]:
				boundary['i_down'] = n_data[i][j][1]
	return boundary

def draw_storke_in_dot(n_data, boundary):
	t_x = boundary['i_right']-boundary['i_left']+1
	t_y = boundary['i_down']-boundary['i_top']+1

	pre_data = np.full((t_x,t_y),255,dtype=np.int16)

	for i in range(len(n_data)):
		for j in range(len(n_data[i])):
			c_x = n_data[i][j][0]
			c_y = n_data[i][j][1]
			pre_data[c_x-boundary['i_left']][c_y-boundary['i_top']]=0
	
	scipy.misc.imsave('outfile.bmp',pre_data)
	im = cv2.imread('outfile.bmp')

	return im

def draw_storke_in_line(n_data):
	boundary = find_boundary(n_data)
	
	im = draw_storke_in_dot(n_data, boundary)
	
	for i in range(len(n_data)):
		for j in range(len(n_data[i])-1):
			c_x = n_data[i][j][0]
			c_y = n_data[i][j][1]
			n_x = n_data[i][j+1][0]
			n_y = n_data[i][j+1][1]
			point_x1 = c_x-boundary['i_left']
			point_y1 = c_y-boundary['i_top']
			point_x2 = n_x-boundary['i_left']
			point_y2 = n_y-boundary['i_top']
			cv2.line(im,(point_y1,point_x1),(point_y2,point_x2),(0,0,0),25)
		

	scipy.misc.imsave('im.bmp',im)
	return im


im = strokes_to_image('Stroke_21.json')

# cast to color image for opencv
im_2=im.copy()
im_3=im.copy()

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

im = imgray.copy()