import json
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy.misc
import shutil
from pylab import *

def strokes_to_image(file_input_name, file_out_strokes_analysis):
	'''
	input : stoke json file
	return : narray of image
	'''
	strokes_data = read_strokes(file_input_name)
	n_data = normalize_x_y(strokes_data)
	im = draw_storke_in_line(n_data, file_out_strokes_analysis)
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

def find_boundary(n_data, file_out_strokes_analysis):
	'''
		find the four point boundary of the handwritting document input
		input: the x,y of strokes
	'''

	fileout = file_out_strokes_analysis

	boundary = {'i_left': 100000, 'i_top': 100000, 'i_right': -1, 'i_down': -1}
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
	
	i_left = boundary['i_left']
	i_top = boundary['i_top']

	fileout.write(str(i_left)+','+str(i_top)+'\n')
	
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

def draw_storke_in_line(n_data, file_out_strokes_analysis):
	boundary = find_boundary(n_data, file_out_strokes_analysis)
	
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

def proj_y_axis(im):
	'''
	Project the 2D strokes(im) onto 1D y axis
	input: im narray
	return: proj array
	'''
	proj = np.zeros(len(im))
	for i in range(len(im)):
		for j in range(len(im[i])):
			if im[i][j]!=255:
				proj[i]=proj[i]+1

	return proj

def find_sgment_section(im, file_out_strokes_analysis):
	outfile = file_out_strokes_analysis
	proj = proj_y_axis(im)
	y_split_points = split_row_y_proj(proj)
	section = find_hand_write_section(y_split_points)
	section_y_split_points_with_index = rule_find_power_index(y_split_points, section)

	for i,j in enumerate(section_y_split_points_with_index,0):
		if i!=len(section_y_split_points_with_index)-1:
			fileout.write(str(j)+',')
		else:
			fileout.write(str(j))

	return section_y_split_points_with_index


def split_row_y_proj(proj):
	'''
		record the possibly y be the segment of the document
		the place between row to row should be blank
		input: projection onto y axis
		output: the point on y that should be the section of the handwritting
	'''
	tmp_section=[]
	flag = False
	for i, j in enumerate(proj, 0):
		if j != 0 and flag==False:
			tmp_section.append(i)
			flag=True
		elif j==0 and flag==True:
			tmp_section.append(i-1)
			flag=False
		elif i==len(proj)-1:
			tmp_section.append(i)
			flag=False

	return tmp_section

def find_hand_write_section(y_split_points):
	'''
		find the handwritting part at y
		input: row line y points
		output: the segments of handwritting part
	'''
	tmp_section = y_split_points
	avg_seg_height=[]
	fn_section=[]

	for i in range(0,len(tmp_section)-1,2):
		tmp_seg_height=[]
		tmp_seg_height.append(tmp_section[i+1]-tmp_section[i])
		tmp_seg_height.append(i)
		tmp_seg_height.append(i+1)
		avg_seg_height.append(tmp_seg_height)

	print avg_seg_height

	return avg_seg_height

def find_average_row_height(hand_write_section):
	'''
		calculate the average height of handwritting sections
		input: handwritting sections
		output: average height
	'''
	avg_seg_height = hand_write_section
	avg_height_int=0.00000

	for i in range(len(avg_seg_height)):
		avg_height_int+=avg_seg_height[i][0]
	avg_height_int/=len(avg_seg_height)
	print avg_height_int

	return avg_height_int

def rule_find_power_index(y_split_points, section):
	'''
		revise the handwrittng section
		combine the index of math power
		if the section height are smaller than average, it may be an index

		input: row line y points
		output: revised section
	'''
	avg_height_int = find_average_row_height(section)
	tmp_section = y_split_points
	fn_section=[]

	fn_section_flag=False #check whether the past section is less than avg_height_int
	for i in range(len(section)):
		if (section[i][0]<=avg_height_int-100): #100 is variable
			fn_section.append(tmp_section[section[i][1]])
			fn_section_flag=True
		else:
			if fn_section_flag==False:
				fn_section.append(tmp_section[section[i][1]])
				fn_section.append(tmp_section[section[i][2]])
			else:
				fn_section.append(tmp_section[section[i][2]])
				fn_section_flag=False

	for i in range(len(fn_section)):
		cv2.line(im,(0,fn_section[i]),(1000,fn_section[i]),(0,0,0),2)

	print fn_section

	return fn_section

def bound_words(imgray, section, im):
	cnt_arr = bound_contours(imgray)
	im_2, section = draw_contours_bound(im, cnt_arr, section)
	# show contours bounds
	show_result_image(im_2, 'roi', 'im_2.bmp')

	im_3, list_retg = draw_charcter_bound(im, section)

	return im_3, list_retg

def bound_contours(imgray):
	'''
		bound the contours on image
		bound is a rectangle
		input: gray scale image
		output: contours array
	'''
	# read http://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html
	ret,thresh = cv2.threshold(imgray, 254, 255, cv2.THRESH_BINARY_INV)
	img,contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# bound the characters(contours)
	cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x:x[1])

	# make a characters bounds list
	arr=[]
	for index, (c, _) in enumerate(cnts):
		(x, y, w, h) = cv2.boundingRect(c)
		arr.append((x, y, w, h))

	return arr

def draw_contours_bound(im, contours_arr, section):
	'''
		draw the bound on an image
		?????
		input: image to draw
		       contours array
		       section list
		output: image with characters bounded
		        section and bound rectangle list(?????)
	'''
	im_2 = im

	arr = contours_arr
	fn_section = section

	a=0
	tmp_roi=[]
	# len(fn_section)/2 lines
	total_lines = len(fn_section)/2
	fn_section_retg=[[]for i in range(total_lines)] #segmentation word in section the third element 0 x 1 y 2 w 3 h

	for cnt in arr:
		x,y,w,h = cnt
		cv2.rectangle(im_2,(x,y),(x+w,y+h),(200,0,0),1)
		tmp_listretg=[x,y,w,h]
		
		for i in range(0,len(fn_section),2):
			if(y>=fn_section[i] and y+h-1<=fn_section[i+1]):
				fn_section_retg[i/2].append(tmp_listretg)

	print fn_section_retg[0]

	# deal with overlaping contours of a character such as '='
	bound_rule_overlap(fn_section_retg)

	return im_2, fn_section_retg

def bound_rule_overlap(fn_section_retg):
	'''
		the contours seperated may be stroke belong to the same words
		the rule find words like '=', overlap rule
		input: not sure????
		output: section and bound rectangle list(???????)
	'''
	for i in range(len(fn_section_retg)):
		j=0
		while j < len(fn_section_retg[i])-1:
			if (fn_section_retg[i][j+1][0]<fn_section_retg[i][j][0]+(fn_section_retg[i][j][2]/2)
				or fn_section_retg[i][j+1][0]+fn_section_retg[i][j+1][2]<=fn_section_retg[i][j][0]+fn_section_retg[i][j][2]):

				#fixed x
				#setting y and h
				if (fn_section_retg[i][j+1][1]<fn_section_retg[i][j][1]):
					fn_section_retg[i][j][3]=fn_section_retg[i][j][1]+fn_section_retg[i][j][3]-fn_section_retg[i][j+1][1]
					fn_section_retg[i][j][1]=fn_section_retg[i][j+1][1]
				else:
					if (fn_section_retg[i][j+1][1]+fn_section_retg[i][j+1][3]>fn_section_retg[i][j][1]+fn_section_retg[i][j][3]):
						fn_section_retg[i][j][3]=fn_section_retg[i][j+1][1]+fn_section_retg[i][j+1][3]-fn_section_retg[i][j][1]
				#setting w
				if (fn_section_retg[i][j+1][0]+fn_section_retg[i][j+1][2]>fn_section_retg[i][j][0]+fn_section_retg[i][j][2]):
					fn_section_retg[i][j][2]=fn_section_retg[i][j+1][0]+fn_section_retg[i][j+1][2]-fn_section_retg[i][j][0]
				
				fn_section_retg[i].pop(j+1)
			else:
				#if(fn_section_retg[i][j+1][1]+(fn_section_retg[i][j+1][3]/2)<=fn_section_retg[i][j][1]):
				#	cha_type[i][j+1]=1

				j+=1

	return fn_section_retg

def draw_charcter_bound(im, section):
	'''
		draw the bound of characters
		???????

		input: image to draw
		       section with rectangles bounding each character
		output: image with bounds
		        a list of all bounds
	'''
	im_3 = im
	fn_section_retg = section

	listretg=[]

	for i in range(len(fn_section_retg)):
		for j in range(len(fn_section_retg[i])):
			listretg.append(fn_section_retg[i][j])
			x=fn_section_retg[i][j][0]
			y=fn_section_retg[i][j][1]
			w=fn_section_retg[i][j][2]
			h=fn_section_retg[i][j][3]
			cv2.rectangle(im_3,(x,y),(x+w,y+h),(200,0,0),1)
	return im_3, listretg


def show_result_image(image, window_title, image_file_name):
	cv2.imshow(window_title, image)
	cv2.waitKey()
	scipy.misc.imsave(image_file_name, image)
	return

####################################################################################################
### start here ###
fileout = open("matrix_myscript.txt","w+")

# draw the handwritting by strokes, collect the analysis data(left and top point),
# and write in matrix_myscript.txt when processing
im = strokes_to_image('Stroke_21.json', fileout)

im_2=im.copy()
im_3=im.copy()

# convert im to grayscale image for convenience im => 2D array
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im = imgray.copy()
segment_section = find_sgment_section(imgray, fileout)

# check result: segment section on image
show_result_image(im, 'roi', 'im.bmp')

# bound the words
im_bounds, list_retg = bound_words(imgray, segment_section, im_3)

# show characters bounds
show_result_image(im_bounds, 'roi', 'im_3.bmp')