# -*- coding: utf-8 -*-
import json
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy.misc
import shutil
from pylab import *
from operator import itemgetter

def strokes_to_image(file_input_name, file_out_strokes_analysis):
	'''
	input : stoke json file
	return : narray of image
	'''
	strokes_data = read_strokes(file_input_name)
	n_data = normalize_x_y(strokes_data)
	im, boundary = draw_storke_in_line(n_data, file_out_strokes_analysis)
	return im, n_data, boundary


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

		input : 
			strokes_data : 3D dictionary json file
		output :
			n_data : 3D int list
				normalized (x, y) of stroke 
				format is 'n_data[strokes_index][stroke_index][x or y]'

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
		output: boudary of the handwritting document
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
		#print 'n_data[][][0], n_data[][][1]:', n_data[i][0][0]-boundary['i_left'], n_data[i][0][1]-boundary['i_top']
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
	return im, boundary

def bound_contours(imgray):
	'''
		bound the contours on image
		bound is a rectangle
		input: gray scale image
		output: contours array sorted by x of rectangles
	'''
	# read http://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html
	ret,thresh = cv2.threshold(imgray, 254, 255, cv2.THRESH_BINARY_INV)
	img,contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	return contours

def sort_conts_y(contours):
	
	cnts = [(c, cv2.boundingRect(c)[0]) for c in contours]
	# extract the rectangles in cnts
	arr=[]
	for index, (c, _) in enumerate(cnts):
		(x, y, w, h) = cv2.boundingRect(c)
		arr.append((x, y, w, h))
	
	arr = sorted(arr, key = itemgetter(1))

	# for i in range(len(arr)):
	# 	print 'arr[',i,']', arr[i]

	return arr

def find_sgment_section(im, file_out_strokes_analysis):
	fileout = file_out_strokes_analysis
	contours = bound_contours(imgray)
	contours_regs = sort_conts_y(contours)
	section_regs, y_split_points = find_section_by_conts(contours_regs)

	section = find_hand_write_section(y_split_points)
	y_split_points = rule_find_power_index(y_split_points, section)
	# print 'after sort by x' 
	# for line in range(len(section_regs)):
	# 	for reg in range(len(section_regs[line])):
	# 		print 'reg[', line, '][', reg, ']', section_regs[line][reg]


	# print 'y points of row line', y_split_points
	im = draw_row_lines(im, y_split_points)
	show_result_image(im, 'im', 'im.bmp')

	section_regs = bound_rule_overlap(section_regs)

	for i,j in enumerate(y_split_points,0):
			if i!=len(y_split_points)-1:
				fileout.write(str(j)+',')
			else:
				fileout.write(str(j))


	return y_split_points, section_regs, contours_regs

def find_section_by_conts(cnt_regs):
	'''
		find section by contours, cast tuple of rectangle to list
		input : 
			cnt_regs : 4 elements tuple list
				the rectangles of contours sorted by y
		output : 
			section rectangles : list of rectangle list
			y spilt points : int list

		pseudocode 
		-----------
		# upper bound = y of cnt_regs
		# lower bound = y + h of cnt_regs
		# for reg in cnt_regs
		# 	x = x of reg
		# 	y = y of reg
		# 	w = w of reg
		# 	h = h of reg
		# 	if current line list is empty(just after appended)
		# 		append reg in current line list
		# 		update upper bound
		# 		update lower bound 
		# 	elif y <= lower bound
		# 		append reg in current line list
		# 		if y + h > lower bound
		# 			lower bound = y + h
		# 	else
		# 		sort current line list by x
		# 		append current line list to section_regs
		# 		record the upper bound and lower bound as y split point
		# 		current line list = []
		# if current line is not empty
		# 	sort current line list by x
		# 	append current line list to section_regs
		# 
		# 
	'''
	upper_bound = 0
	lower_bound = 0

	section_regs = []
	current_section_regs = []
	y_split_points = []

	print('number of regs before find_section_by_conts', len(cnt_regs))

	for reg in cnt_regs:
		x = reg[0]
		y = reg[1]
		w = reg[2]
		h = reg[3]
		temp_list = [x, y, w, h]
		if not current_section_regs:
			current_section_regs.append(temp_list)
			upper_bound = reg[1]
			lower_bound = upper_bound + reg[3]
		elif reg[1] <= lower_bound:
			current_section_regs.append(temp_list)
			if y + h > lower_bound:
				lower_bound = y + h
		else:
			current_section_regs = sorted(current_section_regs, key = itemgetter(0))
			section_regs.append(current_section_regs)
			y_split_points.append(upper_bound)
			y_split_points.append(lower_bound)
			current_section_regs = []

			current_section_regs.append(temp_list)
			upper_bound = reg[1]
			lower_bound = upper_bound + reg[3]

	if current_section_regs:
		current_section_regs = sorted(current_section_regs, key = itemgetter(0))
		section_regs.append(current_section_regs)
		y_split_points.append(upper_bound)
		y_split_points.append(lower_bound)
	
	return section_regs, y_split_points

def find_hand_write_section(y_split_points):
	'''
		find the handwritting part at y
		input: 
			y_split_points : 1D int list
			row line y points
		output: 
			avg_seg_height : 2D int list
				the segments of handwritting part
				[[upper bound, lower bound],...(each line)] is the format
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

	print (avg_seg_height)

	return avg_seg_height

def find_average_row_height(hand_write_section):
	'''
		calculate the average height of handwritting sections
		input: 2D int list
				the segments of handwritting part
				[[upper bound, lower bound],...(for each line)] is the format
		output: 
			avg_height_int : int
				average height
	'''
	avg_seg_height = hand_write_section
	avg_height_int=0.00000

	for i in range(len(avg_seg_height)):
		avg_height_int+=avg_seg_height[i][0]
	avg_height_int/=len(avg_seg_height)
	print(avg_height_int)

	return avg_height_int

def rule_find_power_index(y_split_points, section):
	'''
		revise the handwrittng section
		combine the index of math power
		if the section height are smaller than average, it may be an index
		input: 
			y_split_points : 1D int list
				row line y points
			section : 
		output: 
			fn_section :1D int list
				row line y pointsrevised section
	'''
	avg_height_int = find_average_row_height(section)
	fn_section=[]

	fn_section_flag=False #check whether the past section is less than avg_height_int
	for i in range(len(section)):
		if i+1<len(y_split_points):
			if fn_section_flag:
				fn_section.append(y_split_points[section[i][2]])
				fn_section_flag = False
			else:
				fn_section.append(y_split_points[section[i][1]])
				if (section[i][0]<=avg_height_int-100) and (y_split_points[section[i][1]+1]-y_split_points[section[i][1]] < 100):
					fn_section_flag = True
				else:
					fn_section.append(y_split_points[section[i][2]])

	print('revised y_split_points:',fn_section)

	return fn_section

def draw_contours_bound(im, contours_arr, section):
	'''
		draw the bound on an image
		input: 
			im : 2D narray
				image to draw
		    contours_arr : contours list
		    	contours array
		    section : 1D int list
		    	y split points
		output: image with characters bounded
	'''
	im_2 = im

	arr = contours_arr
	fn_section = section
	print('the segment when draw_contours_bound', section)

	# make a empty list with handwritting document lines length
	total_lines = len(fn_section)/2
	fn_section_retg=[[]for i in range(int(total_lines))] #segmentation word in section the third element 0 x 1 y 2 w 3 h
	
	for cnt in arr:
		x,y,w,h = cnt
		cv2.rectangle(im_2,(x,y),(x+w,y+h),(200,0,0),1)

	return im_2

def bound_rule_overlap(fn_section_retg):
	'''
		the contours seperated may be stroke belong to the same words
		the rule find words like '=', overlap rule
		input: 2D regtangle list
			rectangle list is [x, y, w, h]
		output: 2D regtangle tuple list
			rectangle list is [x, y, w, h]
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

def draw_row_lines(im, y_split_points):
	fn_section = y_split_points
	for i in range(len(fn_section)):
		cv2.line(im,(0,fn_section[i]),(1000,fn_section[i]),(0,0,0),2)

	return im

def draw_charcter_bound(im, section):
	'''
		draw the bound of characters
		make a 1D bounds list

		input: 
			im : 2D narray
				image to draw
			section : 3D int list
		       section with rectangles bounding each character
		       2D rectangle list
		output: 
			im_3 : 2D narray
				image with bounds
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
	
	return im_3

def classify_strokes_in_characters(bounds_retg, n_data, boudary, section_line):
	'''
		掃描所有筆畫，對照文字矩形邊界，歸類屬於各邊界內的筆畫
		input:
			bounds_retg : 3D int list
				2D rectangles in [x, y, w,h] format
			n_data : 3D narray of stoke x, y

	'''
	listretg = []
	for retgs in bounds_retg:
		listretg += retgs

	listretg = sorted(listretg, key = itemgetter(0))

	i_left = boundary['i_left']
	i_top = boundary['i_top']
	fn_section = section_line

	cha_arr=[[] for i in range(len(listretg))]
	# append cha_arr if the (x,y) in char belongs to the rectangle 
	for i,retg in enumerate(listretg):
		for j,strokes in enumerate(n_data):
			if (strokes[0][0]-i_left+1 >= retg[1] 
				and strokes[0][0]-i_left+1 <= retg[1]+retg[3] 
				and strokes[0][1]-i_top+1 >= retg[0] 
				and strokes[0][1]-i_top+1 <= retg[0]+retg[2]):
				cha_arr[i].append(j)

	cha_arr=sorted(cha_arr)

	for i in range(len(cha_arr)):
		for j in range(int(len(fn_section)/2)):
			first_strokes_idx = cha_arr[i][0]
			x = n_data[first_strokes_idx][0][0]-i_left+1
			y = n_data[first_strokes_idx][0][1]-i_top+1
			upper_bound = fn_section[j*2]
			lower_bound = fn_section[j*2+1]
			# 確認該方框區域筆畫是屬於哪行 註：i_top 其實是 i_left以此類推
			if (x>=upper_bound and x<=lower_bound):
			#if (y>=upper_bound and y<=lower_bound):
				cha_arr[i].insert(0,j)
				break
	

	return cha_arr

def write_strokes_in_characters_time(cha_arr, file_out_strokes_analysis):

	fileout = file_out_strokes_analysis

	fileout.write('\n')
	for i,cha in enumerate(cha_arr,0):
		if i!=len(cha_arr)-1:
			for j,stroke_num in enumerate(cha,0):
				if j!=len(cha)-1:
					fileout.write(str(stroke_num)+',')
				else:
					fileout.write(str(stroke_num))
			fileout.write('\n')
		else:
			for j,stroke_num in enumerate(cha,0):
				if j!=len(cha)-1:
					fileout.write(str(stroke_num)+',')
				else:
					fileout.write(str(stroke_num))

	return


def show_result_image(image, window_title, image_file_name):
	# cv2.imshow(window_title, image)
	# cv2.waitKey()
	scipy.misc.imsave(image_file_name, image)
	return

##########################################################################
###########stat here##############

fileout = open("matrix_myscript.txt","w+")

file_out_id = open("Stroke_NN.txt", "r")
file_id = file_out_id.read() # the 'NN' of file name Stroke_NN.json
file_out_id.close()
stroke_file_name = 'Strokes json file/Stroke_' + file_id + '.json' # make the string of the stroke file

im, n_data, boundary = strokes_to_image(stroke_file_name, fileout)

im_2=im.copy()
im_3=im.copy()

# convert im to grayscale image for convenience im => 2D array
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im = imgray.copy()

y_split_points, section_regs, contours_regs = find_sgment_section(im, fileout)

im_2 = draw_contours_bound(im_2, contours_regs, y_split_points)
show_result_image(im_2, 'im 2', 'im_2.bmp')

im_3 = draw_charcter_bound(im_3, section_regs)
show_result_image(im_3, 'im 3', 'im_3.bmp')

cha_arr = classify_strokes_in_characters(section_regs, n_data, boundary, y_split_points)

write_strokes_in_characters_time(cha_arr, fileout)