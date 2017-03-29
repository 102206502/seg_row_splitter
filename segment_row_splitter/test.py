import json
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy.misc
import shutil
from pylab import *

def strokes_to_image(file_input_name, file_output_name):
	strokes_data = read_strokes(file_input_name)


# read the stroke_NN.json, and copy it to testOpenStrokes.txt for c# program
def read_strokes(file_name):
	with open(file_name) as fout:
		data = json.load(fout)

	shutil.copy(file_name, 'testOpenStrokes.txt')

	return data

# combine feild x, fx, and y, fy => new x = x*100 + fx, new y = y*100 + fy
# for opencv 
def normalize_x_y(strokes_data):
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
			print('tmp:', tmp)
			print('listdata:',listdata)
			import pdb; pdb.set_trace()  # breakpoint 41f145d8 //

			tmp=[]
		n_data.append(listdata)

	return n_data

