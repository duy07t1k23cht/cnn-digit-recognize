# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import glob
import matplotlib.pyplot as plt
import pytesseract as tess 
from imutils.object_detection import non_max_suppression
import time

tess.pytesseract.tesseract_cmd =  r'C:\Users\Admin\AppData\Local\Tesseract-OCR\tesseract.exe'
east = r".\frozen_east_text_detection.pb"


################ Some Image-Preprocessing ################
def unsharp_masking(im, ksize=5):
	# A preprocessing for increase contrast
	gaussian = cv2.GaussianBlur(im, (ksize, ksize), 0)
	res = cv2.addWeighted(im.copy(), 1.5, gaussian, -0.5, 0, im.copy())
	return res

def icr_contrast(img):
	# A subprocess for increase contrast
	lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3,3))
	cl = clahe.apply(l)
	limg = cv2.merge((cl,a,b))
	final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	return final

def increase_contrast(img, ksize = 3):
	# Main increase contrast
	gaussian = cv2.GaussianBlur(img, (ksize, ksize), 0)
	contrast = icr_contrast(img)
	res = cv2.addWeighted(contrast, 1.5, gaussian, -0.5, 0, contrast)
	return res

def morphology(img):
	kernel = np.ones((3, 3), dtype = np.float32)
	return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

################### Perfective transform ###################
def order_points(pts):
	rect = np.zeros((4, 2), dtype="float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect


def distance(point_1, point_2):
	abss = point_1 - point_2
	return math.sqrt(abss[0]*abss[0] + abss[1]*abss[1])


def _get_perfective_transform(points, img):
	points = order_points(points)
	object_width = int(max(distance(points[0], points[1]), distance(points[2], points[3])))
	object_height = int(max(distance(points[1], points[2]), distance(points[3], points[0])))
	#print('Width: {}     Height: {}'.format(object_width, object_height))
	dst = np.array([
		[0, 0],
		[object_width - 1, 0],
		[object_width - 1, object_height - 1],
		[0, object_height - 1]], dtype="float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(points, dst)
	warped = cv2.warpPerspective(img, M, (object_width, object_height))
	# return the warped image
	return warped, object_width, object_height

######################### Extract score_table #########################

def _get_largest_contour(contours, width, height):

	max_accept_area = 0.8 * width * height
	min_accept_area = 0.25 * width * height
	areas = [cv2.contourArea(contour) for contour in contours]
	zipped = [(x, y) for (x, y) in zip(contours, areas) if (y < max_accept_area and y > min_accept_area)]
	# Sort from higher to lower
	sorted_by_area = [(x, y) for (x, y) in sorted(zipped, key = lambda x:x[1], reverse = True)]
	# Return the largest
	return sorted_by_area[0] 

def _extract_scoreTable(img):
	base_img = img.copy()
	height, width, _ = img.shape
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
	# cv2.imshow('thresh', threshed)
	# cv2.imshow('thresh_morph', morphology(threshed))
	_, contours, _ = cv2.findContours(threshed, 1, 2)

	cnt, area = _get_largest_contour(contours, width, height)
	print('Found score_table with area: {}'.format(area / width / height))
	
	epsilon = 0.1*cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, epsilon, True)
	points = np.squeeze(approx)
	assert len(points) == 4, 'Error when approximate quadragon score_table'
	img = cv2.polylines(img, [points.astype('int32')], True, (0, 0, 255), 2)
	cv2.imwrite('table.jpg', img)
	#print(points)
	score_table, score_table_width, score_table_height = _get_perfective_transform(points, base_img)
	return score_table, score_table_width, score_table_height, points

######################### Extract Horizontal & Vertical Lines #########################
def _clean_lines_horizontal(lines, max_height):
	# Maybe noise when line in header, remove by thresh distance
	distance = []
	for i in range(1, len(lines)):	
		distance.append(lines[i] - lines[i-1])
	# Using mean to get all line
	mean_distance = np.mean(distance)
	thresh_distance = 0.95
	# Append first line
	accepted_lines = [0]
	for i in range(len(lines)):
		approx = abs(accepted_lines[-1] - lines[i]) / mean_distance 
		if approx > thresh_distance :
			accepted_lines.append(lines[i])
	# Append final line
	approx = abs(accepted_lines[-1] - max_height) / mean_distance 
	if approx > thresh_distance:
		accepted_lines.append(max_height)
	return accepted_lines

def _clean_lines_vertical(lines, max_width):
	# Maybe noise when line in header, remove by thresh distance
	min_distance = 10
	# Using mean to get all line
	accepted_lines = [0]
	for line in lines:
		if line - accepted_lines[-1] > min_distance:
			accepted_lines.append(line)

	if max_width - accepted_lines[-1] > min_distance:
		accepted_lines.append(max_width)
	return accepted_lines

def _projection_extracter_(thresh_img, score_table, extract_type):
	# Given threshed image in binary format
	# Return position of horizontal lines or vertical lines in list()
	# Extracted_type = 1 for horizontal, 0 for vertical
	hist = cv2.reduce(thresh_img, extract_type, cv2.REDUCE_AVG).reshape(-1)
	
	th = 255/2
	H,W = thresh_img.shape[:2]
	if extract_type == 1:
		can_be_lines = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
	else:
		can_be_lines = [y for y in range(W-1) if hist[y]<=th and hist[y+1]>th]

	lines = [can_be_lines[0]]
	for line in can_be_lines[1:]:
		if line - lines[-1] >= 4:
			lines.append(line)

	thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
	if extract_type == 1:
		accepted_lines = _clean_lines_horizontal(lines, H)
	else:
		accepted_lines = _clean_lines_vertical(lines, W)

	# for y in accepted_lines:
	# 	if extract_type == 1:
	# 		cv2.line(score_table, (0,y), (W, y), (0,0, 255), 2)
	# 	else:
	# 		cv2.line(score_table, (y,0), (y, H), (0,0, 255), 2)

	return accepted_lines

def _extract_lines(score_table, score_table_width, score_table_height):
	# Thresh
	img_icr_contrast = increase_contrast(score_table)
	_, threshed = cv2.threshold(cv2.cvtColor(img_icr_contrast, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

	# Create the images that will use to extract the horizontal and vertical lines
	horizontal = np.copy(threshed)
	vertical = np.copy(threshed)

	# Specify size on horizontal axis
	cols = horizontal.shape[1]
	horizontal_size = int(cols / 30)

	# Create structure element for extracting horizontal lines through morphology operations
	horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

	# Apply morphology operations
	horizontal = cv2.erode(horizontal, horizontalStructure)
	horizontal = cv2.dilate(horizontal, horizontalStructure) 

	# Show extracted horizontal lines
	horizontal_lines = _projection_extracter_(horizontal, img_icr_contrast, extract_type = 1)

	# Specify size on vertical axis
	rows = vertical.shape[0]
	verticalsize = int(rows / 30)

	# Create structure element for extracting vertical lines through morphology operations
	verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

	# Apply morphology operations
	vertical = cv2.erode(vertical, verticalStructure)
	vertical = cv2.dilate(vertical, verticalStructure)

	# Show extracted vertical lines
	vertical_lines = _projection_extracter_(vertical, img_icr_contrast, extract_type = 0)
	return horizontal_lines, vertical_lines

######################### Extract Id & Score #########################
def remove_line(thresh_img):
	horizontal = np.mean(thresh_img, axis = 1)
	H, W = thresh_img.shape[:2]
	flag = 0
	for i in range(H//3):
		if horizontal[i] > 0.4 * 255.0:
			flag = i
	thresh_img[: flag] = 0

	vertical = np.mean(thresh_img, axis = 0)
	flag = 0
	for i in range(W//8):
		if vertical[i] > 0.5 * 255.0:
			flag = i
	thresh_img[:, : flag] = 0
	return thresh_img

def _get_batch_imgs(scores):
	# Convert image to right format of MNIST Classifier
	result = []
	for score in scores:
		# Resize
		height, width, _ = score.shape
		new_width = 400
		new_height = int(new_width*height/width)
		score = cv2.resize(score, (new_width, new_height))
		score = increase_contrast(score)
		gray = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)
		_, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

		# Make sure that bounding of image are black
		threshed = remove_line(threshed)

		_, contours, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		boundingContours = [cv2.boundingRect(cnt) for cnt in contours]
		ratioAreaContours = [w*h/new_width/new_height for (x, y, w, h) in boundingContours]
		sorted_by_area = [bounding \
			for (bounding, ratio_area) in sorted(zip(boundingContours, ratioAreaContours), key = lambda x:x[1], reverse = True)\
				 if ratio_area < 0.5]

		current_batch_img = []
		current_coor = []
		for bnd in sorted_by_area:
			x, y, w, h = bnd
			if w > 0.4 * new_height and w/h > 0.5 and w/h < 2:
				# cv2.rectangle(score, (x, y), (x+w, y+h), (0, 255, 0), 2)

				overlaped = False
				new_x, new_y, new_w, new_h = x, y, w, h

				if len(current_coor) > 0:
					# Maybe we need to concat some part of number
					for idx, coor in enumerate(current_coor):
						old_x, old_y, old_w, old_h = coor
						itersection = min(old_x + old_w, x + w) - max(old_x, x) 
						union = max(old_x + old_w, x + w) - min(old_x, x)
						if itersection / union > 0.7:   # it's 2 parts of just 1 number, so concat it
							new_x = min(old_x, x)
							new_y = min(old_y, y)
							new_w = union 
							new_h = max(old_y + old_h, y + h) - new_y
							overlaped = idx
							break		

				part = cv2.resize(threshed[new_y: new_y+ new_h, new_x: new_x+new_w], (28, 28))
				if isinstance(overlaped, bool):
					# Not overlap
					current_coor.append([new_x, new_y, new_w, new_h])
					current_batch_img.append(part)
				else:
					# Overlap, so we update
					current_coor[overlaped] = [new_x, new_y, new_w, new_h]
					current_batch_img[overlaped] = part
				# cv2.rectangle(score, (new_x, new_y), (new_x+new_w, new_y+new_h), (0, 0, 255), 4)
			if len(current_coor) == 2:
				break

		# Sort by x
		current_batch_img = [x for (x, y) in sorted(zip(current_batch_img, current_coor), key = lambda x:x[1][0])]
		result.append(current_batch_img)
		# cv2.imshow('score', score)
		# count = len(glob.glob('tmp/*.jpg'))
		# cv2.imwrite('tmp/image_{}.jpg'.format(count), score)
		# cv2.imwrite('tmp2/image_{}.jpg'.format(count), threshed)
	return result


def _extract_text(score_table, ids_column_index, score_column_index, horizontal_lines, vertical_lines):
	# Extract all ids and their's scores
	score_table = increase_contrast(score_table)
	ids = list()
	scores = list()
	total_horizontal_lines = len(horizontal_lines)
	total_vertical_lines = len(vertical_lines)
	# Line 0-1 is header and the last is end of score_table, so skip 2 line
	for i in range(1, total_horizontal_lines - 1):
		# Get region of ids
		# Top-left
		ids_x1 = vertical_lines[ids_column_index - 1]
		ids_y1 = horizontal_lines[i]
		# Bottom-right
		ids_x2 = vertical_lines[ids_column_index]
		ids_y2 = horizontal_lines[i+1]
		ids.append(score_table[ids_y1: ids_y2, ids_x1: ids_x2])

		# Get region of score
		# Top-left
		sco_x1 = vertical_lines[score_column_index - 1]
		sco_y1 = horizontal_lines[i]
		# Bottom-right
		sco_x2 = vertical_lines[score_column_index]
		sco_y2 = horizontal_lines[i+1]
		scores.append(score_table[sco_y1: sco_y2, sco_x1: sco_x2])
	total_ids = len(ids)
	print('Total {} rows'.format(total_ids))
	return ids, scores
	
def _digit_detection(image):
	orig = image.copy()

	H, W = image.shape[:2]

	(newW, newH) = (320, 320)
	rW = W / float(newW)
	rH = H / float(newH)
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]
	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layer_names = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

	#load the pretrained EAST detector
	net = cv2.dnn.readNet(east)
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layer_names)

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	#loop over the number of rows
	for y in range(0, numRows):
		#extract the scores, followed by the geometrical data used to derive potential bounding box coordiantes taht surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		#loop over the number of columns 
		for x in range(0, numCols):
			if scoresData[x] < 0.5: continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	boxes = non_max_suppression(np.array(rects), probs=confidences)
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)-10
		startY = int(startY * rH)-10
		endX = int(endX * rW)+10
		endY = int(endY * rH)+10

		croped_image = orig[startY:endY, startX:endX]
		string =  tess.image_to_string(croped_image, config='--psm 7 -c tessedit_char_whitelist=0123456789')
		if(len(string) >= 5): break 
	return croped_image, string[-5:]

def extract_class_id(img, horizontal_lines, vertical_lines, score_table_points):
	#crop the limited class id region
	orig = img.copy()
	startX = min(score_table_points[:,0])
	startY = min(score_table_points[:,1])
	limited_image = img[ int(startY / 1.5) : startY , startX+ vertical_lines[3]:startX+ vertical_lines[4]]
	return _digit_detection(limited_image)

#preloaded net, reduce execution time as not loading the net multiple time
def _digit_detection_ver2(image, net):
	orig = image.copy()

	H, W = image.shape[:2]

	(newW, newH) = (320, 320)
	rW = W / float(newW)
	rH = H / float(newH)
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]
	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layer_names = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

	#load the pretrained EAST detector
	#net = cv2.dnn.readNet(east)
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layer_names)

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	#loop over the number of rows
	for y in range(0, numRows):
		#extract the scores, followed by the geometrical data used to derive potential bounding box coordiantes taht surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		#loop over the number of columns 
		for x in range(0, numCols):
			if scoresData[x] < 0.5: continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	boxes = non_max_suppression(np.array(rects), probs=confidences)
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)-10
		startY = int(startY * rH)-10
		endX = int(endX * rW)+10
		endY = int(endY * rH)+10

		croped_image = orig[startY:endY, startX:endX]
		string =  tess.image_to_string(croped_image, config='--psm 7 -c tessedit_char_whitelist=0123456789')
		if(len(string) >= 5): break 
	return croped_image, string[-5:]

def extract_class_id_ver2(img, horizontal_lines, vertical_lines, score_table_points, east_net):
	#crop the limited class id region
	orig = img.copy()
	startX = min(score_table_points[:,0])
	startY = min(score_table_points[:,1])
	limited_image = img[ int(startY / 1.5) : startY , startX+ vertical_lines[3]:startX+ vertical_lines[4]]
	return _digit_detection(limited_image, east_net)


######################### Main excecution #########################
def _main_excecution(img):
	# 1. Extract table
	score_table, score_table_width, score_table_height, score_table_points = _extract_scoreTable(img)
	#cv2.imwrite('score_table_1.jpg', score_table)
	
	# 2. Extract lines on table
	horizontal_lines, vertical_lines = _extract_lines(score_table, score_table_width, score_table_height)
	visualize_score_table = score_table.copy()
	for y in horizontal_lines:
		cv2.line(visualize_score_table, (0,y), (score_table_width, y), (0,0, 255), 2)
	for y in vertical_lines:
		cv2.line(visualize_score_table, (y,0), (y, score_table_height), (0,0, 255), 2)
	#cv2.imwrite('score_table_2.jpg', visualize_score_table)

	
	# 3. Extract text region
	ids, scores = _extract_text(score_table, 2, 5, horizontal_lines, vertical_lines)
	#print(tess.image_to_string(img))

	# 4. Use text region to extract image and feed to CNN
	batch_imgs = _get_batch_imgs(scores)
	

	# 5. extract the class id
	class_id_image, class_id = extract_class_id(img, horizontal_lines, vertical_lines, score_table_points)
	
	# cv2.waitKey(0)
	return ids, class_id
if __name__ == '__main__':
	img = cv2.imread('./data/1.1.png')
	_main_excecution(img)
