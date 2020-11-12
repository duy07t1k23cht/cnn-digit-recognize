import numpy as np
from scipy.misc.pilutil import Image
import xlwt
import cv2
import os
import scipy.misc as mi
from keras.models import model_from_json
from processData import processData
from cropDigit import getDigit1, getDigit2
import utils
import pytesseract as tess 


def reStoreModel():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("loaded model from disk")
    return loaded_model

def recognizeDigit(cnnModel, digit_location):
    digit_img = Image.open(digit_location).convert('L')
    digit_arr = np.asarray(digit_img)
    digit_arr.setflags(write=1)
    digit_arr = cv2.resize(digit_arr, (28, 28))
    digit_arr[0] = 0
    digit_arr[1] = 0
    digit_arr[2] = 0
    for x in digit_arr:
        i = 0
        for y in x:
            if y < 30:
                x[i] = 0
            i += 1
    
    # img = np.zeros((20,20,3), np.uint8)
    # mi.imsave('test.jpg', test)
    
    digit_arr = digit_arr / 255.0
    digit_arr = digit_arr.reshape(-1,28,28,1)
    # predict results
    results = cnnModel.predict(digit_arr)
    # select the indix with the maximum probability
    accuracy  = np.amax(results,axis = 1)
    results = np.argmax(results,axis = 1)
    ketqua = [results[0], accuracy[0]]
    # results = pd.Series(result,name="Label")
    return ketqua

print("--> running")
model = reStoreModel()
print("loading the east detector")
east_net = cv2.dnn.readNet("frozen_east_text_detection.pb")
print("loaded")
inputdir = "./input"
outputdir = "./output"
filelist = os.listdir(inputdir)
filelist = sorted(filelist ,key=lambda x: x[1])
filelist1 = []
check = 0
for i in range(1, len(filelist)):
    n1 = ""
    n2 = ""
    if i == len(filelist) - 1:
        name = filelist[i][:-4]
        for j in range(0, len(name)):
            if name[j] == ".":
                for k in range(0, j):
                    n1 += name[k]
                break
        name = filelist[i-1][:-4]
        for j in range(0, len(name)):
            if name[j] == ".":
                for k in range(0, j):
                    n2 += name[k]
                break
        if n1 == n2:
            filelist1.append([filelist[i-1], filelist[i]])
        else:
            filelist1.append([filelist[i-1]])
            filelist1.append([filelist[i]])
        break
    
    if check == 1:
        check = 0
        continue
    name = filelist[i-1][:-4]
    for j in range(0, len(name)):
        if name[j] == ".":
            for k in range(0, j):
                n1 += name[k]
            break
    name = filelist[i][:-4]
    for j in range(0, len(name)):
        if name[j] == ".":
            for k in range(0, j):
                n2 += name[k]
            break
    if n1 == n2:
        filelist1.append([filelist[i-1], filelist[i]])
        check = 1
    else:
        filelist1.append([filelist[i-1]])

print(filelist1)

for name in filelist1:
    book = xlwt.Workbook()
    sh = book.add_sheet("Sheet 1")
    sh.write(0, 0, 'STT')
    sh.write(0, 1, 'MSSV')
    sh.write(0, 2, 'Diem')
    sh.write(0, 3, 'Do chinh xac')
    sh.write(0, 4, 'Check?')
    sh.write(0, 5, 'Ma so lop')
    stt = 1
    for element in name:
        print("processing " + element)
        direction = './input/' + element
        if len(name) == 2:
            element = element[:-6]
        elif len(name) == 1:
            element = element[:-4]
        output_filename =  element + "_result.xls"
        digit_name = element
        # Start recognize digit
        input_img = Image.open(direction).convert('L')
        # Process input data
        coordinates = processData(direction)
        
        # check empty case
        for x in coordinates:
            # for debug
        # =============================================================================
        #     if stt != 15:
        #         stt += 1
        #         continue
        # =============================================================================
            
            img = input_img.crop(x)
            check = np.asarray(img)
            check.setflags(write=1)
            for x in check:
                i = 0
                for y in x:
                    t = 255 - y
                    x[i] = t
                    i += 1
            for x in check:
                i = 0
                for y in x:
                    if y < 40:
                        x[i] = 0
                    i += 1
            for x in check.T:
                check_weigh = len(x)
                break
            i = 0
            j = 0
            k = False
            for x in check:
                if i == int(check_weigh/2):
                    for y in x:
                        if j < 10 or j > len(x) - 10:
                            j += 1
                            continue
                        if y == 0:
                            k = False
                        else: 
                            k = True
                            break
                        j += 1
                i += 1
            if k == False:
                digit1 = [0, 1.0]
                digit2 = [0, 1.0]
            else:
                # get and recognize digit
                getDigit1(img, digit_name, stt)
                getDigit2(img, digit_name, stt)
                digit1_location = 'temp/digit1.jpg'
                digit2_location = 'temp/digit2.jpg'
                digit1 = recognizeDigit(model, digit1_location)
                digit2 = recognizeDigit(model, digit2_location)
                if digit2[0] != 5:
                    digit2[0] = 0
            sh.write(stt, 0, stt)
            sh.write(stt, 2, str(str(digit1[0]) + ',' + str(digit2[0])))
            sh.write(stt, 3, str(round(digit1[1], 4)))
            if digit1[1] < 0.5:
                sh.write(stt, 4, 'check')
            # print("Ket qua: %s,%s (%s)" %(digit1[0], digit2[0], digit1[1]))
            stt += 1
        
    #processing student's ID and class's ID
    stt =1 
    for element in name:
        direction = './input/' + element
        if len(name) == 2:
            element = element[:-6]
        elif len(name) == 1:
            element = element[:-4]
        output_filename =  element + "_result.xls"

        img = cv2.imread(direction)
        student_ids, class_id = utils._main_excecution(img)
        
        for student_id in student_ids:
            student_id_string =  tess.image_to_string(student_id, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            sh.write(stt, 1, student_id_string)
            stt += 1
    sh.write(1, 5, class_id)
    book.save('output/' + output_filename)
print("--> done")
