import numpy as np
import imageio
from scipy.misc.pilutil import Image
import os
# from recognizeDigit import digit_name1, digit_name2
import scipy.misc as mi

def getDigit1(image, digit_name, stt):
    test = np.asarray(image)
    test.setflags(write=1)
    for x in test:
        i = 0
        for y in x:
            t = 255 - y
            x[i] = t
            i += 1
    for x in test:
        i = 0
        for y in x:
            if y < 40:
                x[i] = 0
            i += 1
    test[0] = 0
    test[1] = 0
    test[2] = 0
    test[3] = 0
    test_width = len(test[0])
    for column in test.T:
        test_weight = len(column)
        break
    test[test_weight - 1] = 100
    for x in test:
        i = 0
        x[0] = 0
        x[1] = 0
        x[2] = 0
        x[3] = 0
        x[4] = 0
        x[5] = 0
        x[len(x)-1] = 0
        x[len(x)-2] = 0
        x[len(x)-3] = 0
        x[len(x)-4] = 0
        x[len(x)-5] = 0
        i += 1
        
    # xoa duong ke ngang
# =============================================================================
#     i = 0
#     for x in test:
#         if i == test_weight - 2:
#             x[3] = 255
#             break
#         i += 1
# =============================================================================
    arr = []
    n = 0
    for x in test:
        if n < test_weight - 3:
            arr.clear()
            for y in x:
                arr.append(y)
            n += 1
            continue
        i = 3
        j = 0
        if x[5] > 30 or x[6] > 30 or x[7] > 30 or x[8] > 30 or x[9] > 30:
            a = 0
            for y in x:
                if a < 40:
                    x[i] = 0
                    arr[i] = 0
                else:
                    arr[i] = x[i]
                    break
                # print(arr)
                i += 1
                if i > len(x) - 1:
                    break
                else:
                    a = arr[i]
                    # print(a)
            k = len(x) - 3
            for y in x:
                if arr[k - j] < 40:
                    x[k - j] = 0
                    arr[k - j] = 0
                else:
                    arr[k - j] = x[k - j]
                    break
                # print(arr)
                j += 1
        else:
            arr.clear()
            for y in x:
                arr.append(y)
                i += 1
    
    # tinh kich thuoc so
    pixel_a = 0
    i = 0
    for column in test.T:
        j = 0
        for x in reversed(column):
            if j > len(column) - 10:
                break
            if x > 30:
                pixel_a = i
                break
            j += 1
        if pixel_a > 0:
            break
        i += 1
    digit_width = 0
    i = 0
    for column in test.T:
        if i < pixel_a + 13:
            i += 1
            continue
        j = 0
        for x in column:
            if j < 3:
                j += 1
                continue
            if j == len(column) - 12:
                break
            if x == 0:
                check = 1
                j += 1
                continue
            else:
                check = 0
                break
        if check == 1:
            digit_width = 8
            break
        else:
            digit_width = 13
            break
        i += 1
    pixel_b = 0
    i = 0
    for column in test.T:
        if i < pixel_a + digit_width:
            i += 1
            continue
        j = 0
        for x in column:
            if j < 3:
                j += 1
                continue
            if j == len(column) - 13:
                break
            if x == 0:
                check = 1
                j += 1
                continue
            else:
                check = 0
                break
        if check == 1:
            pixel_b = i
            break
        i += 1
    # pixel_b += 1
    if pixel_b > pixel_a + 30:
        pixel_b = pixel_a + 20
        
    # get digit
    # digit1 = np.empty((0,20), int)
    digit1 = []
    i = 0
    k = 1
    for x in test:
        if i < 3:
            i += 1
            continue
        j = 0
        if i > test_weight - 3:
            # arr1[pixel_b - pixel_a + 1] = 0
            x[pixel_b] = 0
            x[pixel_b - 1] = 0
        arr1 = []
        for y in x:
            if j < pixel_a - 3:
                j += 1
                continue
            arr1.append(y)
            j += 1
            if j == pixel_b:
                arr1.append(0)
                arr1.append(0)
                arr1.append(0)
                break
        digit1.append(arr1)
        # digit1 = np.append(digit1, arr1[0], axis=0)
        i += 1
    arr2 = []
    for x in arr1:
        arr2.append(0)
    digit1.append(arr2)
    digit1.append(arr2)
    # imageio.imwrite('input/digit1.jpg', np.asarray(digit1))
    mi.imsave('temp/digit1.jpg', digit1)
    dir = "./temp/" + str(digit_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_location = dir + "/" + str(stt) + "a.jpg"
    mi.imsave(save_location, digit1)
# =============================================================================
#     digit_img = Image.open('input/digit1.jpg').convert('L')
#     digit_arr = np.asarray(digit_img)     
#     digit_arr.setflags(write=1)
# =============================================================================
    
def getDigit2(image, digit_name, stt):
    test = np.asarray(image)
    test.setflags(write=1)
    for x in test:
        i = 0
        for y in x:
            t = 255 - y
            x[i] = t
            i += 1
    for x in test:
        i = 0
        for y in x:
            if y < 40:
                x[i] = 0
            i += 1
    test[0] = 0
    test[1] = 0
    test[2] = 0
    test[3] = 0
    test_width = len(test[0])
    for column in test.T:
        test_weight = len(column)
        break
    test[test_weight - 1] = 255
    for x in test:
        i = 0
        x[0] = 0
        x[1] = 0
        x[2] = 0
        x[len(x)-1] = 0
        x[len(x)-2] = 0
        x[len(x)-3] = 0
        i += 1
    
    # xoa duong ke ngang
    arr = []
    for x in test:
        i = 3
        j = 0
        if x[3] > 30 or x[4] > 30 or x[5] > 30:
            a = 0
            for y in x:
                if a < 40:
                    x[i] = 0
                    arr[i] = 0
                else:
                    arr[i] = x[i]
                    break
                # print(arr)
                i += 1
                if i > len(x) - 1:
                    break
                else:
                    a = arr[i]
                    # print(a)
            k = len(x) - 3
            for y in x:
                if arr[k - j] < 40:
                    x[k - j] = 0
                    arr[k - j] = 0
                else:
                    arr[k - j] = x[k - j]
                    break
                # print(arr)
                j += 1
        else:
            arr.clear()
            for y in x:
                arr.append(y)
                i += 1
    
    # tinh kich thuoc so
    pixel_a = 0
    i = 0
    for column in reversed(test.T):
        j = 0
        for x in column:
            if j > len(column) - 5:
                break
            if x > 30:
                pixel_a = i
                break
            j += 1
        if pixel_a > 0:
            break
        i += 1
    digit_width = 0
    i = 0
    for column in reversed(test.T):
        if i < pixel_a + 13:
            i += 1
            continue
        j = 0
        for x in column:
            if j < 3:
                j += 1
                continue
            if j == len(column) - 12:
                break
            if x == 0:
                check = 1
                j += 1
                continue
            else:
                check = 0
                break
        if check == 1:
            digit_width = 5
            break
        else:
            digit_width = 13
            break
        i += 1
    pixel_b = 0
    i = 0
    for column in reversed(test.T):
        if i < pixel_a + digit_width:
            i += 1
            continue
        j = 0
        for x in column:
            if j < 3:
                j += 1
                continue
            if j == len(column) - 10:
                break
            if x == 0:
                check = 1
                j += 1
                continue
            else:
                check = 0
                break
        if check == 1:
            pixel_b = i
            break
        i += 1
    if pixel_b > pixel_a + 30:
        pixel_b = pixel_a + 20
        
    # get digit2
    digit2 = []
    i = 0
    k = 1
    for x in test:
        if i < 3:
            i += 1
            continue
        j = 0
        arr1 = []
        for y in reversed(x):
            if j < pixel_a - 2:
                j += 1
                continue
            arr1.append(y)
            j += 1
            if j == pixel_b:
                arr1.append(0)
                arr1.append(0)
                arr1.append(0)
                break
# =============================================================================
#         if i > test_weight - 4:
#             arr1[pixel_b - pixel_a + 2] = 0
# =============================================================================
        digit2.append(arr1)
        i += 1
    arr2 = []
    for x in arr1:
        arr2.append(0)
    digit2.append(arr2)
    digit2.append(arr2)
    # imageio.imwrite('input/digit2.jpg', np.asarray(digit))
    mi.imsave('temp/digit2.jpg', digit2)
    # fix digit2  
    digit2_img = Image.open('temp/digit2.jpg').convert('L')
    digit2_arr = np.asarray(digit2_img)
    digit2_arr.setflags(write=1)
    digit2_arr_new = []
    for x in digit2_arr:
        arr2 = []
        for y in reversed(x):
            arr2.append(y)
        digit2_arr_new.append(arr2)
    mi.imsave('temp/digit2.jpg', digit2_arr_new)
    dir = "./temp/" + str(digit_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_location = dir + "/" + str(stt) + "b.jpg"
    mi.imsave(save_location, digit2_arr_new)