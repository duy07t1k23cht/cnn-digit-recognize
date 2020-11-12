import numpy as np
import imageio
import cv2

print(cv2.__version__)

def processData(direction):
    input_img = cv2.imread(direction)
    gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=40, maxLineGap=1)
    arr_lines = []
    for x in lines:
        for y in x:
            arr_lines.append([y[0], y[1], y[2], y[3]])
    lines_x = []
    lines_y = []
    for x in arr_lines:
        if abs(x[0] - x[2]) < 5:
            lines_x.append(x)
        else:
            lines_y.append(x)
    lines_x.sort()
    lines_y = sorted(lines_y ,key=lambda x: x[1])
    for x in edges:
        length_x = len(x)
        break
    for y in edges.T:
        length_y = len(y)
        break
        
    lines_x1 = []
    x1 = lines_x[1][0]
    x_min = lines_x[0][0]
    x_max = lines_x[0][2]
    for i in range(0, len(lines_x) - 2):
        if x1 - lines_x[i][0] < 5:
            x1 = lines_x[i + 2][0]
            if i == len(lines_x) - 3:
                x_max = lines_x[i + 2][2]
                lines_x1.append([x_min, 0, x_max, length_y])
            continue
        else:
            x_max = lines_x[i][2]
            lines_x1.append([x_min, 0, x_max, length_y])
            x1 = lines_x[i+2][0]
            x_min = lines_x[i + 1][0]
            x_max = lines_x[i + 1][2]
            
    lines_y1 = []
    j = 0
    y1 = lines_y[1][1]
    for i in range(0, len(lines_y) - 2):
        if y1 - lines_y[i][1] < 5:
            y1 = lines_y[i + 2][1]
            if i == len(lines_y) - 3:
                x = 0
                y = 0
                z = 0
                t = 0
                for k in range(j, i + 1):
                    x += lines_y[k][0]
                    y += lines_y[k][1]
                    z += lines_y[k][2]
                    t += lines_y[k][3]
                x_avg = int(x / (i + 1 - j))
                y_avg = int(y / (i + 1 - j))
                z_avg = int(z / (i + 1 - j))
                t_avg = int(t / (i + 1 - j))
                lines_y1.append([x_avg, y_avg, z_avg, t_avg])
            continue
        else:
            x = 0
            y = 0
            z = 0
            t = 0
            for k in range(j, i + 1):
                x += lines_y[k][0]
                y += lines_y[k][1]
                z += lines_y[k][2]
                t += lines_y[k][3]
            if i == j :
                x_avg = lines_y[i][0]
                y_avg = lines_y[i][1]
                z_avg = lines_y[i][2]
                t_avg = lines_y[i][3]
            else:
                x_avg = int(x / (i + 1 - j))
                y_avg = int(y / (i + 1 - j))
                z_avg = int(z / (i + 1 - j))
                t_avg = int(t / (i + 1 - j))
            lines_y1.append([x_avg, y_avg, z_avg, t_avg])
            j = i + 1
            y1 = lines_y[i+2][1]
    
    lines_y2 = []
    for i in range(0, len(lines_y1) - 1):
        if lines_y1[i+1][1] - lines_y1[i][1] < 50:
            lines_y2.append(lines_y1[i])
        if i == len(lines_y1) - 2:
            lines_y2.append(lines_y1[i + 1])
    lines_y1.clear()
    lines_y1 = lines_y2
    lines_y1.remove(lines_y1[0])
    lines_x.clear()
    lines_y.clear()
    lines_y = lines_y1
    lines_x.append(lines_x1[4])
    lines_x.append(lines_x1[5])
    
    
    lines_x1 = []
    for x in lines_x:
        lines_x1.append(x)
            
    lines_y1 = []
    for y in lines_y:
        if y[1] == y[3]:
            lines_y1.append([0, y[1], length_x, y[1]])
        else:
            
            a = (y[1] - y[3]) / (y[0] - y[2])
            b = (y[0] * y[3] - y[1] * y[2]) / (y[0] - y[2])
            lines_y1.append([0, int(b), length_x, int(a*length_x + b)])
    
    for x in lines_x1:
        cv2.line(input_img, (x[0],x[1]), (x[2],x[3]), (0,0,255), 1)
    for y in lines_y1:
        cv2.line(input_img, (y[0],y[1]), (y[2], y[3]), (0,0,255), 1)
        
    cv2.imwrite('./temp/pic1.jpg',input_img)
    
    coordinate1 = []
    coordinate2 = []
    i = 0
    for x in lines_x:
        if x[0] == x[2]:
            x1 = x[0]
            for y in lines_y:
                if y[1] == y[3]:
                    y1 = y[1]
                    xy = []
                    xy.append(x1)
                    xy.append(y1)
                    if i == 0:
                        coordinate1.append(xy)
                    else:
                        coordinate2.append(xy)
                else:
                    a1 = (y[1] - y[3]) / (y[0] - y[2])
                    b1 = (y[0] * y[3] - y[1] * y[2]) / (y[0] - y[2])
                    y1 = int(a1 * x1 + b1)
                    xy = []
                    xy.append(x1)
                    xy.append(y1)
                    if i == 0:
                        coordinate1.append(xy)
                    else:
                        coordinate2.append(xy)
        else:
            for y in lines_y:
                if y[1] == y[3]:
                    y1 = y[1]
                    a1 = (x[1] - x[3]) / (x[0] - x[2])
                    b1 = (x[0] * x[3] - x[1] * x[2]) / (x[0] - x[2])
                    x1 = int((y1 - b1) / a1)
                    xy = []
                    xy.append(x1)
                    xy.append(y1)
                    if i == 0:
                        coordinate1.append(xy)
                    else:
                        coordinate2.append(xy)
                else:
                    a1 = (x[1] - x[3]) / (x[0] - x[2])
                    b1 = (x[0] * x[3] - x[1] * x[2]) / (x[0] - x[2])
                    a2 = (y[1] - y[3]) / (y[0] - y[2])
                    b2 = (y[0] * y[3] - y[1] * y[2]) / (y[0] - y[2])
                    x1 = int((b2 - b1) / (a1 - a2))
                    y1 = int((a1 * b2 - a2 * b1) / (a1 - a2))
                    xy = []
                    xy.append(x1)
                    xy.append(y1)
                    if i == 0:
                        coordinate1.append(xy)
                    else:
                        coordinate2.append(xy)
        i += 1
    
    for y in reversed(coordinate1):
        coordinate1.remove(y)
        break
    for x in coordinate2:
        coordinate2.remove(x)
        break
    for x in coordinate1:
        t = x[1] + 2
        x[1] = t
    for x in coordinate2:
        t = x[1] + 2
        x[1] = t
        
    coordinates = []
    i = 0
    for x in coordinate1:
        coor = []
        coor.append(x[0])
        coor.append(x[1])
        j = 0
        for y in coordinate2:
            if j == i:
                coor.append(y[0])
                coor.append(y[1])
            j += 1
        i += 1
        coordinates.append(coor)
        
    return coordinates    