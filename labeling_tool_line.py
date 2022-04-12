import cv2
import os
import json
import copy
import pandas as pd
import csv
from sympy import Symbol , solve
import numpy as np
import math
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.ransac import ransac

imgPath = "images/"
eye_label_path = "eye_labels/"

eye_region = 0
iris_region = 0
eye_label = []
origin_value = []
line_value = [] 
switch = True
img_eyeline = np.array([])
x =0
y =0


def draw_circle(event,rx,ry,flags,param):

    global eye_region
    global iris_region
    global eye_label
    global iris_label
    global origin_value
    global line_value
    global switch            # true --> eye , false --> iris
    global img_eyeline
    global x 
    global y 

    point_size = 1
    point=(rx,ry)
    thickness = 2

    if event == cv2.EVENT_LBUTTONDOWN:  
        if eye_region <= 15 :
            eye_region += 1
            b,g,r = img_eye[point[1],point[0]] 
        else:
            switch = False
            iris_region += 1
            b,g,r = img_ir[point[1],point[0]] 
            
        origin_value.append(copy.deepcopy([int(b),int(g),int(r)]))  # 원래 색깔 넣어 두는 곳

        if switch: # eye_region
            point_color = (0, 0, 255) 
            if eye_region <= 2:
                cv2.circle(img_eye, point, point_size, point_color, thickness)
                eye_label.append(copy.deepcopy([rx,ry]))
            else:  
                # 보정하기 
                cdot = copy.deepcopy([rx,ry])
                line_num = eye_region -3 if eye_region <= 9 else 16 - eye_region # 가장 가까이 있는 직선 찾기 
                
                eq_fix_x = x - cdot[0]                           # x고정하고의 거리 , y고정하고의 거리 비교해서 가까운 쪽 직선위로 보정      
                eq_fix_y = y - cdot[1]
                fix_x = solve((line_value[line_num], eq_fix_x))    
                fix_y = solve((line_value[line_num], eq_fix_y))
                if not fix_x :   # 기울기가 0일 경우 해가 없을 것임 --> list가 비어 있을 꺼임   
                    cdot = (int(fix_y[x]),int(fix_y[y]))
                else:
                    revise_x , revise_y = fix_y[x], fix_x[y]
                    x_dis , y_dis= abs(revise_x - rx) , abs(revise_y-ry)
                    if x_dis > y_dis: cdot[1] = int(revise_y) 
                    else : cdot[0] = int(revise_x)
                
                cv2.circle(img_eyeline, cdot, point_size, point_color, thickness)
                eye_label.append(cdot)

        else: # iris_region
            point_color = (200, 125, 0)
            cv2.circle(img_ir, point, point_size, point_color, thickness)
            eye_label.append(copy.deepcopy([rx,ry]))

        print("현재 체크 갯수 --> 눈 주위 점 : {}개 , 홍채 점 : {}개".format(eye_region,iris_region))

        # 처음에 눈쪽 2끝
        
        font=cv2.FONT_HERSHEY_SIMPLEX

        if eye_region == 2 :   # 눈 끝점 잇고 , 수직인 직선 그려주기  # 저장 및 전역 변수 초기화   
            img_eyeline = copy.deepcopy(img_eye)
            cv2.line(img_eyeline,tuple(eye_label[0]),tuple(eye_label[1]),(0,0,255),1)  #  두점 선으로 긋기
            length = np.linalg.norm(np.array(eye_label[0])-np.array(eye_label[1]))     #  두점 사이의 길이 
            id_length = length / 8                             # 8등분 길이
            x_var = (eye_label[0][0] - eye_label[1][0])        # x 변화량
            y_var = (eye_label[0][1] - eye_label[1][1])        # y 변화량 
            gradient = y_var/x_var if x_var != 0 else 0        # 기울기  
            
            start = (eye_label[0][0],eye_label[0][1])            # 첫번째 점
            divide = []

            x = Symbol('x')                                     # from sympy import Symbol , solve --> 사용법은 구글 참고
            y = Symbol('y')                             
            # 직선의 방정식
            eq1 = gradient * (x-start[0]) -y + start[1]         # 초기 두점을 이은 직선의 방정식 

            for idx in range(7):                                # 나는 총 8등분해서 직선 그은 거임
                # 거리 방정식
                eq2 = (start[0] - x ) ** 2 + (start[1] - y ) ** 2 - id_length**2        # 첫번째 점에서 등분한 길이까지 거리방정식
                sol = solve((eq1,eq2),dict=True)                                        # 등분한 길이에 해당하는 점들 찾음
                start = (int(sol[1][x]),int(sol[1][y]))
                divide.append(start)

            line_value = [] 
       
            if gradient == 0:
                for idx in range(1,8):
                    eq_l = x - (eye_label[0][0]+idx*id_length)                           
                    line_value.append(eq_l)
                    cv2.line(img_eyeline,(int(eye_label[0][0]+idx*id_length),0),(int(eye_label[0][0]+idx*id_length),191),(0,255,0),1)
                     
            else:   
                for idx in divide:
                    new_gradient = -1/ gradient                                         # 초기직선에 수직인 기울기
                    eq3 = new_gradient * (x-idx[0]) -y + idx[1]                         # 초기 직선을 등분한 점을 지나는 수직방정식       
                    eq4 = y - 0                                                         
                    eq5 = y - 191                                                       # # 이미지 높이에 맞춰서
                    s = int(solve((eq3,eq4))[x])                                        # 이부분은 직선의 끝이 이미지의 높이의 양끝이 되도록 한거임   
                    e = int(solve((eq3,eq5))[x])  
                    line_value.append(eq3)
                    cv2.line(img_eyeline,(s,0),(e,191),(0,255,0),1)                     # 수직인 직선이 이미지 높이의 시작과 끝에서 그려지게끔 

        if eye_region == 16 and iris_region == 0 :   # 눈 주위 점 입력 다하고 넘어가기 
            switch =  False

            # #ransac algorithm
            # gray = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
            # pnts = list(eye_label[:16])
            # ellipse_params = ransac.FitEllipse_RANSAC(np.array(pnts), gray)
            # img_ellipse = img_ir.copy()
            # cv2.ellipse(img_ellipse,ellipse_params,(0,0,255),1)
            # cv2.imshow('image',img_ellipse)
            
            cv2.imshow('image', img_eyeline)
            cv2.waitKey(500)
            cv2.imshow('image',img_ir)
            return     

        if switch:
            cv2.imshow('image', img_eye) if eye_region < 2 else cv2.imshow('image', img_eyeline)
        else:
            cv2.imshow('image',img_ir) 
        
        if iris_region == 8 :   ####### 홍채 8개점 찍고 ransac 통해서 타원 찾아주기  ######
            gray = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
            pnts = list(eye_label[16:])
            ellipse_params = ransac.FitEllipse_RANSAC(np.array(pnts), gray) # ((centx,centy), (width,height), angle)
            img_ellipse = img_ir.copy()
            cv2.ellipse(img_ellipse,ellipse_params,(0,0,255),2)
            #cv2.ellipse(img_ellipse,(ellipse_params[0],ellipse_params[1],0),(255,0,0),1)
            x = Symbol('x')
            y = Symbol('y')

            center = (int(ellipse_params[0][0]), int(ellipse_params[0][1]))
            length = (int(ellipse_params[1][0]/2),int(ellipse_params[1][1]/2))
            radian = np.radians(int(45/4))  # 180도를 16등분 
            res = np.radians(int(ellipse_params[2]))


            eq_t = (x-center[0])**2 / length[0]**2 +  (y-center[1])**2 / length[1]**2 - 1
            iris_label_top = []
            iris_label_bot = []
            iris_label =[]
            
            for idx in range(0,16):
                gradient = np.tan(-radian*idx + res)
                eq_l = gradient * (x-center[0]) - y + center[1]


            # 회전 변환한 값 가지고 있어 줘야됨  
                s1, s2 = solve((eq_t,eq_l))  

                # rs1_x = int(s1[x])
                # rs1_y = int(s1[y])
                # rs2_x = int(s2[x])  
                # rs2_y = int(s2[y])

                rs1_x = int(- np.cos(res)*(s1[x]-ellipse_params[0][0]) - np.sin(res)*(s1[y]-ellipse_params[0][1]) + ellipse_params[0][0])
                rs1_y = int(-np.sin(res)*(s1[x]-ellipse_params[0][0]) + np.cos(res)*(s1[y]-ellipse_params[0][1]) + ellipse_params[0][1])
                rs2_x = int(- np.cos(res)*(s2[x]-ellipse_params[0][0]) - np.sin(res)*(s2[y]-ellipse_params[0][1])  + ellipse_params[0][0])  
                rs2_y = int(-np.sin(res)*(s2[x]-ellipse_params[0][0]) + np.cos(res)*(s2[y]-ellipse_params[0][1]) + ellipse_params[0][1])

                # 홍채 좌표값 위 아래로 나누기 
                if idx == 0 :
                    if rs1_x <= rs2_x : 
                        iris_label_top.append((rs1_x,rs1_y)) 
                        iris_label_bot.append((rs2_x,rs2_y))
                    else :
                        iris_label_top.append((rs2_x,rs2_y)) 
                        iris_label_bot.append((rs1_x,rs1_y)) 
                else:
                    if rs1_y <= rs2_y: 
                        iris_label_top.append((rs1_x,rs1_y)) 
                        iris_label_bot.append((rs2_x,rs2_y))
                    else:
                        iris_label_top.append((rs2_x,rs2_y)) 
                        iris_label_bot.append((rs1_x,rs1_y))  

                #cv2.line(img_ellipse,(center[0],center[1]),(rs1_x,rs1_y),(0,255,0),1)
                #cv2.line(img_ellipse,(center[0],center[1]),(rs2_x,rs2_y),(0,255,0),1)
                
            for idx in range(0,32): # 데이터 모으기 
                iris_label.append(iris_label_top[idx]) if idx < 16 else iris_label.append(iris_label_bot[idx-16]) 

            iris_label.append(center)
            for idx in range(0,len(iris_label)):
                cv2.circle(img_ellipse, iris_label[idx], point_size, (0,255,255), thickness)
                cv2.putText(img_ellipse, str(idx),iris_label[idx],2,0.3,(255,255,255))

            cv2.putText(img_ellipse, "file index : " + str(image_index+1),(10,10),2,0.5,(0,0,0)) 
            cv2.imshow('image',img_ellipse)
            
    if event == cv2.EVENT_RBUTTONDOWN:     # 한 칸 지우기 
        if switch and iris_region == 0 :
            eye_region -= 1
            eye_region = 0 if eye_region <= 0 else eye_region
            latest = eye_label.pop()
            pc = origin_value.pop()
            if eye_region <2:
                cv2.circle(img_eye, tuple(latest), point_size, tuple(pc), thickness)
                cv2.imshow('image', img_eye)
            else:
                cv2.circle(img_eyeline, tuple(latest), point_size, tuple(pc), thickness)
                cv2.imshow('image', img_eyeline)
        else: 
            if iris_region == 0:
                switch = True
                cv2.imshow('image', img_eyeline)
            else:
                iris_region -= 1
                latest = eye_label.pop()
                pc = origin_value.pop()
                cv2.circle(img_ir, tuple(latest), point_size, tuple(pc), thickness)
                cv2.putText(img_eyeline, "file index : " + str(image_index+1),(10,10),2,0.5,(0,0,0)) 
                cv2.imshow('image', img_eyeline) if switch else cv2.imshow('image',img_ir)
                
        print("현재 체크 갯수 --> 눈 주위 점 : {}개 , 홍채 점 : {}개".format(eye_region,iris_region))
        
if __name__ == '__main__':
    
    images = os.listdir(imgPath)
    
    file_name = "keypoints_labels.csv"
    eye_label_path = os.path.join(eye_label_path,file_name)   
     
    items = ['image_name']
    for i in range(17):
        items.append('eye_region_{}_x'.format(i))
        items.append('eye_region_{}_y'.format(i))
    for i in range(33):
        items.append('iris_region_{}_x'.format(i))
        items.append('iris_region_{}_y'.format(i))

    print("*********************************************************")
    print("*********************************************************")
    print(" 빨간색 16개 , 파란색 8개를 차례 대로 찍어 주시면 됩니다! ")
    print("*** 잘못 찍으셨다면 오른쪽 마우스 눌러 주시면 됩니다! ***")
    print("********** 종료는 ESC를 눌러주시면 됩니다! **************")
    print("*********************************************************")
    print("*********************************************************")
    
    # 시작 지점 찾아주기
    previous = pd.read_csv(eye_label_path)
    if previous.empty:
        start = 0
    else:
        latest = previous.iloc[-1]                                                          # name은 인덱스 인듯 , 값 없는게 있으면 Nan으로 뜨네 
        start = latest.name if latest.isna().sum() >= 1 else latest.name + 1                  # nan 하나당 1값을 가지고 있음


    for image_index in range(start,len(images)):       #
        
        img_path=os.path.join(imgPath,images[image_index])

        print("--------------------------------------------------------")
        print("이미지가 이상할 경우 s를 눌러 넘어가고 파일 이름을 체크해 주세요!")
        print("현재 파일 이름 : {}".format(img_path))

        switch = True
        cv2.namedWindow("image")
        img = cv2.imread(img_path)
        img = cv2.resize(img,dsize=(320,192),interpolation=cv2.INTER_AREA) #  원래 이미지의 2배 키운 거임 # 160x96 --> 320x192
        img_eye = img.copy()
        cv2.putText(img_eye, "file index : " + str(image_index+1),(10,10),2,0.5,(0,0,0)) 
        img_ir = cv2.resize(img,dsize=(320,192),interpolation=cv2.INTER_AREA) #  원래 이미지의 4배 키운 거임 # 160x96 --> 640x384 (640,384)
        cv2.putText(img_ir, "file index : " + str(image_index+1),(10,10),2,0.5,(0,0,0)) 

        cv2.setMouseCallback('image',draw_circle) 
        cv2.putText(img, "file index : " + str(image_index+1),(10,10),2,0.5,(0,0,0)) 
        cv2.imshow('image', img)          # 처음 사진 보여주는 거임
        
        waitkey_num = cv2.waitKeyEx()

        data = []                                #

        if waitkey_num == 115:  # 소문자 s ascii 번호 
            
            with open(eye_label_path,'a',newline='') as csv_f:  # with as 로 열면
                
                writer = csv.writer(csv_f)

                print("파일 저장 중입니다..")

                img_name = img_path            
                data.append(img_name)                             
                
                interior_margine = eye_label[:16]
                eye_endpoint= interior_margine.pop(1)
                interior_margine.insert(8,eye_endpoint)                 
                
                cen_x1, cen_y1 = [], []   # eye center 좌표 계산
                for idx in range(len(interior_margine)):
                    cen_x1.append(interior_margine[idx][0])
                    cen_y1.append(interior_margine[idx][1])
                eye_center = (sum(cen_x1) / len(interior_margine), sum(cen_y1) / len(interior_margine))
                data.append(eye_center[0])                        
                data.append(eye_center[1])                         

                for idx in range(len(interior_margine)):           # 32개       
                    data.append(interior_margine[idx][0])
                    data.append(interior_margine[idx][1])
 
                data.append(iris_label[32][0])                   # 2개  
                data.append(iris_label[32][1])
                                                                   
                for idx in range(len(iris_label)-1):               # 101개의 데이터 
                    data.append(iris_label[idx][0])
                    data.append(iris_label[idx][1])

                writer.writerow(data)
                
                eye_region = 0                                   ##  전역변수 초기화 
                iris_region = 0
                eye_label = []
                origin_value = []
                line_value = [] 
                switch = True

        if waitkey_num == 112:
            with open(eye_label_path,'a',newline='') as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow([0])        
        if waitkey_num == 27:
            exit(1)

    csv_f.close()    
    cv2.destroyAllWindows()
 