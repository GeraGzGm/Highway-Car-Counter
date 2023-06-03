from ultralytics import YOLO

import time
import cv2
import numpy as np
from math import sqrt, pow

from Frame_Preprocessing import Frame_Preprocessing
from Kalman_Fillter import KF

#Equation for the euclidean distance
Euclidean = lambda a,b: sqrt( pow(b[0]-a[0], 2) + pow(b[1]-a[1] ,2))

class Car_Counter():
    Measurement_variance = 0.01 ** 2 
    ULTRALYTIC_CAR = 2
    
    def __init__(self) -> None:
        self.FP = Frame_Preprocessing()
        self.model = YOLO('yolov8n.pt')
        
        self.Cars_Left = []
        self.Cars_Right = []
        self.N_cars_Left = 0
        self.N_cars_Right = 0
        
    def Draw_Circle(self, img: np.array, center: np.array, color: tuple) -> np.array:
        img = cv2.circle(img, (center[0],center[1]), 4, color, 2)
        return img
    
    def Draw_contours(self, img: np.array) -> np.array:
        img = cv2.polylines(img, [self.FP.Left_points], True, (0,255,0), 1)
        img = cv2.polylines(img, [self.FP.Right_points], True, (0,255,0), 1)
        return img
    
    def Compute_Counting(self, image: np.array, cropped_image: np.array, side: str) -> np.array:
        sTime = time.time()
        result = self.model(cropped_image,  conf=0.35, verbose=False)
        #annotated_frame = result[0].plot()
        
        for box in result[0].boxes:
            Class = box.cls
            
            if Class == self.ULTRALYTIC_CAR:
                xyxy = np.array(box.xyxy[0], dtype=np.int16)

                #x: column, y: row
                x0, y0 = xyxy[0], xyxy[1]
                x1, y1 = xyxy[2], xyxy[3]

                center = (int((x1+x0)/2),int((y1+y0)/2))

                box1 = np.array([(x0,y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.int16)
                image = cv2.polylines(image, np.int32([box1]), True, (0,0,255), 2)

                #Create KF object 
                _kf= KF(x0 = center[0], y0 =  center[1], dot_x0 = 0.1, dot_y0 = 0.1, a_variance = 1)
                is_Already = False

                if side == "LEFT":
                    if self.Cars_Left != []:
                        for car in self.Cars_Left:
                            p_car = car["KF"].pos

                            if Euclidean(center, p_car) < 50:
                                is_Already = True
                                car["KF"].Update(np.array([center[0], center[1]]) + np.random.randn() * np.sqrt(self.Measurement_variance), [self.Measurement_variance, self.Measurement_variance])
                                car["KF"].Prediction(time.time() - sTime)
                                p_center = car["KF"].pos

                    if self.Cars_Left == [] or not is_Already:
                        _kf.Update(np.array([center[0], center[1]]) + np.random.randn() * np.sqrt(self.Measurement_variance), [self.Measurement_variance, self.Measurement_variance])
                        _kf.Prediction(time.time() - sTime)

                        p_center = _kf.pos  
                        self.N_cars_Left += 1
                        self.Cars_Left.append({"KF": _kf, "R_Center": center, "Number": self.N_cars_Left})    
                
                elif side == "RIGHT":
                    if self.Cars_Right != []:
                        for car in self.Cars_Right:
                            p_car = car["KF"].pos

                            if Euclidean(center, p_car) < 50:
                                is_Already = True
                                car["KF"].Update(np.array([center[0], center[1]]) + np.random.randn() * np.sqrt(self.Measurement_variance), [self.Measurement_variance, self.Measurement_variance])
                                car["KF"].Prediction(time.time() - sTime)
                                p_center = car["KF"].pos

                    if self.Cars_Left == [] or not is_Already:
                        _kf.Update(np.array([center[0], center[1]]) + np.random.randn() * np.sqrt(self.Measurement_variance), [self.Measurement_variance, self.Measurement_variance])
                        _kf.Prediction(time.time() - sTime)

                        p_center = _kf.pos  
                        self.N_cars_Right += 1
                        self.Cars_Right.append({"KF": _kf, "R_Center": center, "Number": self.N_cars_Right})            

                image = self.Draw_Circle(image, center, (0,0,255))
                image = self.Draw_Circle(image, (int(p_center[0]), int(p_center[1])), (255,0,0))

        return image
    
    def Run(self) -> None:
        Images = self.FP.Frame_Extraction()
        
        for image in Images:
            image_L, image_R = self.FP.Preprocessing(image)
            
            image = self.Draw_contours(image)
            image = self.Compute_Counting(image, image_L, "LEFT")
            image = self.Compute_Counting(image, image_R, "RIGHT")
             
            text_Number_Cars = f"Number of cars: {self.N_cars_Left}"
            image = cv2.putText(image, text_Number_Cars, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            text_Number_Cars = f"Number of cars: {self.N_cars_Right}"
            image = cv2.putText(image, text_Number_Cars, (900,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            
            cv2.imshow("YOLOv8 Inference", image)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

def main(*args, **kwargs):
    CC = Car_Counter()
    CC.Run()
    
