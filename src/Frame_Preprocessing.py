import numpy as np
import cv2

class Frame_Preprocessing():
    Left_points = np.array([(0,290), (265, 215), (540, 385), (405,670)])
    Right_points = np.array([ (1280, 290), (850,670), (725,385), (995, 215)])
    BGR_RED = (0,0,255)
    
    def __init__(self) -> None:
        self.Video = cv2.VideoCapture('data/Highway.mp4')
        assert True == self.Video.isOpened(), f"Video not found"
    
    def Preprocessing(self, Frame_display: np.array) -> np.array:
        Gray = cv2.cvtColor(Frame_display, cv2.COLOR_RGB2GRAY) + 1
        
        MASK_LEFT = np.zeros_like(Gray)
        MASK_LEFT = cv2.fillPoly(MASK_LEFT, [self.Left_points], (255,255,255))
        Cropped_frame_LEFT = cv2.bitwise_and(Frame_display + (1,1,1), Frame_display + (1,1,1), mask=MASK_LEFT) 
        Cropped_frame_LEFT = np.where(Cropped_frame_LEFT == (0,0,0), (255,255,255), Cropped_frame_LEFT ).astype(np.uint8)
        
        MASK_RIGHT = np.zeros_like(Gray)
        MASK_RIGHT = cv2.fillPoly(MASK_RIGHT, [self.Right_points], (255,255,255))
        Cropped_frame_RIGHT = cv2.bitwise_and(Frame_display + (1,1,1), Frame_display + (1,1,1), mask=MASK_RIGHT) 
        Cropped_frame_RIGHT = np.where(Cropped_frame_RIGHT == (0,0,0), (255,255,255), Cropped_frame_RIGHT ).astype(np.uint8)
        
        #cv2.imshow("Cropped_frame_L", Cropped_frame_LEFT)
        #cv2.imshow("Cropped_frame_R", Cropped_frame_RIGHT)
        #cv2.waitKey(0)
        
        return (Cropped_frame_LEFT,Cropped_frame_RIGHT)
    
    def Frame_Extraction(self):

        while (self.Video.isOpened()):    
            status, frame  = self.Video.read()
            
            if status:
                yield frame
            else:
                break