from ipaddress import ip_address
from tkinter import Y
from cv2 import STEREO_BM_PREFILTER_XSOBEL
from numpy import disp
from pythonosc import udp_client, osc_server
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer



import mediapipe as mp 
mp_pose = mp.solutions.pose
import time


class Landmark:
    def __init__(self,lm):

        self.lm = lm

    def data(self,label):
        
        landmark = [
            getattr(mp_pose.PoseLandmark,label).name,\
            getattr(mp_pose.PoseLandmark,label).value,\
            self.lm[getattr(mp_pose.PoseLandmark,label).value].x, \
            self.lm[getattr(mp_pose.PoseLandmark,label).value].y,  
            self.lm[getattr(mp_pose.PoseLandmark,label).value].z,
            self.lm[getattr(mp_pose.PoseLandmark,label).value].visibility] 
        return landmark

    def distance(self, label1, label2, axis):
        dict = {
            'x':2,
            'y':3,
            'z':4
            }
  
        i = dict['{}'.format(axis)]

        landmark1 = self.data(label1)[i]
        landmark2 = self.data(label2)[i]
 
        return abs(landmark1 - landmark2)

    def average(self):

        landmarks = self.lm
        # Average landmark coordinates
        lm_x = []
        lm_y = []
        lm_z = []
        # Average landmark visibility
        lm_v = []
        for coord in range(len(landmarks)):
            lm_x.append(landmarks[coord].x)
            lm_y.append(landmarks[coord].y)
            lm_z.append(landmarks[coord].z)
            lm_v.append(landmarks[coord].visibility)
     
        av_lm_x = sum(lm_x)/len(landmarks)
        av_lm_y = sum(lm_y)/len(landmarks)
        av_lm_z = sum(lm_z)/len(landmarks)
        av_lm_v = sum(lm_v)/len(landmarks)

        return [av_lm_x, av_lm_y, av_lm_z, av_lm_v]

class SendOSC:
    def __init__(self):
        self.ip = "127.0.0.1"

    def landmark_data(self, landmark, port):
        label = landmark[0]
        id = landmark[1]
        coordinates = landmark[2:5]

        # Create client 
        client = udp_client.SimpleUDPClient(self.ip,port)
        client = client.send_message("/" + str(label) + "/" + str(id),coordinates)
        #time.sleep(1)
        return client

    def data(self, label, data, port):
        client = udp_client.SimpleUDPClient(self.ip,port)
        client = client.send_message(str(label),data)
        return client

    def av_location(self,landmarks,port):
        # Average landmark coordinates
        lm_x = []
        lm_y = []
        lm_z = []

        # Average landmark visibility
        for lm in range(len(landmarks)):
            lm_x.append(landmarks[lm].x)
            lm_y.append(landmarks[lm].y)
            lm_z.append(landmarks[lm].z)
                
        av_lm_x = sum(lm_x)/len(landmarks)
        av_lm_y = sum(lm_y)/len(landmarks)
        av_lm_z = sum(lm_z)/len(landmarks)
        
        # List of average landmark location
        coordinates = [av_lm_x, av_lm_y, av_lm_z]

        label = 'Av_location'
        client = udp_client.SimpleUDPClient(self.ip,port)
        client = client.send_message(str(label),coordinates)
        return client





