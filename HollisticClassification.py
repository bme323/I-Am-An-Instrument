from cProfile import label
from calendar import c
from cmath import pi
from gettext import install
from math import atan2
from operator import delitem
import cv2
from cv2 import circle 
import mediapipe as mp 
import time
from sklearn import pipeline
from sklearn.metrics import classification_report
from sympy import im 
from send_OSC import SendOSC, Landmark
from pythonosc import udp_client
import matplotlib.pyplot as plt
import pandas as pd

import argparse

from pythonosc import dispatcher
from pythonosc import osc_server
import csv
import numpy as np
from regex import B

import numpy as np
import csv
import os
mp_holistic = mp.solutions.holistic 
mp_pose = mp.solutions.pose 
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def PoseEstimation():
    # Get webcam input
    cap = cv2.VideoCapture(1) # 'Videos/no-leg-ball.mp4'
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #800, 1024, 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #600, 780, 1080

    #`Initialise time and fps variables 
    time_start = 0
    fps = 0
    frame = 0
    landmark_list = []
    av_visibility = []
    fps_array = []

    # Begin new instance of mediapipe feed
    with mp_holistic.Holistic(
        model_complexity = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as holistic:

        while cap.isOpened():

            isTrue, image = cap.read() 

            if not isTrue:
                print("Empty camera frame")
                continue 

            # Improve performance 
            image.flags.writeable = False
            # Recolour image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Make detection and store in output array
            output = holistic.process(image)
            image.flags.writeable = True
            # Recolour to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            h,w,c = image.shape
            
            # Draw landmarks on image
            mp_draw.draw_landmarks(
                image,
                output.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks
            
            mp_draw.draw_landmarks(
                image,
                output.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks

            mp_draw.draw_landmarks(
                image,
                output.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks

            # Extract available landmarks
            # for lm in mp_pose.PoseLandmark: print(lm)
            try:
                # extract normalised coordinates
                pose_norm = np.array([[res.x, res.y, res.z, res.visibility] for res in output.pose_landmarks.landmark]).flatten() if output.pose_landmarks else np.zeros(132)
                lh_norm = np.array([[res.x, res.y, res.z] for res in output.left_hand_landmarks.landmark]).flatten() if output.left_hand_landmarks else np.zeros(21*3)
                rh_norm = np.array([[res.x, res.y, res.z] for res in output.right_hand_landmarks.landmark]).flatten() if output.right_hand_landmarks else np.zeros(21*3)
                
                # extract world coordinates
                pose_world = np.array([[res.x, res.y, res.z, res.visibility] for res in output.pose_world_landmarks.landmark]).flatten() if output.pose_world_landmarks else np.zeros(132)

                test = np.concatenate([pose_world,lh_norm,rh_norm])
                print(test)
            
            except:
                pass # pass if landmarks detected

            # Display 
            # Mirror image for webcam display
            image = cv2.flip(image,1)

            # Calculate fps
            time_end = time.time()
            dt = time_end - time_start
            fps = 1/dt
            time_start = time_end
            # Draw fps onto image
            show_fps = "FPS: {:.3} ".format(fps)

            fps_array.append(fps)

            cv2.putText(image, show_fps, (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            # Visualise image and flip display
            cv2.imshow('Video', image)

            # break loop and close windows if q key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return output
#np.savetxt('right_heel_05.csv',right_heel_list)
#PoseEstimation()

def create_csv():
    # Create headings for CSV file
    headings = ['Class']
    # 33 landmarks for pose
    for val in range(1,34):
        headings += ['X{}'.format(val),'Y{}'.format(val),'Z{}'.format(val),'V{}'.format(val)]
    # 21 for rh and lh
    for val in range(1,22):
        headings += ['XL{}'.format(val),'YL{}'.format(val),'ZL{}'.format(val)]
    for val in range(1,22):
        headings += ['XR{}'.format(val),'YR{}'.format(val),'ZR{}'.format(val)]

    # Write csv file
    with open('landmark_data.csv', mode = 'w', newline = '') as file:
        write_csv = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        write_csv.writerow(headings)
    return 
#create_csv()

def export_to_csv(input,user,class_name,no_sequences, sequence_length,wait_time):
    # Get webcam input
    cap = cv2.VideoCapture(input) # 'Videos/no-leg-ball.mp4'
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #800, 1024, 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #600, 780, 1080

    path = 'Class-Images'

    frame = 0
    
    # Begin new instance of mediapipe feed
    with mp_holistic.Holistic(
        model_complexity = 1,
        min_detection_confidence = 0.8,
        min_tracking_confidence = 0.8) as holistic:
        toc = time.perf_counter()
        i=0
              
        # loop through number of sequences for each class
        for sequence in range(no_sequences):

            # loop through sequence length (number of frames per sequence)            
            for frame_num in range(sequence_length):

                isTrue, image = cap.read() 
                if not isTrue:
                    print("Empty camera frame")
                    continue 
                # Improve performance 
                image.flags.writeable = False
                # Recolour image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Make detection and store in output array
                output = holistic.process(image)
                image.flags.writeable = True
                # Recolour to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                h,w,c = image.shape
                
                # Draw landmarks on image
                mp_draw.draw_landmarks(
                    image,
                    output.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks
                
                mp_draw.draw_landmarks(
                    image,
                    output.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks

                mp_draw.draw_landmarks(
                    image,
                    output.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks

                # Extract available landmarks
                # for lm in mp_pose.PoseLandmark: print(lm)
                try:
                    # extract normalised coordinates
                    pose_norm = np.array([[res.x, res.y, res.z, res.visibility] for res in output.pose_landmarks.landmark]).flatten() if output.pose_landmarks else np.zeros(132)
                    lh_norm = np.array([[res.x, res.y, res.z] for res in output.left_hand_landmarks.landmark]).flatten() if output.left_hand_landmarks else np.zeros(21*3)
                    rh_norm = np.array([[res.x, res.y, res.z] for res in output.right_hand_landmarks.landmark]).flatten() if output.right_hand_landmarks else np.zeros(21*3)
                    
                    # extract world coordinates
                    pose_world = np.array([[res.x, res.y, res.z, res.visibility] for res in output.pose_world_landmarks.landmark]).flatten() if output.pose_world_landmarks else np.zeros(132)

                    # create list of landmark coordinates
                    holistic_row = list(np.concatenate([pose_world,lh_norm,rh_norm]))
                    # Append class name
                    holistic_row.insert(0,class_name)
                    
                    image = cv2.flip(image,1)
                except:
                        pass # pass if landmarks detected

                if frame_num == 0: 
                    cv2.putText(image, 'PREPARE COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Class: {}, Video: {}'.format(class_name, sequence), (30,50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    # wait 5 seconds before capturing
                    cv2.waitKey(wait_time)

                else: 
                    cv2.putText(image, 'COLLECT', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Class: {}, Video: {}'.format(class_name, sequence), (30,50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                cv2.imwrite(os.path.join(path,'{}'.format(class_name) + '-{}'.format(user) + '-{}'.format(i)  + '.png'),image)
                with open('landmark_data.csv', mode = 'a', newline = '') as file:
                    write_csv = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                    write_csv.writerow(holistic_row)

                frame+=1
                print('frame',frame)

                print('sequence number',sequence)        

                # break loop and close windows if q key is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    return 

#class_name = ['Punch','Kick','Float','Flick','Neutral']
#export_to_csv(1,'Ben_2','Neutral',15,30,2000)

def train_model():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline # build ML pipeline
    from sklearn.preprocessing import StandardScaler # Normalise data
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsRegressor

    df = pd.read_csv('landmark_data.csv') 
    df = df.fillna(0)
    # Remove class oclumn
    x = df.drop('Class',axis=1) #features
    y = df['Class'] # target
    # set random_state for initial testing for reproducibility
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,stratify=y)
    pipelines = {
        'lr':make_pipeline(StandardScaler(), LogisticRegression()),
        'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        #'kn':make_pipeline(StandardScaler(), KNeighborsRegressor())
    }
    pipelines.keys()

    fit_models = {}
    for algorithm, pipeline in pipelines.items():
        model = pipeline.fit(x_train, y_train)
        fit_models[algorithm] = model
    print(fit_models)
    #print(fit_models['rf'].predict(x_test))
    return x_train, x_test, y_train, y_test, fit_models

#train_model()

def evaluate_model():
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import pickle # Save model 

    x_train, x_test, y_train, y_test, fit_models = train_model()
    
    # loop through each model to make prediction 
    for algorithm, model in fit_models.items():
        # store predictions
        y_hat = model.predict(x_test)
        print(algorithm, accuracy_score(y_test,y_hat))

    y_pred = fit_models['rf'].predict(x_test)
    # Create classification report 
    report = classification_report(y_test,y_pred, output_dict=True)
    # Save as dataframe and print latex code
    df = pd.DataFrame(report).transpose()
    print(df.to_latex(index=True))

    cf_matrix = confusion_matrix(y_test,y_pred)
    print(cf_matrix)

    # Write binary pkl file and export logistic regression model
    with open('pose_detection_model.pkl','wb') as file:
        pickle.dump(fit_models['rf'],file) 

#evaluate_model()

def send_landmark_class():
    import pickle
    
    # read model 
    with open('pose_detection_model.pkl','rb') as file:
        model = pickle.load(file)
    
    # Get webcam input
    cap = cv2.VideoCapture(1) # 'VideosSelf/Arms-Raise-Front.mov' 'Videos/no-leg-ball.mp4'
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #800, 1024, 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #600, 780, 1080

    # Begin new instance of mediapipe feed
    with mp_holistic.Holistic(
        model_complexity = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as pose:

        frame = 0
        height = 0
        height_sum = []

        while cap.isOpened():

            isTrue, image = cap.read() 

            if not isTrue:
                print("Empty camera frame")
                continue 

            # Improve performance 
            image.flags.writeable = False
            # Recolour image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Make detection and store in output array
            output = pose.process(image)
            image.flags.writeable = True
            # Recolour to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks on image
            mp_draw.draw_landmarks(
                image,
                output.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks
            
            mp_draw.draw_landmarks(
                image,
                output.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks

            mp_draw.draw_landmarks(
                image,
                output.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks

            # Extract available landmarks
            try:
                # extract normalised coordinates
                pose_norm = output.pose_landmarks.landmark
                lh_norm = output.left_hand_landmarks.landmark
                rh_norm = output.right_hand_landmarks.landmark

                pose_world = output.pose_world_landmarks.landmark

                # normalised row
                pose_row = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_norm]).flatten() if output.pose_landmarks else np.zeros(132)
                lh_row = np.array([[res.x, res.y, res.z] for res in lh_norm]).flatten() if output.left_hand_landmarks else np.zeros(21*3)
                rh_row = np.array([[res.x, res.y, res.z] for res in rh_norm]).flatten() if output.right_hand_landmarks else np.zeros(21*3)
                
                # extract world coordinates
                pose_world_row = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_world]).flatten() if output.pose_world_landmarks else np.zeros(132)

                # create list of landmark coordinates
                holistic_row = list(np.concatenate([pose_world_row,lh_row,rh_row]))
                coords = pd.DataFrame([holistic_row])
                
                # Predict classes
                gesture_class = model.predict(coords)[0]
                # Extract classification probabilities 
                gesture_prob = model.predict_proba(coords)[0]
                #print(gesture_class,gesture_prob)

                # Calibrate body proportions for new detection

                # Determine relative height of person 
                # Extract right eye y coordinate
                if 50 < frame < 100:
                    #print(frame)

                    height = Landmark(pose_world).distance('LEFT_EYE','RIGHT_HEEL','y')

                    height_sum.append(height)
                    print('height',height)
                    #print('range',range)

                # Send calibrated height 
                if frame == 100:
                    new_height = sum(height_sum)/len(height_sum)
                    print('av height:', new_height)
                    SendOSC().data('User height',new_height,500)
            

                # Extract data for specific landmarks:
                # w.r.t image
                left_wrist = Landmark(pose_norm).data('LEFT_WRIST')
                right_wrist = Landmark(pose_norm).data('RIGHT_WRIST')
                av_position = Landmark(pose_norm).average()
                #print(av_position)
                
                # w.r.t world
                # Average hegiht of hands
                lh = Landmark(pose_world).data('LEFT_WRIST')
                rh = Landmark(pose_world).data('RIGHT_WRIST')
                av_hand_width = (lh[2] + rh[2])/2
                av_hand_height = (lh[3] + rh[3])/2
                av_hand_depth = (lh[4] + rh[4])/2

                # Calculate arm angle
                x_a = Landmark(pose_world).distance('LEFT_SHOULDER','LEFT_WRIST','x')
                x_b = Landmark(pose_world).distance('RIGHT_SHOULDER','RIGHT_WRIST','x')
                y_a = Landmark(pose_world).distance('LEFT_SHOULDER','LEFT_WRIST','y')
                y_b = Landmark(pose_world).distance('RIGHT_SHOULDER','RIGHT_WRIST','y')

                alpha = (180/pi)*atan2(x_a,y_a)
                beta = (180/pi)*atan2(x_b,y_b)
                angles = [alpha,beta]

                # Calculate height
                height_world = Landmark(pose_world).distance('LEFT_EYE','LEFT_ANKLE','y')


                # Send OSC landmark data 
                SendOSC().landmark_data(left_wrist,9900)
                SendOSC().landmark_data(right_wrist,8900)
                SendOSC().landmark_data(lh,9500)
                SendOSC().landmark_data(rh,8500)
                SendOSC().data('Height',height_world,7000)
                SendOSC().data('Average_position', av_position, 6000)
                SendOSC().data('Hand height',av_hand_height,4000)
                SendOSC().data('Hand depth',av_hand_depth,3000)
                SendOSC().data('right_wrist_world',av_hand_width,2000)   
                SendOSC().data('arm_angles',angles,1000)   
                SendOSC().data('l_arm_height',y_a,1100)   
                SendOSC().data('r_arm_height',y_b,1200)   

                # Send OSC classification data
                SendOSC().data(gesture_class,gesture_prob,5000)   
   
            except:
                pass # pass if landmarks detected
            
            frame += 1             

            # Display 
            # Mirror image for webcam display
            image = cv2.flip(image,1)

            try:
                # Display class prediction
                cv2.putText(image, gesture_class, (540, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)
                # Display class probability
                cv2.putText(image, str(round(gesture_prob[np.argmax(gesture_prob)],2)), (680, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)
            except:
                pass
            
            # Visualise image
            cv2.imshow('Video', image)

            # break loop and close windows if q key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows
send_landmark_class()


# Neutral-Flick, Neutral-Float, Neutral-Punch, Neutral-Kick
# Scenario 2: Punch-Flick, Neutral-Float 

#cap.release()

#cv2.destroyAllWindows
