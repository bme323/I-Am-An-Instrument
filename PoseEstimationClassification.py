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
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils



def PoseEstimation(input,tracking_val,model_complexity,num_frames):
    # Get webcam input
    cap = cv2.VideoCapture(input) # 'Videos/no-leg-ball.mp4'
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
    with mp_pose.Pose(
        model_complexity = model_complexity,
        min_detection_confidence = 0.5,
        min_tracking_confidence = tracking_val) as pose:

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
            h,w,c = image.shape
            
            # Draw landmarks on image
            mp_draw.draw_landmarks(
                image, 
                output.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2),
                mp_draw.DrawingSpec(color=(0,255,255),thickness=2,circle_radius=2)) # connect landmarks

            # Extract available landmarks
            # for lm in mp_pose.PoseLandmark: print(lm)
            try:
                landmarks = output.pose_world_landmarks.landmark
                
                # Average landmark coordinates
                lm_x = []
                lm_y = []
                lm_z = []
                # Average landmark visibility
                lm_v = []
                for lm in range(len(landmarks)):
                    lm_x.append(landmarks[lm].x)
                    lm_y.append(landmarks[lm].y)
                    lm_z.append(landmarks[lm].z)
                    lm_v.append(landmarks[lm].visibility)
                
                    
                av_lm_x = sum(lm_x)/len(landmarks)
                av_lm_y = sum(lm_y)/len(landmarks)
                av_lm_z = sum(lm_z)/len(landmarks)
                av_lm_v = sum(lm_v)/len(landmarks)

                av_lm = [av_lm_x, av_lm_y, av_lm_z, av_lm_v]

                # Create list of average landmark data
                av_lm_data = ['0','Average_lm',av_lm_x,av_lm_y,av_lm_z,av_lm_v]
                #  Send averaage OSC lm data
                SendOSC().landmark_data(av_lm_data,6000)

                # Create list of average visibility for plotting
                av_visibility.append(av_lm_v)

                # Get landmark data for specific landmarks
                left_wrist = Landmark(landmarks).data('LEFT_WRIST')
                right_wrist = Landmark(landmarks).data('RIGHT_WRIST')
                left_hip = Landmark(landmarks).data('LEFT_HIP')
                right_hip = Landmark(landmarks).data('RIGHT_HIP')

                landmark_list.append(left_wrist[5])

                # Send OSC data 
                #SendOSC(left_wrist).port(9900)

                SendOSC().landmark_data(left_wrist,9900)
                SendOSC().landmark_data(right_wrist,8900)
                SendOSC().landmark_data(left_hip,9100)
                SendOSC().landmark_data(right_hip,8100)



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

            frame+=1
            print(frame)

            cv2.putText(image, show_fps, (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            # Visualise image and flip display
            cv2.imshow('Video', image)

            # break loop and close windows if q key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if input == 1:
                pass
            elif frame == num_frames:
                break
    frame = 0
    return output,fps_array,av_visibility



    #np.savetxt('right_heel_05.csv',right_heel_list)
    """
    plt.figure(1)
    plt.plot(av_visibility,linewidth = 0.5, label = i)
    # plt.plot(right_heel_list[0:150],linewidth = 0.5, color = 'black')


    plt.title('Average Landmark Visibility')
    plt.xlabel('Frame')
    plt.ylabel('Visibility')
    plt.legend()
    plt.show()
    """
#landmarks,_ = PoseEstimation(1,0.5,1,100)

def create_csv():
    # Create headings for CSV file
    headings = ['Class']
    for val in range(1,34):
        headings += ['X{}'.format(val),'Y{}'.format(val),'Z{}'.format(val),'Visibility {}'.format(val)]

    # Write csv file
    with open('landmark_data.csv', mode = 'w', newline = '') as file:
        write_csv = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        write_csv.writerow(headings)
    return 
#create_csv()

def export_to_csv(input,user,class_name,frames):
    # Get webcam input
    cap = cv2.VideoCapture(input) # 'Videos/no-leg-ball.mp4'
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #800, 1024, 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #600, 780, 1080

    path = 'Class-Images'

    frame = 0
    
    # Begin new instance of mediapipe feed
    with mp_pose.Pose(
        model_complexity = 1,
        min_detection_confidence = 0.8,
        min_tracking_confidence = 0.8) as pose:
        toc = time.perf_counter()
        i=0
        
        while cap.isOpened():
            tic = time.perf_counter()
            t = tic - toc
            count = (int(t)%5)    
            countdown = 4-count

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
            h,w,c = image.shape
            
            # Draw landmarks on image
            mp_draw.draw_landmarks(
                image, 
                output.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255,0,255),thickness=2,circle_radius=2),
                mp_draw.DrawingSpec(color=(0,255,255),thickness=2,circle_radius=2)) # connect landmarks

            # Extract available landmarks
            # for lm in mp_pose.PoseLandmark: print(lm)
            try:
                landmarks = output.pose_world_landmarks.landmark

                # Export coordinates
                # Create list of landmark coordinates
                pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten())
                # Append class name
                pose_row.insert(0,class_name)

                # Take picture every 5 seconds and export to CSV
                if countdown == 0:
                    print('Capture')

                    i+=1
                    print(i)
                    cv2.imwrite(os.path.join(path,'{}'.format(class_name) + '-{}'.format(user) + '-{}'.format(i)  + '.png'),image)
        
                    with open('landmark_data.csv', mode = 'a', newline = '') as file:
                        write_csv = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                        write_csv.writerow(pose_row)

                    
                
            
            except:
                pass # pass if landmarks detected

            # Display 
            # Mirror image for webcam display
            image = cv2.flip(image,1)

            frame+=1

            # Draw time onto image
            
            if countdown > 0:
                show_time = "{} ".format(countdown)
                print(countdown)        

            if countdown == 0:
                show_time = "Capturing"

            frame+=1
            #print(frame)


            cv2.putText(image, str(class_name), (5, 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
            cv2.putText(image, str(show_time), (240, 320), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 2)

            
            # Visualise image and flip display
            cv2.imshow('Video', image)

            # break loop and close windows if q key is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print(t)
                break

            # Stop recording after n frames
            if frame == (2*frames):
                print("{:.3} seconds".format(t))
                break
    return 

#class_name = ['Punch','Kick','Float','Flick','Neutral']
#export_to_csv(1,'Janine','Punch',1000)

def train_model():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline # build ML pipeline
    from sklearn.preprocessing import StandardScaler # Normalise data
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsRegressor

    df = pd.read_csv('landmark_data.csv') 
    # Remove class oclumn
    x = df.drop('Class',axis=1) #features
    y = df['Class'] # target
    # set random_state for initial testing for reproducibility
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=900,stratify=y)
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
    import seaborn 

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

    ax = seaborn.heatmap(cf_matrix, annot = True, fmt = 'd', cmap = 'Blues', cbar = False)
    
    ax.set_title('Random Forest Classification Confusion Matrix')
    ax.set_xlabel('Predicted Gesture \n')
    ax.set_ylabel('\n Actual Gesture')

    ax.xaxis.set_ticklabels(['Flick','Float','Kick','Neutral','Punch'])
    ax.yaxis.set_ticklabels(['Flick','Float','Kick','Neutral','Punch'])
    plt.savefig('confusion_matrix_rf.eps'.format(input), format='eps')

    plt.show()

    FP = cf_matrix.sum(axis=0) - np.diag(cf_matrix)  
    print(FP)
    FN = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
    print(FN)
    TP = np.diag(cf_matrix)
    print(TP)
    """
    TN = confusion_matrix.values.sum() - (FP + FN + TP)
    print(TN)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    """
    # Write binary pkl file and export logistic regression model
    with open('pose_detection_model.pkl','wb') as file:
        pickle.dump(fit_models['rf'],file) 

#evaluate_model()

def make_detections():
    import pickle
    
    # read model 
    with open('pose_detection_model.pkl','rb') as file:
        model = pickle.load(file)
    
    # Get webcam input
    cap = cv2.VideoCapture(1) # 'Videos/no-leg-ball.mp4'
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #800, 1024, 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #600, 780, 1080

    #`Initialise time and fps variables 
    time_start = 0
    fps = 0
    frame = 0

    # Begin new instance of mediapipe feed
    with mp_pose.Pose(
        model_complexity = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as pose:

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
            h,w,c = image.shape
            
            # Draw landmarks on image
            mp_draw.draw_landmarks(
                image, 
                output.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2),
                mp_draw.DrawingSpec(color=(0,255,255),thickness=2,circle_radius=2)) # connect landmarks

            # Extract available landmarks
            # for lm in mp_pose.PoseLandmark: print(lm)
            try:
                landmarks = output.pose_world_landmarks.landmark
                
                # Create list of landmark coordinates
                pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten())

                # Make detections
                x = pd.DataFrame([pose_row])
                gesture_class = model.predict(x)[0]
                # Probability of that class
                gesture_prob = model.predict_proba(x)[0]
                print(gesture_class,gesture_prob)

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

            frame+=1

            cv2.putText(image, show_fps, (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Display class prediction
            cv2.putText(image, gesture_class, (540, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)
            # Display class probability
            cv2.putText(image, str(round(gesture_prob[np.argmax(gesture_prob)],2)), (680, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)
 
            
            # Visualise image
            cv2.imshow('Video', image)

            # break loop and close windows if q key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows

#make_detections()

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
    with mp_pose.Pose(
        model_complexity = 2,
        min_detection_confidence = 0.7,
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
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2),
                mp_draw.DrawingSpec(color=(0,255,255),thickness=2,circle_radius=2)) # connect landmarks

            # Extract available landmarks
            # for lm in mp_pose.PoseLandmark: print(lm)
            try:
                # Extract normalised landmarks 
                landmarks_norm = output.pose_landmarks.landmark
                # Extract landmarks in world coordinates
                landmarks_world = output.pose_world_landmarks.landmark

                # Make Pose Classifications
                # Create list of landmark world coordinates
                pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks_world]).flatten())
                coords = pd.DataFrame([pose_row])
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

                    height = Landmark(landmarks_world).distance('LEFT_EYE','RIGHT_HEEL','y')
                    range = Landmark(landmarks_world).distance('LEFT_WRIST','RIGHT_WRIST','x')

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
                left_wrist = Landmark(landmarks_norm).data('LEFT_WRIST')
                right_wrist = Landmark(landmarks_norm).data('RIGHT_WRIST')
                av_position = Landmark(landmarks_norm).average()
                
                # w.r.t world
                # Average hegiht of hands
                lh = Landmark(landmarks_world).data('LEFT_WRIST')
                rh = Landmark(landmarks_world).data('RIGHT_WRIST')
                av_hand_width = (lh[2] + rh[2])/2
                av_hand_height = (lh[3] + rh[3])/2
                av_hand_depth = (lh[4] + rh[4])/2

                # Calculate arm angle
                x_a = Landmark(landmarks_world).distance('LEFT_SHOULDER','LEFT_WRIST','x')
                x_b = Landmark(landmarks_world).distance('RIGHT_SHOULDER','RIGHT_WRIST','x')
                y_a = Landmark(landmarks_world).distance('LEFT_SHOULDER','LEFT_WRIST','y')
                y_b = Landmark(landmarks_world).distance('RIGHT_SHOULDER','RIGHT_WRIST','y')

                alpha = (180/pi)*atan2(x_a,y_a)
                beta = (180/pi)*atan2(x_b,y_b)
                angles = [alpha,beta]

                # Send OSC landmark data 
                SendOSC().landmark_data(left_wrist,9900)
                SendOSC().landmark_data(right_wrist,8900)
                SendOSC().landmark_data(lh,9500)
                SendOSC().landmark_data(rh,8500)
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

def plot_class_probabilities(input,title,num_samples):
    import pickle
    
    # Initialise class array
    class_prob = np.zeros((1,5)).flatten()


    # read model 
    with open('pose_detection_model.pkl','rb') as file:
        model = pickle.load(file)
    
    # Get webcam input
    cap = cv2.VideoCapture('VideosSelf/{}.mov'.format(input)) # 'Videos/no-leg-ball.mp4'
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800) #800, 1024, 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600) #600, 780, 1080


    # Begin new instance of mediapipe feed
    with mp_pose.Pose(
        model_complexity = 2,
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.5) as pose:

        frame = 0

        while cap.isOpened():

            isTrue, image = cap.read() 
            cap.set(cv2.CAP_PROP_FPS,2)


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
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2),
                mp_draw.DrawingSpec(color=(0,255,255),thickness=2,circle_radius=2)) # connect landmarks

            # Extract available landmarks
            # for lm in mp_pose.PoseLandmark: print(lm)
            try:
                # Extract normalised landmarks 
                landmarks_norm = output.pose_landmarks.landmark
                # Extract landmarks in world coordinates
                landmarks_world = output.pose_world_landmarks.landmark

                # Make Pose Classifications
                # Create list of landmark world coordinates
                pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks_world]).flatten())
                coords = pd.DataFrame([pose_row])
                # Predict classes
                gesture_class = model.predict(coords)[0]
                # Extract classification probabilities 
                gesture_prob = model.predict_proba(coords)[0]

                #print(gesture_class,gesture_prob)
                
                if 10 < frame < num_samples:
                    print(frame)
                    class_prob = np.vstack([class_prob,gesture_prob])
                if frame == num_samples:
                    break   

                    
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

    df = pd.DataFrame(data = class_prob,columns = ['Flick','Float','Kick','Neutral','Punch'])
    df = df.iloc[1: , :]

    #df.plot(linewidth=0.5)
    # ['Flick','Float','Kick','Neutral','Punch']
    styles = ['k:','k-.','k--','r-','k-']
    linewidths = [0.7, 0.7, 0.7,1,0.7]
    fig, ax = plt.subplots()
    for col, style, lw in zip(df.columns, styles, linewidths):
        df[col].plot(style=style, lw=lw, ax=ax)
    
    plt.title('{} Gesture Classification'.format(title), fontsize = 12)
    plt.xlabel('Sample')
    plt.ylabel('Probability')
    plt.style.use('seaborn-deep') #fivethirtyeight #seaborn-colorblind # seaborn-dark-palette, seaborn-deep
    plt.legend(bbox_to_anchor=(1.04,1),borderaxespad=0)
    plt.tight_layout()
    plt.xticks(np.arange(0, num_samples+1, 25))
    #fig.savefig('{}.eps'.format(input), format='eps')


    plt.show()

    print(df)
    
    
#plot_class_probabilities('Arms-Raise-Side','Arm-Raise x',201)

# Neutral-Flick, Neutral-Float, Neutral-Punch, Neutral-Kick
# Scenario 2: Punch-Flick, Neutral-Float 

#cap.release()

#cv2.destroyAllWindows
