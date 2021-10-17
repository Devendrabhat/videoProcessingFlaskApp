# IMPORTING REQUIRED LIBRARIES

import cv2
import os
import glob
import numpy as np
import time
import logging
import matplotlib.pyplot as plt


def processor(filename):
    # MODEL
    logging.info("Started the script.")
    yolo=cv2.dnn.readNet("yolov3_training_last.weights","yolov3_testing.cfg")
    classes=[]
    with open("classes.txt") as f:
        classes = f.read().splitlines()

    # VARIABLES

    video_file=os.getcwd()+"/"+filename
    # video_name=video_file.split("/")[-1].replace(".mp4","")
    video_name=video_file.split("Tremor")[-1][1:].replace(".mp4","")
    print(video_name)
    count_list=[]
    time_list=[]
    count=0
    frequency=0
    none_frame=0
    end=0
    sec=0.0
    framerate=1.0
    x_disp=[]
    y_disp=[]
    time_axis=[]



    # READ VIDEO FILE

    cap=cv2.VideoCapture(video_file)

    # VIDEOWRITER

    # vout=None
    # out_vid_file="/home/root1/kunal/{}_output.mp4".format(video_name)
    # # fps= cap.get(cv2.CAP_PROP_FPS)
    # # fps=int(fps/3)
    # # print(fps)


    while True:

        _,img=cap.read()

        # SKIP FRAMES
        # sec+=framerate
        # if sec%15!=0:
        #     continue


        # INCREMENT none_frame IF VIDEO GIVES NULL/NONE FRAME
        if img is None:
            none_frame+=1
        else:
            none_frame=0

        # BREAK WHILE LOOP WHEN THERE ARE CONSECUTIVE 5 NONE FRAMES (END OF VIDEO)
        if none_frame==5:
            end=1
            break

        # IF IMAGE IS NONE, CONTINUE LOOP
        if img is None:
            continue


        # SCALE (DOWN) IMAGE / RESIZE

        # scale_percent=40
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # img=cv2.resize(img,(width,height),interpolation=cv2.INTER_AREA)


        # FOR OUTPUT VIDEO
        # height,width= img.shape[:2]
        # size=(width,height)
        # if vout==None:
        #     vout = cv2.VideoWriter(out_vid_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        #     print("videowriter created")

        # GET WIDTH & HEIGHT OF IMAGE/FRAME
        height,width= img.shape[:2]


        # IMAGE PROCESSING FOR YOLO MODEL
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo.setInput(blob)
        ln = yolo.getUnconnectedOutLayersNames()
        layerOutputs = yolo.forward(ln)

        boxes=[]
        confidences=[]
        class_ids=[]

        for output in layerOutputs:
            for detection in output:
                scores=detection[5:]
                class_id=np.argmax(scores)
                confidence=scores[class_id]

                # CONSIDER DETECTED OBJECT ONLY IF CONFIDENCE VALUE IS ABOVE SET THRESHOLD
                confidence_threshold=0.5
                if confidence>confidence_threshold:
                    center_x=int(detection[0]*width)
                    center_y = int(detection[1] * height)
                    w=int( detection[2] * width )
                    h=int( detection[3] * height )
                    x=int( center_x - w /2 )
                    y=int( center_y- h / 2 )

                    # STORING BOUNDING BOX , CONFIDENCE , CLASS ID VALUES IN THEIR RESPECTIVE LIST
                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # CHECK IF ANYTHING DETECTED
        indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

        # IF ANY OBJECT IS DETECTED
        if len(indexes)>0:
            for i in indexes.flatten():
                confidence=confidences[i]
                x,y,w,h=boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]

                # IF scissor DETECTED
                if classes[class_ids[i]]=="scissor":
                    cv2.rectangle(img, (x, y), (x + 1 + w + 1, y + 3 + h + 2), (100, 100, 100), 2)
                    cv2.putText(img, classes[class_ids[i]], (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 210), 2)
                    count+=1

                    # FOR PLOT-2
                    x_disp.append(x)
                    y_disp.append(y)
                    time_axis.append(time.time())



        # FOR PLOT-1
        if int(time.time())%5==0:
            frequency+=1
            count_list.append(count)
            time_list.append(5*frequency)
            # print(frequency,time_list,count,count_list)
            count=0


        # SAVE VIDEO
        # vout.write(img)

        # DISPLAY IMAGE/VIDEO
        # cv2.imshow("image",img)
        # cv2.waitKey(1)




    # PLOTTING AND SAVING GRAPH

    # PLOT-1

    plt.plot(time_list,count_list)
    plt.xlim([time_list[0],time_list[-1]])
    plt.grid()
    plt.xlabel("Time (s)")
    plt.ylabel("Tremor Acceleration (mm/s2)")
    plt.title("\nAcceleration VS Time\n")
    plt.savefig("graph_{}.jpg".format(filename))
    # plt.show()

    # PLOT-2

    plt.subplot(2,1,1)
    plt.plot(time_axis,x_disp)
    plt.title("Plot 1 - X_Displacement & Plot 2 - Y_Displacement")
    plt.subplot(2,1,2)
    plt.plot(time_axis,y_disp)
    plt.xlabel("Time (Sec)")
    plt.ylabel("{}Position".format(" "*35))
    plt.savefig("graph_2_{}.jpg".format(filename))
    # plt.show()