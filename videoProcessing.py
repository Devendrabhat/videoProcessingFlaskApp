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
    yolo = cv2.dnn.readNet("yolov3_training_last.weights",
                           "yolov3_testing.cfg")
    classes = []
    with open("classes.txt") as f:
        classes = f.read().splitlines()

    # VARIABLES
    video_file=os.getcwd()+"/"+filename
    # video_file = "v2_trim.mp4"
    # video_name=video_file.split("/")[-1].replace(".mp4","")
    # video_name=video_file.split("Tremor")[-1][1:].replace(".mp4","")
    video_name = "YYYYYYYY"
    print(video_name)
    count_list = []
    time_list = []
    count = 0
    frequency = 0
    none_frame = 0
    end = 0
    sec = 0.0
    framerate = 1.0
    x_disp = []
    y_disp = []
    time_axis = []

    # READ VIDEO FILE

    cap = cv2.VideoCapture(video_file)

    # VIDEOWRITER

    # vout=None
    # out_vid_file="/home/root1/kunal/{}_output.mp4".format(video_name)
    # # fps= cap.get(cv2.CAP_PROP_FPS)
    # # fps=int(fps/3)
    # # print(fps)

    Counter = 0
    scissors_count = 0
    x_axis_disp = []
    y_axis_disp = []
    scissors_list = []
    frequency = []
    frequency_new = []
    velocity_x = []
    velocity_y = []
    velocity = []
    velocity_new = []

    while True:
        Counter += 1
        print(Counter)

        _, img = cap.read()

        # SKIP FRAMES
        # sec+=framerate
        # if sec%15!=0:
        #     continue

        # INCREMENT none_frame IF VIDEO GIVES NULL/NONE FRAME
        if img is None:
            none_frame += 1
        else:
            none_frame = 0

        # BREAK WHILE LOOP WHEN THERE ARE CONSECUTIVE 5 NONE FRAMES (END OF VIDEO)
        if none_frame == 5:
            end = 1
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
        height, width = img.shape[:2]

        # IMAGE PROCESSING FOR YOLO MODEL
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo.setInput(blob)
        ln = yolo.getUnconnectedOutLayersNames()
        layerOutputs = yolo.forward(ln)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # CONSIDER DETECTED OBJECT ONLY IF CONFIDENCE VALUE IS ABOVE SET THRESHOLD
                confidence_threshold = 0.5
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # STORING BOUNDING BOX , CONFIDENCE , CLASS ID VALUES IN THEIR RESPECTIVE LIST
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # CHECK IF ANYTHING DETECTED
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # IF ANY OBJECT IS DETECTED
        if len(indexes) > 0:
            for i in indexes.flatten():
                confidence = confidences[i]
                x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

                # IF scissor DETECTED
                if classes[class_ids[i]] == "scissor":
                    cv2.rectangle(img, (x, y), (x + 1 + w + 1, y + 3 + h + 2), (100, 100, 100), 2)
                    cv2.putText(img, classes[class_ids[i]], (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 210), 2)
                    # count+=1
                    scissors_count += 1

                    # FOR PLOT-2

                    x_disp.append(x)
                    y_disp.append(y)
                    # time_axis.append(time.time())

        # # FOR PLOT-1
        # if int(time.time())%5==0:
        #     frequency+=1
        #     count_list.append(count)
        #     time_list.append(5*frequency)
        #     # print(frequency,time_list,count,count_list)
        #     count=0
        if Counter % 30 == 0:

            count += 1

            # GET THE FREQUENCY VALUE
            # Scissors count per second is the frequency value
            frequency.append(scissors_count)
            # scissors_list.append(scissors_count)
            print("Scissor-{}".format(scissors_count))

            # GET THE VELOCITY/ACCELERATION - FOR BOTH AXIS COMBINED (unit is : meter/sec : here we will consider every one coordinate movement of scissor as 1 meter)
            a = 0
            for ind, c in enumerate(zip(x_disp, y_disp)):
                if ind + 1 == len(x_disp):
                    break
                d = np.sqrt((x_disp[ind + 1] - x_disp[ind]) ** 2 + (y_disp[ind + 1] - y_disp[ind]) ** 2)
                a = a + d
            velocity.append(a)

            # IF THE CURRENT SEC FREQUENCY IS BETWEEN 4 to 9 THEN ONLY CONSIDER CALCULATED VELOCITY VALUE OTHERWISE LET's CONSIDER 50(some random value)
            # if frequency[-1] in range(4,10):
            #     acceleration.append(v)
            # else:
            #     acceleration.append(50)

            print(frequency)
            print(velocity)
            print(len(frequency), len(velocity))

            # VELOCITY ONLY FOR Y-AXIS
            # vy=0
            # for ind,i in enumerate(y_disp):
            #     if ind+1==len(y_disp):
            #         break
            #     vy=vy+abs(y_disp[ind+1]-y_disp[ind])
            # velocity_y.append(vy)
            # print(len(time_axis),len(x_axis_disp),len(y_axis_disp),len(scissors_list))

            # GETTING AVG POSITION VALUE OF SCISSOR ON X & Y AXIS
            # x_axis_disp.append(sum(x_disp) / 5)
            # y_axis_disp.append(sum(y_disp) / 5)
            # FOR X-AXIS
            if (len(x_disp) > 0):
                x_axis_disp.append(sum(x_disp) / len(x_disp))
            else:
                x_axis_disp.append(0)
            # FOR Y-AXIS
            if (len(y_disp) > 0):
                y_axis_disp.append(sum(y_disp) / len(y_disp))
            else:
                y_axis_disp.append(0)

            # GET SECOND OF VIDEO (as per fps=30)
            time_axis.append(count)

            # print("disp-v:{}-{}".format(x_axis_disp[-1],vx))
            # print("disp-v:{}-{}".format(y_axis_disp[-1],vy))
            # if count==5:
            #     break

            # RESET VARIABLES
            x_disp.clear()
            y_disp.clear()
            scissors_count = 0
            # comment above line if we need cummulative summation of scissors count

        # SAVE VIDEO
        # vout.write(img)

        # DISPLAY IMAGE/VIDEO
        # cv2.imshow("image",img)
        # cv2.waitKey(1)

    # GET ONLY THOSE ACCELERATION VALUE WHERE THE FREQUENCY IS BETWEEN 4 TO 9
    print(frequency)
    print(velocity)
    print(len(frequency), len(velocity))
    print("*************")
    for ind in range(len(frequency)):
        if frequency[ind] in range(4, 10):
            frequency_new.append(frequency[ind])
            velocity_new.append(velocity[ind])
    print(frequency_new)
    print(velocity_new)
    print(len(frequency_new), len(velocity_new))
    print("*************")

    frequency_sorted = []
    velocity_sorted = []
    indices = np.argsort(frequency_new)
    for ind in indices:
        frequency_sorted.append(frequency_new[ind])
        temp_v = velocity_new[ind] / 100.0
        velocity_sorted.append(temp_v)
    print(frequency_sorted)
    print(velocity_sorted)
    print(len(frequency_sorted), len(velocity_sorted))
    print("*************")

    # PLOTTING AND SAVING GRAPH

    # time_axis=[i for i in range(1,100)]
    # scissors_list=[i for i in range(1,100)]
    # x_axis_disp=[i for i in range(1,100)]
    # y_axis_disp=[i for i in range(1,100)]
    # print(time_axis,x_axis_disp,y_axis_disp)

    # PLOT-0

    # plt.subplot(2,1,1)
    # plt.plot(time_axis,velocity_x)
    # plt.grid()
    # plt.subplot(2,1,2)
    plt.plot(frequency_sorted, velocity_sorted)
    plt.title("\nVelocity vs Frequency\n")
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    # plt.ylabel("{} Y-axis {} Tremor Velocity {} X-axis".format(" "*50," "*5," "*5))
    plt.ylabel("Tremor Velocity (m/s)")
    plt.savefig("graph_final0_{}.jpg".format(filename))
    # plt.show()
    plt.clf()
    print("Fig1 saved")

    # PLOT-1
    plt.subplot(1, 1, 1)
    plt.plot(time_axis, frequency)
    plt.title("\nAcceleration vs Time\n")
    plt.grid()
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration(mm/s2)")
    plt.savefig("graph_final1_{}.jpg".format(filename))
    # plt.show()
    plt.clf()
    print("Fig2 saved")

    # PLOT-2
    # plt.subplot(2,1,1)
    # plt.plot(time_axis,x_axis_disp)
    # plt.title("Avg position on axis vs Time")
    # plt.subplot(2,1,2)
    # plt.plot(time_axis,y_axis_disp)
    # plt.xlabel("Time (s)")
    # plt.ylabel("{} Y-axis {} Position on axis {} X-axis".format(" "*50," "*5," "*5))
    # plt.savefig("/home/myelin/Kunal/yolo/graph_2_{}.jpg".format(filename))
    # # plt.show()
    # plt.clf()
    # print("Fig2 saved")


# processor("XYZ")

#############################################################################

# import numpy as np

# X=[4,6,25,29,30,5,7,6]
# Y=[100,200,35,70,50,10,70,90]

# # Z = [x for _,x in sorted(zip(Y,X))]
# Z=np.argsort(X)