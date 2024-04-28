import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker 
import cvzone
import time

model = YOLO('yolov8s.pt')

# function to track the mouse, prints the mouse coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('v2.mp4')

# yolov8 can find objects that are mentioned in coco.txt
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") # saving the items in a list , parsing the file, save word when encounter new line
# print(class_list)

count = 0  # someting to do with frames
tracker = Tracker() # creating object 
counter = 0 # total number people that visited/went by the region/area

personup = {}  # Dictionary to keep track of persons who have already entered the square
person_times = {}  # Dictionary to keep track of entry and exit times for each person

# for constructing the square / coordinates of the area / region
cy1 = 150 
cy2 = 390
cx1 = 500
cx2 = 730

offset = 2  # used to assign the minimum proximity for the algo to detect the person

while True:
    
    """
        read the frame , if the frames is not rendered properly then break and throw exception
    """
    ret, frame = cap.read()
    if not ret:
        break
    # frame = stream.read()
    
    """

        count += 1: Increments the count variable. This is likely used to keep track of the number of frames processed.
        if count % 3 != 0: continue: Checks if count is not divisible by 3 
            (i.e., if this is not the third frame since the last processing step). 
            If true, it skips the rest of the loop and moves on to the next frame. T
            his is a way to process frames at a lower frequency than the frame rate of the video (processing every third frame in this case).
    
    """
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    # predict items mentioned in the coco.txt
    results = model.predict(frame)
    # result , percentage confidence
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float") # convert to table , store in floating format

    list = []

    for index, row in px.iterrows():
        # coordinates of the person
        x1 = int(row[0]) 
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])  # class id , what kind of item
        c = class_list[d] # class id , what kind of item
        
        # if the object is person , then add the coordinates to find the bounding box
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    # bbox = bounding box
    # new rectangle coordinates, using tracker, since it tracks the person
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        # drawing the center of object
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)  # new rectangle center point

        # for drawing bounding box for people only when the person center point touches the line
        if cx1 < cx < cx2 and cy1 < cy < cy2:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Drawing the new rectangle
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)  # Put id in top left
            # if it is a new person coming , then append it personup
            if id not in personup:
                counter += 1
                personup[id] = True
                person_times[id] = {'entry_time': time.time(), 'exit_time': None} # add the entry time for that person
        
        # when the person leaves the area/region , when the person leaves , mark his exit time       
        else:
            if id in personup:
                person_center_x = (x3 + x4) // 2
                person_center_y = (y3 + y4) // 2
                if cx1 - 10 <= person_center_x <= cx2 + 10 and cy1 - 10 <= person_center_y <= cy2 + 10:
                    person_times[id]['exit_time'] = time.time()
                    del personup[id]

   
    # draw the area / region 
    cv2.line(frame, (cx1, cy1), (cx2, cy1), (0, 255, 0), 2)  # top line
    cv2.line(frame, (cx2, cy1), (cx2, cy2), (50, 50, 50), 2)  # right line
    cv2.line(frame, (cx1, cy2), (cx2, cy2), (0, 255, 0), 2)  # bottom line
    cv2.line(frame, (cx1, cy1), (cx1, cy2), (50, 50, 50), 2)  # left line

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

print(f"Total number of persons visited the square: {counter}")

# the person who fully crossed the area/region
actual_person_data = {}
for person_id, times in person_times.items():
    entry_time = times['entry_time']
    exit_time = times['exit_time'] if times['exit_time'] is not None else 0
    # only include person who fully crossed the area
    if exit_time - entry_time < 0:
        continue
    total_time_spent = exit_time - entry_time
    actual_person_data[person_id] = [entry_time, exit_time, total_time_spent]

# Printing the dictionary
for person_id, data in actual_person_data.items():
    print(f"Person ID: {person_id}, Entry Time: {data[0]}, Exit Time: {data[1]}, Total Time Spent: {data[2]}")

# Writing to a CSV file
df = pd.DataFrame.from_dict(actual_person_data, orient='index', columns=['Entry Time', 'Exit Time', 'Total Time Spent'])
df.to_csv('person_data_1.csv', index_label='Person ID')

cap.release()
cv2.destroyAllWindows()
