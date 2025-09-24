from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import urllib.parse
import time
import pandas as pd
import os
import numpy as np
import easyocr
import re
from datetime import datetime, timedelta
import json
import pymongo







model = YOLO("yolov8x.pt")
ip_address = {"Kannakurukai":"103.168.199.92", "kukudupatti":"38.188.181.29"}


class Process:

    def unv_url(self,username, password, ip, port, channel,start_time, end_time):
        encoded_password = urllib.parse.quote(password)

        rtsp_url = (f"rtsp://{username}:{encoded_password}@{ip}:{port}/"
                    f"c{channel}/b{start_time}/e{end_time}/replay/")
        
        print(f"Connecting to recorded video at {start_time}")
        print(f"URL: {rtsp_url}")
        return rtsp_url

    def datetime_to_unix_timestamp(self,dt):
        """
        Convert a datetime object to a Unix timestamp (seconds since Jan 1, 1970 UTC).
        
        Parameters:
        dt (datetime): The datetime object to convert
        
        Returns:
        int: Unix timestamp (seconds since epoch)
        """
        return int(dt.timestamp())
    
    def save_first_frame_with_person(self, username, password, ip, port, channel,
                                 start_time, end_time, formatted_date, output_folder, conf_threshold=0.9):
        print("#####################")
        os.makedirs(output_folder, exist_ok=True)
        model = YOLO("yolov8n.pt")

        # Adjust stream time until available
        while True:
            start_time_stamp = self.datetime_to_unix_timestamp(start_time)
            end_time_stamp = self.datetime_to_unix_timestamp(end_time)

            video_path = self.unv_url(username, password, ip, port, channel, start_time_stamp, end_time_stamp)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Stream not available for {start_time.strftime('%H:%M:%S')}, trying next second...")
                start_time += timedelta(seconds=1)
                cap.release()
            else:
                print(f"Stream available at {start_time.strftime('%H:%M:%S')}")
                break

        frame_count = 0
        saved_frame = False
        fail_count = 0
        max_retries = 5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                print(f"Failed to read frame ({fail_count}/{max_retries})")
                if fail_count >= max_retries:
                    print("Stream failed repeatedly, exiting...")
                    break
                time.sleep(2)  # wait a bit before retry
                continue

            fail_count = 0  # reset on successful read
            frame_count += 1
            results = model(frame, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)

                    if cls == 0 and conf > conf_threshold:
                        full_frame = frame.copy()
                        cv2.rectangle(full_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                        cv2.putText(full_frame, f"Person {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        full_frame_path = os.path.join(output_folder, f"{formatted_date}.jpg")
                        cv2.imwrite(full_frame_path, full_frame)

                        saved_frame = True
                        break

                if saved_frame:
                    break

            if saved_frame:
                break

        cap.release()

        if not saved_frame:
            print("No person detected in the video.")
            return "No person detected in the video."
        else:
            return f"Processed {frame_count} frames before detection."

 

    # def save_first_frame_with_person(self,video_path,formatted_date,output_folder, conf_threshold=0.9):
    #     print("#####################")
    #     os.makedirs(output_folder, exist_ok=True)
    #     model = YOLO("yolov8n.pt")


    #     cap = cv2.VideoCapture(video_path)
    #     frame_count = 0
    #     saved_frame = False

    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
                
    #         frame_count += 1

    #         results = model(frame, verbose=False)
    #         for result in results:
    #             boxes = result.boxes
    #             for box in boxes:
    #                 cls = int(box.cls.item())
    #                 conf = box.conf.item()
    #                 xyxy = box.xyxy[0].cpu().numpy().astype(int)  # Get bounding box coordinates

    #                 if cls == 0 and conf > conf_threshold:  # Check if it's a person
    #                     full_frame = frame.copy()
    #                     cv2.rectangle(full_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)  # Draw box
    #                     cv2.putText(full_frame, f"Person {conf:.2f}", (xyxy[0], xyxy[1] - 10),
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add confidence score

    #                     full_frame_path = os.path.join(output_folder, f"{formatted_date}.jpg")
    #                     cv2.imwrite(full_frame_path, full_frame)

    #                     saved_frame = True
    #                     break  

    #             if saved_frame:
    #                 break

    #         if saved_frame:
    #             break

    #     cap.release()
   

    #     if not saved_frame:
    #         print("No person detected in the video.")
    #         return ("No person detected in the video.")
    #     else:
    #         return (f"Processed {frame_count} frames before detection.")


    def person_enter(self,image_path):
        image = cv2.imread(image_path)
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image,detail=0,allowlist='0123456789:/-APMapm')
        text = " ".join(results)
        # time_pattern = r'\b\d{2}\s*[:.]\d{2}\s*[:.]\d{2}\b'
        time_pattern = [
            r'(\d{1,2}:\d{2}:\d{2})', # HH:MM:SS
            r'(\d{1,2}:\d{2}(:\d{2})?\s*[APap][Mm])', # HH:MM:SS AM/PM or HH:MM AM/PM
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?)', # Date + time
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})' # ISO format date + time
        ]
        for time_match in time_pattern:
            matches = re.search(time_match, text)
            if matches:
                print(f"Found Time stamp on image -{matches.group(1)}")
                self.entry_time= matches.group(1)
                print(self.entry_time)
                return self.entry_time
              
            

    # def alter_time (self,start_time,end_time,ip):
    #     obj = Process() 
        
    #     while True:
    #         start_time_stamp = obj.datetime_to_unix_timestamp(start_time)
    #         end_time_stamp = obj.datetime_to_unix_timestamp(end_time)

    #         print(f"Trying start time: {start_time.strftime('%H:%M:%S')}")
    #         url = obj.unv_url(username, password, ip, port, channel, start_time_stamp, end_time_stamp)

    #         cap = cv2.VideoCapture(url)
    #         if not cap.isOpened():
    #             ret, frame = cap.read()
    #             if not ret:    
    #                 print(f"Stream not available for {start_time.strftime('%H:%M:%S')}, trying next second...")
    #                 start_time += timedelta(seconds=1)            

    #         else:
    #             print(f"Stream found at {start_time.strftime('%H:%M:%S')}")
    #             cap.release() 
    #             break
    #         cap.release()  
    #     return url    
  


def main(_ip):
    obj = Process()  
    ip = _ip
    file_path = "demo.json"
    key = next(k for k, v in ip_address.items() if v == ip)

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump([], f)

    try:
        with open(file_path, "r") as f:
            data_list = json.load(f)
            if not isinstance(data_list, list):
                data_list = []  
    except json.JSONDecodeError:
        data_list = []

    for i in range(30, 31):
        start_time = datetime(2025, 4, i, 6, 8, 10)  
        end_time = datetime(2025, 4, i, 7, 20, 0)  

        env_startime = datetime(2025, 4, i, 17, 44, 0)  
        env_end_time = datetime(2025, 4, i, 19, 0, 0)  

        formatted_date = start_time.strftime("%d-%m-%Y")
        print(f"***{formatted_date}***")
        
        AM_result = None
        PM_result =  None
        # shift_1 = obj.alter_time(start_time, end_time, ip)
        # shift_2 = obj.alter_time(env_startime, env_end_time, ip)
        today = datetime.now()
        # man_detect = obj.save_first_frame_with_person(shift_1, formatted_date, output_folder="shift_1/AM")
        # man_detect_2 = obj.save_first_frame_with_person(shift_2, formatted_date, output_folder="shift_2/PM")
        man_detect = obj.save_first_frame_with_person(username,password, ip, port, channel, start_time, end_time, formatted_date, output_folder="shift_01/AM")
        man_detect_2 = obj.save_first_frame_with_person(username, password, ip, port, channel, env_startime, env_end_time, formatted_date, output_folder="shift_02/PM")

        print(man_detect) 
        print(man_detect_2) 
        print("Saved successfully")  

        am_value, pm_value = "NaN", "NaN"  
  
        if man_detect != "No person detected in the video.":
            try:
                extract_date = obj.person_enter(f"shift_11/AM/{formatted_date}.jpg")
                am_value = extract_date if extract_date else "Failed to extract_1"
                am_buffer_timing = "07:00:00"
                am_buffer_str = datetime.strptime(am_buffer_timing, "%H:%M:%S")
                before_30mins_am = am_buffer_str - timedelta(minutes=30)
                time_before_30mins_am = before_30mins_am.strftime("%H:%M:%S")
                if time_before_30mins_am >= am_value:
                    print("yes")
                    AM_result = "YES"
                else:
                    print("no") 
                    AM_result = "NO"   
            except Exception as e:
                print(f"Error processing AM frame for {formatted_date}: {e}")
                am_value = "person_enter failed"

        if man_detect_2 != "No person detected in the video.":
            try:
                extract_date_2 = obj.person_enter(f"shift_22/PM/{formatted_date}.jpg")
                pm_value = extract_date_2 if extract_date_2 else "Failed to extract_2"
                pm_buffer_timing = "18:30:00"
                pm_buffer_str = datetime.strptime(pm_buffer_timing, "%H:%M:%S")
                before_30mins_pm = pm_buffer_str - timedelta(minutes=30)
                time_before_30mins_pm = before_30mins_pm.strftime("%H:%M:%S")
                if time_before_30mins_pm >= pm_value:
                    print("yes")
                    PM_result="YES"
                else:
                    print("no")  
                    PM_result ="NO"
            except Exception as e:
                print(f"Error processing PM frame for {formatted_date}: {e}")
                pm_value = "person_enter failed"

        new_data = {"Location":key,
                    "Date": formatted_date,
                    "AM":am_value,
                    "PM":pm_value,
                    "createdAt":today.strftime("%Y-%m-%d %H:%M:%S")}        
        
        data_list.append(new_data) 

        new_format = {
                "Location": key,
                "Date": formatted_date,
                "timing": {
                    "AM": am_value,
                    "PM": pm_value
                },
                "incharge_enry": {
                    "AM": AM_result.lower() if AM_result else "no",
                    "PM": PM_result.lower() if PM_result else "no"
                },
                "createdAt": today.strftime("%Y-%m-%d %H:%M:%S")
            }
        with open(file_path, "w") as f:
            json.dump(data_list, f, indent=4)        

        new_file_path = "demo_1.json"
        if not os.path.exists(new_file_path):
            with open(new_file_path, "w") as f:
                json.dump([new_format], f, indent=4)
        else:
            with open(new_file_path, "r") as f:
                existing_data = json.load(f)
                existing_data.append(new_format)
            with open(new_file_path, "w") as f:
                json.dump(existing_data, f, indent=4)



    return("AM===========>",AM_result,"PM========>",PM_result)     
    
print("====Process done====")



def py_to_mongo():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['va_automation']
    collection = db['nc_01']

    with open('testing_data') as file:
        file_data = json.load(file)
    collection.insert_many(file_data)
    print("data insertion completed")
    client.close()


if __name__ == "__main__":
    obj = Process() 
    username = "admin"
    password = "rmc@12345"  
    port = "554"
    channel = 1  
    main(ip_address["kukudupatti"])
    #py_to_mongo()
    print("====== 0Nc_01 completed ========")
   

