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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

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



    def person_enter(self,image_path):
        image = cv2.imread(image_path)
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image,detail=0,allowlist='0123456789:/-APMapm')
        text = " ".join(results)
        time_pattern = r'\b\d{2}\s*[:.]\d{2}\s*[:.]\d{2}\b'
        time_match = re.search(time_pattern, text)
        if time_match:
            print(f"Found Time stamp on image -{time_match.group()}")
            self.entry_time = time_match.group()
            print(self.entry_time)
            return self.entry_time
        
              
    def person_with_broom_stick(self, username, password, ip, port, channel, broom_endtime, formatted_date, playback_start_time, conf_threshold=0.85):
        model = YOLO("best_nano.pt")
         # Adjust stream time until available
        print("2222222222222222222222")
        while True:
            print(f"------------{playback_start_time,broom_endtime}")
            start_time_stamp = self.datetime_to_unix_timestamp(playback_start_time)
            end_time_stamp = self.datetime_to_unix_timestamp(broom_endtime)
            print(f"------------{start_time_stamp,end_time_stamp}")
            video_path = self.unv_url(username, password, ip, port, channel, start_time_stamp, end_time_stamp)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Stream not available for {playback_start_time.strftime('%H:%M:%S')}, trying next second...")
                playback_start_time += timedelta(seconds=1)
                cap.release()
            else:
                print(f"Stream available at {playback_start_time.strftime('%H:%M:%S')}")
                break
        output_dir = f"detected_frame1/NC02/{formatted_date}"
        os.makedirs(output_dir, exist_ok=True)
       
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS: {fps}")
 
        sweeping_detected = False  # Track whether any sweeping happened
 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
 
            results = model(frame, save=False, verbose=False)
            boxes = results[0].boxes
            class_ids = boxes.cls.tolist()
            confidences = boxes.conf.tolist()
 
            found_person = False
            found_broom = False
 
            for cls_id, conf in zip(class_ids, confidences):
                if conf >= conf_threshold:
                    if int(cls_id) == 0:
                        found_broom = True
                    elif int(cls_id) == 1:
                        found_person = True
 
            annotated_frame = results[0].plot()
            cv2.imshow("annotated_window", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
        
            if found_person and found_broom:
                simulated_frame_time = playback_start_time + timedelta(seconds=frame_count / fps)
                hour = simulated_frame_time.hour
                minute = simulated_frame_time.minute
                current_minutes = hour * 60 + minute
 
                if 360 <= current_minutes < 420:  # 06:00–07:00
                    shift = "morning"
                    is_nc = False
                elif 420 <= current_minutes <= 570:  # 07:00–09:30
                    shift = "morning"
                    is_nc = True
                elif 1080 <= current_minutes < 1110:  # 18:00–18:30
                    shift = "evening"
                    is_nc = False
                elif 1110 <= current_minutes <= 1230:  # 18:30–20:30
                    shift = "evening"
                    is_nc = True
                else:
                    shift = "unknown"
                    is_nc = True
 
                sweeping_detected = True
 
                if is_nc:
                    # Save NC frame and JSON
                    raw_path = os.path.join(output_dir, f"frame_{frame_count}_raw.jpg")
                    annotated_path = os.path.join(output_dir, f"frame_{frame_count}_annotated.jpg")
                    json_path = os.path.join(output_dir, f"frame_{frame_count}_meta.json")
 
                    cv2.imwrite(raw_path, frame)
                    cv2.imwrite(annotated_path, annotated_frame)
 
                    metadata = {
                        # "frame": frame_count,
                        "location" : "kannakurukai",
                        "timestamp": simulated_frame_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "activity": "sweeping",
                        "NC": "YES",
                        "shift": shift
                    }
                    with open(json_path, "w") as jf:
                        json.dump(metadata, jf, indent=4)
 
                    print(f" NC: Saved sweeping after shift at {metadata['timestamp']} | Shift: {shift}")
                    break  # Stop loop after first NC detection
                        
 
                else:
                    # Save NC frame and JSON
                    raw_path = os.path.join(output_dir, f"frame_{frame_count}_raw.jpg")
                    annotated_path = os.path.join(output_dir, f"frame_{frame_count}_annotated.jpg")
                    json_path = os.path.join(output_dir, f"frame_{frame_count}_meta.json")
 
                    cv2.imwrite(raw_path, frame)
                    cv2.imwrite(annotated_path, annotated_frame)
                    # Just update sweeping status, no frame saving
                    json_path = os.path.join(output_dir, f"sweeping_within_shift_{shift}.json")
                    metadata = {
                        "activity": "sweeping",
                        "NC": "NO",
                        "shift": shift,
                        "timestamp": simulated_frame_time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    with open(json_path, "w") as jf:
                        json.dump(metadata, jf, indent=4)
 
                    print(f" Sweeping done within shift ({shift}) at {metadata['timestamp']}. JSON updated.")
                    break  # Stop loop after compliant sweeping
 
            frame_count += 1
 
        cap.release()
        cv2.destroyAllWindows()
 
        # If no sweeping detected at all
        if not sweeping_detected:
            shift_detected = "morning" if 360 <= playback_start_time.hour * 60 + playback_start_time.minute <= 570 else "evening"
            metadata = {
                "activity": "not sweeping",
                "shift": shift_detected,
                "NC": "YES"
            }
            json_path = os.path.join(output_dir, f"no_sweeping_{shift_detected}.json")
            with open(json_path, "w") as jf:
                json.dump(metadata, jf, indent=4)
            print(f" No sweeping detected during {shift_detected} shift. JSON saved.")
            return (f" No sweeping detected during {shift_detected} shift. JSON saved.")
  


def main(_ip):
    obj = Process()  
    ip = _ip
    file_path = "testing_date_kukudipatti.json"
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

    for i in range(20, 21):
        start_time = datetime(2025, 4, i, 5, 30, 0)  
        end_time = datetime(2025, 4, i, 7, 20, 0)  

        env_startime = datetime(2025, 4, i, 17, 0, 0)  
        env_end_time = datetime(2025, 4, i, 19, 0, 0)  

        formatted_date = start_time.strftime("%d-%m-%Y")
        print(f"***{formatted_date}***")
        
        AM_result = None
        PM_result =  None
        today = datetime.now()
        man_detect = obj.save_first_frame_with_person(username,password, ip, port, channel, start_time, end_time, formatted_date, output_folder="shift_11/AM")
        # man_detect_2 = obj.save_first_frame_with_person(username, password, ip, port, channel, env_startime, env_end_time, formatted_date, output_folder="shift_22/PM")

        print(man_detect) 
        # print(man_detect_2) 
        print("Saved successfully")  

        am_value, pm_value = "NaN", "NaN"  
  
        if man_detect != "No person detected in the video.":
            try:
                extract_date = obj.person_enter(f"shift_11/AM/{formatted_date}.jpg")
                am_value = extract_date if extract_date else "Failed to extract_1"
                am_buffer_timing = "06:20:00"
                am_buffer_str = datetime.strptime(am_buffer_timing, "%H:%M:%S")
                before_30mins_am = am_buffer_str - timedelta(minutes=30)
                time_before_30mins_am = before_30mins_am.strftime("%H:%M:%S")
                if time_before_30mins_am >= am_value:
                    print("yes")
                    AM_result = "YES"
                else:
                    print("no") 
                    AM_result = "NO"   
                print(am_value)
                print("44444444444444444")

                am_v =  formatted_date +  " " + am_value 
                am_value_datetime = datetime.strptime(am_v, "%d-%m-%Y %H:%M:%S")
                broom_starttime = obj.datetime_to_unix_timestamp(am_value_datetime)
                print(broom_starttime)

                am_end =  formatted_date +  " " + "09:30:00" 
                am_endvalue_datetime = datetime.strptime(am_end, "%d-%m-%Y %H:%M:%S")

                broom_endtime = obj.datetime_to_unix_timestamp(am_endvalue_datetime)
                print(broom_endtime)
                broom_url = obj.unv_url(username, password, ip, port, channel, broom_starttime, broom_endtime)
                print(broom_url)

                playback_time = datetime.strptime(formatted_date + " " + am_value, '%d-%m-%Y %H:%M:%S')
                print(playback_time)
                print("--------------------------")

                broom_endtime = datetime.strptime(formatted_date + " " + "09:00:00", '%d-%m-%Y %H:%M:%S')
                print(broom_endtime)

                obj.person_with_broom_stick(username, password, ip, port, channel, broom_endtime, formatted_date, playback_time)

                print("===========")

            except Exception as e:
                print(f"Error processing AM frame for {formatted_date}: {e}")
                am_value = "person_enter failed"
        man_detect_2 = obj.save_first_frame_with_person(username, password, ip, port, channel, env_startime, env_end_time, formatted_date, output_folder="shift_22/PM")
        if man_detect_2 != "No person detected in the video.":
            try:
                extract_date_2 = obj.person_enter(f"shift_22/PM/{formatted_date}.jpg")
                pm_value = extract_date_2 if extract_date_2 else "Failed to extract_2"
                pm_buffer_timing = "17:40:00"
                pm_buffer_str = datetime.strptime(pm_buffer_timing, "%H:%M:%S")
                before_30mins_pm = pm_buffer_str - timedelta(minutes=30)
                time_before_30mins_pm = before_30mins_pm.strftime("%H:%M:%S")
                if time_before_30mins_pm >= pm_value:
                    print("yes")
                    PM_result="YES"
                else:
                    print("no")  
                    PM_result ="NO"


                pm_v =  formatted_date +  " " + pm_value 
                pm_value_datetime = datetime.strptime(pm_v, "%d-%m-%Y %H:%M:%S")
                broom_startptime = obj.datetime_to_unix_timestamp(pm_value_datetime)
                print(broom_startptime)

                pm_end =  formatted_date +  " " + "09:30:00" 
                pm_endvalue_datetime = datetime.strptime(pm_end, "%d-%m-%Y %H:%M:%S")

                broom_endptime = obj.datetime_to_unix_timestamp(pm_endvalue_datetime)
                print(broom_endptime)
                broom_purl = obj.unv_url(username, password, ip, port, channel, broom_starttime, broom_endtime)
                print(broom_purl)

                playback_ptime = datetime.strptime(formatted_date + " " + pm_value, '%d-%m-%Y %H:%M:%S')
                print(playback_ptime)
                print("--------------------------")

                broom_pendtime = datetime.strptime(formatted_date + " " + "20:30:00", '%d-%m-%Y %H:%M:%S')
                print(broom_pendtime)

                obj.person_with_broom_stick(username, password, ip, port, channel, broom_pendtime, formatted_date, playback_ptime)

                print("===========")

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

        new_file_path = "all_kukudapatti_nc.json"
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
   

