import re
import os
import cv2
import json
import time
import easyocr
import logging
import numpy as np
import pandas as pd
import urllib.parse
from ultralytics import YOLO
from datetime import datetime, timedelta
from ultralytics.utils.plotting import Annotator



ip_address = {"Kannakurukai":"103.168.199.92", "Kukudupatti":"38.188.181.29"}
model  = YOLO("yolov8n.pt")

logging.basicConfig(
    filename = "z_process_log_kuku.log", 
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    filemode = "a"
)


class Process:

    def unv_url(self, username, password, ip, port, channel, start_time, end_time):
        encoded_password = urllib.parse.quote(password)
        rtsp_url = (f"rtsp://{username}:{encoded_password}@{ip}:{port}/"f"c{channel}/b{start_time}/e{end_time}/replay/")
        logging.info(f"Connecting to recorded video at {start_time}")
        logging.info(f"URL: {rtsp_url}")
        return rtsp_url

    def datetime_to_unix_timestamp(self, dt):

        return int(dt.timestamp())


    def person_enter(self, image_path):
        # image = cv2.imread(image_path)
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image_path, detail=0, allowlist='0123456789:/-APMapm')
        text = " ".join(results)

        time_pattern = [
            r'(\d{1,2}:\d{2}:\d{2})',
            r'(\d{1,2}:\d{2}(:\d{2})?\s*[APap][Mm])',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?)',
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'
        ]
        for time_match in time_pattern:
            matches = re.search(time_match, text)
            if matches:
                logging.info(f"Found Time stamp on image -{matches.group(1)}")
                self.entry_time = matches.group(1)
                return self.entry_time
            

    def save_first_frame_with_person(self, username, password, ip, port, channel,
                                    start_time, end_time, formatted_date, conf_threshold=0.9):

        logging.info("Starting to process video frames...")

        while True:
            start_time_stamp = self.datetime_to_unix_timestamp(start_time)
            end_time_stamp = self.datetime_to_unix_timestamp(end_time)
            video_path = self.unv_url(username, password, ip, port, channel, start_time_stamp, end_time_stamp)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logging.warning(f"Stream not available for {start_time.strftime('%H:%M:%S')}, trying next second...")
                start_time += timedelta(seconds=1)
                cap.release()
            else:
                logging.info(f"Stream available at {start_time.strftime('%H:%M:%S')}")
                break

        frame_count = 0
        saved_frame = None
        fail_count = 0
        max_retries = 5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                logging.warning(f"Failed to read frame ({fail_count}/{max_retries})")
                if fail_count >= max_retries:
                    logging.error("Stream failed repeatedly, exiting...")
                    break
                time.sleep(2)
                continue

            fail_count = 0
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
                        saved_frame = full_frame
                        break
                if saved_frame is not None:
                    break
            if saved_frame is not None:
                break

        cap.release()

        if saved_frame is None:
            logging.error("No person detected in the video.")
            return None 
        else:
            logging.info(f"Processed {frame_count} frames before detection.")
            timing = self.person_enter(saved_frame)
            return {"frame": saved_frame, "formated_date": formatted_date,"timing":timing, "Check_List":"First_Entry"}




def main(_ip):
    obj = Process()  
    ip = _ip
    file_path = "z_kuku_testing_date_1.json"
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

    for i in range(22, 23):
        start_time = datetime(2025, 4, i, 6, 12, 30)
        end_time = datetime(2025, 4, i, 7, 20, 0)

        env_startime = datetime(2025, 4, i, 17, 46, 15)
        env_end_time = datetime(2025, 4, i, 19, 0, 0)

        formatted_date = start_time.strftime("%d-%m-%Y")
        logging.info(f"Processing date: {formatted_date}")
        
        AM_result = None
        PM_result = None
        today = datetime.now()

        man_detect = obj.save_first_frame_with_person(username, password, ip, port, channel, start_time, end_time, formatted_date)
        man_detect_2 = obj.save_first_frame_with_person(username, password, ip, port, channel, env_startime, env_end_time, formatted_date)

        logging.info(man_detect) 
        logging.info(man_detect_2)



        # if man_detect != "No person detected in the video.":
        #     try:
        #         extract_date = obj.person_enter(f"kuku_AM/{formatted_date}.jpg")
        #         am_value = extract_date if extract_date else "Failed to extract_1"
        #         am_buffer_timing = "07:00:00"
        #         am_buffer_str = datetime.strptime(am_buffer_timing, "%H:%M:%S")
        #         before_30mins_am = am_buffer_str - timedelta(minutes=30)
        #         time_before_30mins_am = before_30mins_am.strftime("%H:%M:%S")
                
        #         if time_before_30mins_am >= am_value:
        #             AM_result = "YES"
        #         else:
        #             AM_result = "NO"   
        #     except Exception as e:
        #         logging.error(f"Error processing AM frame for {formatted_date}: {e}")
        #         am_value = "person_enter failed"

        # if man_detect_2 != "No person detected in the video.":
        #     try:
        #         extract_date_2 = obj.person_enter(f"kuku_PM/{formatted_date}.jpg")
        #         pm_value = extract_date_2 if extract_date_2 else "Failed to extract_2"
        #         pm_buffer_timing = "18:30:00"
        #         pm_buffer_str = datetime.strptime(pm_buffer_timing, "%H:%M:%S")
        #         before_30mins_pm = pm_buffer_str - timedelta(minutes=30)
        #         time_before_30mins_pm = before_30mins_pm.strftime("%H:%M:%S")
        #         if time_before_30mins_pm >= pm_value:
        #             PM_result = "YES"
        #         else:
        #             PM_result = "NO"
        #     except Exception as e:
        #         logging.error(f"Error processing PM frame for {formatted_date}: {e}")
        #         pm_value = "person_enter failed"

        # new_data = {"Location":key,
        #             "Date": formatted_date,
        #             "AM":am_value,
        #             "PM":pm_value,
        #             "createdAt":today.strftime("%Y-%m-%d %H:%M:%S")}        
        
#         data_list.append(new_data) 

#         new_format = {
#                 "Location": key,
#                 "Date": formatted_date,
#                 "timing": {
#                     "AM": am_value,
#                     "PM": pm_value
#                 },
#                 "incharge_enry": {
#                     "AM": AM_result.lower() if AM_result else "no",
#                     "PM": PM_result.lower() if PM_result else "no"
#                 },
#                 "createdAt": today.strftime("%Y-%m-%d %H:%M:%S")
#             }
#         with open(file_path, "w") as f:
#             json.dump(data_list, f, indent=4)        

#         new_file_path = "all_nc_kukudupatti.json"
#         if not os.path.exists(new_file_path):
#             with open(new_file_path, "w") as f:
#                 json.dump([new_format], f, indent=4)
#         else:
#             with open(new_file_path, "r") as f:
#                 existing_data = json.load(f)
#                 existing_data.append(new_format)
#             with open(new_file_path, "w") as f:
#                 json.dump(existing_data, f, indent=4)

#     return("AM===========>",AM_result,"PM========>",PM_result)  

# logging.info("====Process done====")


if __name__ == "__main__":
    obj = Process() 
    username = "admin"
    password = "rmc@12345"  
    port = "554"
    channel = 1  
    main(ip_address["Kukudupatti"])
    #py_to_mongo()
    print("====== 0Nc_01 completed ========")
   

