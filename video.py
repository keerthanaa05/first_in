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

model = YOLO("yolov8n.pt")
ip_address = {"Kannakurukai":"036.168.145.92", "Kukudupatti":"22.789.654.29"}

class Process:

    def unv_url(self, username, password, ip, port, channel, start_time, end_time):
        encoded_password = urllib.parse.quote(password)

        rtsp_url = (f"rtsp://{username}:{encoded_password}@{ip}:{port}/"
                    f"c{channel}/b{start_time}/e{end_time}/replay/")
        
        print(f"Connecting to recorded video at {start_time}")
        # logging.info(f"URL: {rtsp_url}")
        return rtsp_url

    def datetime_to_unix_timestamp(self, dt):
        """
        Convert a datetime object to a Unix timestamp (seconds since Jan 1, 1970 UTC).

        Parameters:
        dt (datetime): The datetime object to convert

        Returns:
        int: Unix timestamp (seconds since epoch)
        """
        return int(dt.timestamp())    
    
    def person_enter(self, image):
        # image = cv2.imread(image_path)
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image, detail=0, allowlist='0123456789:/-APMapm')
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
                print(f"Found Time stamp on image -{matches.group(1)}")
                self.entry_time = matches.group(1)
                return self.entry_time

    

    def save_first_frame_with_person(self, username, password, ip, port, channel,
                                     start_time, end_time, formatted_date, output_folder):
        model = YOLO("yolov8n.pt")

        print("Starting to download video...")
        os.makedirs(output_folder, exist_ok=True)

        start_time_stamp = self.datetime_to_unix_timestamp(start_time)
        end_time_stamp = self.datetime_to_unix_timestamp(end_time)
        video_url = self.unv_url(username, password, ip, port, channel, start_time_stamp, end_time_stamp)

        cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():
            print(f"Failed to open video stream at {start_time.strftime('%H:%M:%S')}")
            return "Failed to open video"

        formatted_time = f"{start_time.strftime('%H-%M-%S')}_to_{end_time.strftime('%H-%M-%S')}"
        output_path = os.path.join(output_folder, f"{formatted_date}_{formatted_time}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Saving video to {output_path}...")
        frame_count = 0
        person_identify = False
        first_in = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]  # Run detection on the frame

            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]
            
            conf = float(box.conf[0].item())
            if label == "person" and conf >= 0.4:
                person_identify = True
            if person_identify:
                # x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                while not first_in:
                    first_in = obj.person_enter(frame)
                print(f"first_in = {first_in}")

                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                # cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
                
                cv2.putText(frame,f"first_in identified at {first_in}",(150,100),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
            cv2.imshow("frame",frame)
            out.write(frame)
            frame_count += 1
            if cv2.waitKey(1)& 0xFF ==ord("q"):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Video saved successfully: {frame_count} frames")
        print(f"Expected duration: {(end_time - start_time).total_seconds():.2f} seconds")
        print(f"Actual saved duration: {frame_count / fps:.2f} seconds")

        return output_path






def main(_ip):
    obj = Process()  
    ip = _ip

    for i in range(30, 31):
        start_time = datetime(2025, 4, i, 5 ,59, 0)
        end_time = datetime(2025, 4, i, 6, 0, 0)

        # env_startime = datetime(2025, 4, i, 18, 1, 10)
        # env_end_time = datetime(2025, 4, i, 18, 2, 15)

        formatted_date = start_time.strftime("%d-%m-%Y")
        # print(f"Processing date: {formatted_date}")
        print(formatted_date)
        today = datetime.now()

        video_path = obj.save_first_frame_with_person(username, password, ip, port, channel,
                                                      start_time, end_time, formatted_date,
                                                      output_folder="video_kuku")
        
        # video_path1 = obj.save_first_frame_with_person(username, password, ip, port, channel,
        #                                               env_startime, env_end_time, formatted_date,
        #                                               output_folder="video_kanna")
 

print("====Process done====")


if __name__ == "__main__":
    obj = Process() 
    username = "admin"
    password = "rmc@12345"  
    port = "554"
    channel = 1  
    main(ip_address["Kukudupatti"])
    #py_to_mongo()
    print("====== 0Nc_01 completed ========")
   
