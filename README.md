# installing the necessary libraries
!pip install ultralytics
!pip install supervision
!pip install pytube

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, clear_output
from tqdm.notebook import tqdm
from base64 import b64encode
from pytube import YouTube

from ultralytics import YOLO

from supervision import VideoInfo, VideoSink
from supervision import get_video_frames_generator
from supervision import Detections
from supervision import ByteTrack

def download_video(link:str) -> None:
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully")

    def show_video(path: str):
  path_compressed = path.removesuffix(".mp4") + "_compressed.mp4"
  os.system(f"ffmpeg -i {path} -vcodec libx264 {path_compressed}")

  with open(path_compressed, "rb") as video:
    mp4 = video.read()

  os.remove(path_compressed)

  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML(f"""
  <video width=800 controls>
        <source src="{data_url}" type="video/mp4">
  </video>
  """)

model_small = YOLO("yolov8n.pt")
model_medium = YOLO("yolov8s.pt")
model_large = YOLO("yolov8m.pt")
tracker = ByteTrack()

  from google.colab import files
upload = files.upload()

VIDEO_PATH = "golubi_1.mp4"

gen = get_video_frames_generator(VIDEO_PATH)

for i, frame in enumerate(gen):
  if i > 5: break
  plt.imshow(frame)
  plt.axis("off")
  plt.show()
  clear_output(True)
  
<img width="512" height="389" alt="image" src="https://github.com/user-attachments/assets/d71a7934-f2cb-480b-92c3-5935a20775ae" />

model_small = YOLO("yolov8n.pt")
gen = get_video_frames_generator(VIDEO_PATH)
frame = next(gen)

yolo_detections = model_small(frame)[0]
supervision_detections = Detections.from_ultralytics(yolo_detections)
supervision_detections
print(supervision_detections.tracker_id)

tracker = ByteTrack()
supervision_detections = tracker.update_with_detections(supervision_detections)

supervision_detections.tracker_id
from supervision import BoundingBoxAnnotator, LabelAnnotator, EllipseAnnotator

bbox_annotator = BoundingBoxAnnotator()
label_annotator = LabelAnnotator()
ellipse_annotator = EllipseAnnotator()

annotators = [bbox_annotator, label_annotator, ellipse_annotator]

fig, axes = plt.subplots(3, 1, figsize=(15, 15))

for i, annotator in enumerate(annotators):
  if isinstance(annotator, LabelAnnotator):
    annotated_frame = annotator.annotate(
      scene=frame.copy(),
      detections=supervision_detections,
      labels=list(map(lambda x: f"person: {x}", supervision_detections.tracker_id))
    )
  else:
    annotated_frame = annotator.annotate(
      scene=frame.copy(),
      detections=supervision_detections,
    )
  axes[i].axis("off")
  axes[i].imshow(annotated_frame)

fig.show()
<img width="472" height="1175" alt="image" src="https://github.com/user-attachments/assets/80a87c37-4b08-4ece-a76c-601d4dad10fe" />





