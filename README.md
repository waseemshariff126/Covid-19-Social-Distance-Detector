# Covid-19-Social-Distance-Detector
Social distancing is a method used to control the spread of contagious diseases.  As the name suggests, social distancing implies that people should physically distance themselves from one another, reducing close contact, and thereby reducing the spread of a contagious disease (such as coronavirus)

We’ll be using the YOLO object detector to detect people in our video stream.

Using YOLO with OpenCV requires a bit more output processing than other object detection methods (such as Single Shot Detectors or Faster R-CNN), so in order to keep our code tidy, let’s implement a detect_people function that encapsulates any YOLO object detection logic.

# We also apply Non maxima Suppression 
- The purpose of non-maxima suppression is to suppress weak, overlapping bounding boxes.

Assuming the result of NMS yields at least one detection, we loop over them, extract bounding box coordinates, and update our results list consisting of the:
- Confidence of each person detection
- Bounding box of each person
- Centroid of each person
Finally, we return the results to the calling function.
# Download and place the following under yolo-coco/ : 
- yolov3.cfg 
- yolov3.weights
- coco.names 
# Usage :
- python social_distance_violation.py --input pedestrians.mp4
- python social_distance_violation.py --input pedestrians.mp4 --output output.avi
# <img
<img src="Capture45.png”/>
