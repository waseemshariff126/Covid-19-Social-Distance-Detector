# USAGE
# python social_distance_violation.py --input pedestrians.mp4
# python social_distance_violation.py --input pedestrians.mp4 --output output.avi

from detectors import social_distancing_params as config
from detectors.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())


labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"]) # Load the COCO labels
LABELS = open(labelsPath).read().strip().split("\n")

# YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# Load our YOLO object detector 
print(" Loading YOLO")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# If GPU avaible for CUDA
if config.USE_GPU:
	# CUDA as the preferable backend and target
	print("CUDA as the preferable backend and target")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("Initialize the video stream and pointer to output video file")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# loop over the frames from the video stream
while True:
	# read the next frame from the file
	(found, frame) = vs.read()

	# if the frame was not found, then we have reached the end of the stream
	if not found:
		break

	# Resize each frame and detect people
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	# Initialize the set which violate the social distance
	violate = set()

	if len(results) >= 2: #Only 2 or more than people will be affected when in contact
		centroids = np.array([r[2] for r in results]) # Calculaye the distance
		D = dist.cdist(centroids, centroids, metric="euclidean")
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				if D[i, j] < config.MIN_DISTANCE:
					violate.add(i) # update our violation set with the indexes of the centroid pairs
					violate.add(j)
        # Found results
	for (i, (prob, bbox, centroid)) in enumerate(results):		
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0) # initialize the color of the annotation

		if i in violate:
			color = (0, 0, 255) # update the color if violated
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	#  Count of social distancing violations on the output frame
	text = "Social Distancing Vioaltion Count: {}".format(len(violate))
	cv2.putText(frame, text, (20, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (55, 155, 255), 3)

	if args["display"] > 0:
		cv2.imshow("Frame", frame) # show the output frame
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,(frame.shape[1], frame.shape[0]), True)
	if writer is not None: # If video writer is not none
		writer.write(frame)
