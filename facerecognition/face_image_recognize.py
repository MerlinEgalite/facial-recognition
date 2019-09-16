# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2


dataset = '../Data'
encoded = '../encodings.pickle'


def recognize_faces_image(image_test):
	"""
	_ brief: return the input image with the name of the persons present in
	photo if they are in the dataset
	_ param: path of the image to test
	_ return: the input image with the names of the persons
	"""

	# load the known faces and embeddings
	print("[INFO] loading encodings...")
	data = pickle.loads(open(encoded, "rb").read())

	# load the input image and convert it from BGR to RGB
	image = cv2.imread(image_test)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	print("[INFO] recognizing faces...")
	boxes = face_recognition.face_locations(rgb, model='hog')
	encodings = face_recognition.face_encodings(rgb, boxes)

	# initialize the list of names for each face detected
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"], encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize
			matched_idxs = [i for (i, b) in enumerate(matches) if b]

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matched_idxs:
				name = data["names"][i]

		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

recognize_faces_image('/Users/merlinegalite/Desktop/CS/CodingWeeks/facialrecognition/Test/IMG_0392.jpeg')
