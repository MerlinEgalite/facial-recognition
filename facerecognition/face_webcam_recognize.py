# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2

dataset = '../Data'
encoded = '../encodings.pickle'


def recognize_faces_webcam():
    """
	_ brief: return the webcam video with the name of the persons present in
	the video if they are in the dataset
	_ param: path of the image to test
	_ return: the input image with the names of the persons
	"""

    # launch the webcam
    video_capture = cv2.VideoCapture(0)

    # load the known faces and embeddings
    data = pickle.loads(open(encoded, "rb").read())

    while True:
        ret, frame = video_capture.read()
        image = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
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
                # find the indexes of all matched faces then initialize a
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
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow('img', image)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# start the function that recongnizes the visage through the webcam
recognize_faces_webcam()
