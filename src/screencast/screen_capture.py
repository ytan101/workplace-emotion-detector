# Import packages
import av
import cv2
import numpy as np
import pyautogui
from keras.models import load_model
from mss import mss

from .. import constants

# Load models from Directory
model = load_model(constants.MODEL_PATH)


class EmotionDetectionScreen:
    def rescale_frame(self, frame, scale):
        # Resize the frame to a thumbnail
        # works for image, video, live video
        """Creates a thumbnail of the live video

        :param frame: image feed for webcam
        :type frame: video
        :param scale: rescaling of the live feed
        :type scale: float
        :return: resized image feed from webcam
        :rtype: video
        """
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)
        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

    def screen_capture_emotion(self):
        labels_dict = {
            0: "angry",
            1: "digust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
        }
        color_dict = {
            0: (0, 0, 255),
            1: (0, 255, 0),
            2: (0, 0, 255),
            3: (255, 255, 0),
            4: (255, 0, 255),
            5: (0, 0, 0),
        }

        size = 4

        # We load the xml file from opencv
        classifier = cv2.CascadeClassifier(constants.HAAR_PATH)

        width, height = pyautogui.size()

        bounding_box = {"top": 0, "left": 0, "width": width, "height": height}

        sct = mss()

        while True:
            # (rval, im) = webcam.read()
            im = np.array(sct.grab(bounding_box))
            # im=cv2.flip(im,1,1) #Flip to act as a mirror
            frame_resized = self.rescale_frame(im, scale=0.2)
            cv2.imshow("Video Resized", frame_resized)
            if cv2.waitKey(20) & 0xFF == ord("q"):  # press "q" key to exit loop
                break

            # # Resize the image to speed up detection
            mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

            # detect MultiScale / faces
            faces = classifier.detectMultiScale(mini)

            # Draw rectangles around each face
            for f in faces:
                (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
                # Save just the rectangle faces in SubRecFaces
                face_img = im[y : y + h, x : x + w]
                resized = cv2.resize(face_img, (90, 90))
                normalized = resized / 255.0
                normalized = normalized[:, :, :3]
                print(normalized.shape)
                reshaped = np.reshape(normalized, (1, 90, 90, 3))
                reshaped = np.vstack([reshaped])
                result = model.predict(reshaped)
                print(result)

                label = np.argmax(result, axis=1)[0]

                cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 3)
                cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
                cv2.putText(
                    im,
                    labels_dict[label],
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    3,
                )

            # Show the image
            cv2.imshow("LIVE", im)
            key = cv2.waitKey(10)
            # if Esc key is press then break out of the loop
            if key == 27:  # The Esc key
                break


if __name__ == "__main__":
    emotion_detection_screen = EmotionDetectionScreen()
    emotion_detection_screen.screen_capture_emotion()
