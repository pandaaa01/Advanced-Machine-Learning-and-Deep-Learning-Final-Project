import cv2
from ultralytics.utils.plotting import Annotator
from itertools import chain
import numpy as np


def detect_ppe(video_file, ppe):
    i = 0
    # Read the video file
    vid = cv2.VideoCapture(video_file)


    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video/pose_output_video.mp4', -1, 15.0, (int(vid.get(3)), int(vid.get(4))))

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        
        # Predict
        result = ppe.predict(frame)
        plotted = result[0].plot()
        
        out.write(plotted)
        cv2.imshow('frame', plotted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()

def haar_helper(frame, model):
    # Reshape the input data to match the model's input shape
    frame = frame.reshape(1, -1)  # Reshape to (1, 34) or (1, -1) depending on the actual shape

    # Perform prediction
    prediction = model.predict(frame)
    res = prediction[0] >= 0.6
    return int(res)

def haar_recognition(video, haar_model, keypoints_detector):
    PRED_MAP = {
        0: 'Not Working',
        1: 'Working'
    }

    vid = cv2.VideoCapture(video)

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video/haar_output_video.mp4', -1 , 15.0, (int(vid.get(3)), int(vid.get(4))))


    while True:
        ret, frame = vid.read()
        if not ret:
            break
        
        # Predict
        results = keypoints_detector.predict(frame, conf = 0.70)
        annotator_frame = Annotator(frame)
        boxes = results[0].boxes
        for idx, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = box

            # Preprocess keypoints to predict
            input_data = results[0].keypoints.xy[idx]
            input_data = [[max(i-x1, 0), max(0, j-y1)] for i, j in input_data]
            input_data = np.array(list(map(float, list(chain(*input_data)))))

            # Predict
            pred = haar_helper(input_data, haar_model)
            annotator_frame.box_label(box, PRED_MAP[pred])

        to_display = annotator_frame.result()
        out.write(to_display)
        cv2.imshow('Test', to_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
    


def detect_img(image, ppe_model):
    frame = cv2.imread(image)
    result = ppe_model.predict(frame)

    plotted = result[0].plot()
    cv2.imwrite('output_video/pose.jpg', plotted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()