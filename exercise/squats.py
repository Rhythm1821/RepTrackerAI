import mediapipe as mp
import cv2
from utils import calculate_angle, mp_drawing,mp_pose
import numpy as np

cap = cv2.VideoCapture(0)

# squat counter variables
counter = 0
stage = None

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret,frame = cap.read()

        # Recolor image
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False

        # Make detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable=True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angle
            angle = calculate_angle(hip,knee,ankle)

            # Visualize
            cv2.putText(image,str(angle),
                        tuple(np.multiply(knee,[640,480]).astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)

            # Count squats
            if angle > 160:
                stage = "Up"
            if angle < 90 and stage=="Up":
                stage="Down"
                counter+=1
        except Exception as e:
            print(e)

        # Render squats counter
        # setup status boxes
        cv2.rectangle(image,(0,0),(225,73),(245,117,16),-1)

        # Rep data
        cv2.putText(image,'REPS',(15,12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,str(counter),(10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image,'STAGE',(65,12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,stage,(90,60),
                    cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))

        cv2.imshow('Webcam',image)

        if cv2.waitKey(10) & 0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()