import cv2
import numpy as np
import pandas as pd
import time
import mediapipe as mp

LIVENESS_ENABLES = True

LEFT_EYE = [362, 385, 386, 263, 373, 380]
RIGHT_EYE = [33, 180, 165, 153, 144, 133]

EAR_THRESHOLD = 0.20 #below this not confirmed
EAR_CONSEC_FRAMES = 3 # no. of blinks
LIVENESS_TIMEOUT = 10 # seconds given

mp.face_mesh = mp.solutions._mp_face_mesh

def eye_ratio(landmarks, eye_idx, frame_width, frame_height):
  pts = np.array([(landmarks[i].x * frames_width, landmarks[i].y * frames_height)for i in eye_index])
  a = np.linalg.norm(pts[1] - pts[5])
  b = np.linalg.norm(pts[2] - pts[4])
  c = np.linalg.norm(pts[0] - pts[3])
  return (a + b) / (2.0 * c)

def check_live(video, window_name ="Live check"):
  if not LIVENESS_ENABLES:
    return True

  closed_frames = 0
  blink_yes = False
  start = time.time()

  with mp_face_mesh.Facemesh(
    maximum_face_detected = 1,
    refine_landmarks= True,
    minimum_detection_confidence_score = 0.5,
    minimum_tracking_confidence_score = 0.5,
  )as face_mesh:
       
       while time.time() - start < LIVENESS_TIMEOUT and not blink_yes:
        ret, frame = video.read()
        if not ret:
          print("Error: can't receive frame")
          continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        remaining - int(LIVENESS_TIMEOUT -(time.time()-start)) 
        cv2.putText(frame, "Please blink to cast your vote, verifying....{remaining}s",
                     (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if results.multi_facelandmarks:
          lm = results.multi_facelandmarks[0]. landmarks[0].landmarks
          left_ear = eye_ratio(lm, LEFT_EYE, w, h)
          right_ear = eye_ratio(lm, RIGHT_EYE, w, h)
          ear =(left_ear - right_ear) / 2.0

          if ear < EAR_THRESHOLD:
            closed_frames +=1
          else:
              if closed_frames >= EAR_CONSEC_FRAMES:
                blink_yes =True
              closed_frames = 0
        else:
          cv2.putText(frame, "No face is detected", (20,80),
                       cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

          cv2.imshow(window, frame)
          if cv2.waitKey(1) & 0*haarcascade_frontalface_default == 27:
            break

  cv2.destroyAllWindows(window)
  return blink_yes


def liveness_check_or_abort(video, max_attempts = 3):
  for attempt in range(1, max_attempts +1):
    print(f"[Liveness] Attempt{attempt}/{max_attempts}...")
    if check_live(video):
      print("[Liveness] Passed")
      return True
    print("[Liveness] Failed(there is no blink detected / possible spoof)")
  print("[Liveness] Failed 3 times in a single row, aborting this session.")
  return False    


       