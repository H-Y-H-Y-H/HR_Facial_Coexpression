from cProfile import label
import cv2
import mediapipe as mp
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from test_model import inverse_model
from normalization import human2robot, reorder_lmks, matric2simple, R_static_face, robot_edge
import time
import socket
# import predictive model
import sys
sys.path.append("../Yuhang/")
from log_folder.model3_lighter.model import pred_model

from lmks_data.face_geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)


class remote_Env():
    def __init__(self, episode_len=300, debug=0):
        # Socket Conneciton
        # MAC find WiFi IP - ipconfig getifaddr en0
        HOST = '192.168.0.83'
        # Port to listen on (non-privileged ports are > 1023)
        PORT = 8888

        # Set up Socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))

        print('Waiting for connection[s]...')
        # Wait for client to join socket
        self.s.listen()
        self.conn, addr = self.s.accept()
        print('Connected by: ', addr)

        self.robot_state = None
        self.servo_cmds = np.zeros(11, dtype=np.float32)
        # Start counter
        self.i = 0
        self.debug = debug

    def process_action(self, cmd):
        cmd = np.asarray(cmd,dtype=np.float32)
        # Prepare and send position data to clients
        self.conn.sendall(cmd.tobytes())

    def get_msg_from_pi(self):
        # Robot state is returned by reading from the motors
        self.robot_state = np.frombuffer(self.conn.recv(1024), dtype=np.float32)
        return np.array(list(self.robot_state), dtype=np.float32)

    def step(self, cmds):
        # send cmds to robot
        self.process_action(cmds)
        # Receive data from robot
        recv_msg = self.get_msg_from_pi()

        # Update counter
        self.i += 1
        return recv_msg



def process_img2cmds(i):
    global H_static_face

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array(
                        [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                    )
            landmarks = landmarks.T
            landmarks = landmarks[:, :468]

            metric_landmarks, _ = get_metric_landmarks(
                landmarks.copy(), pcf
            )

            simple_lmks = matric2simple(metric_landmarks)


            if i < static_flag:
                H_static_face = np.copy(simple_lmks)


            # data for pred model
            input_buffer.append(simple_lmks)
            del input_buffer[0]
            # !!!!!!!!!! if it is not activate, static face output!
            normalized_lmks = human2robot(H_static_face, H_static_face)

            distance = np.sum((simple_lmks-H_static_face)**2)
            if distance > threshold:
                print("PRED MODEL ACTIVE!")
                input_lmks = np.asarray(input_buffer)
                input_lmks = torch.from_numpy(input_lmks).to(device, dtype=torch.float).unsqueeze(0)
                input_lmks = torch.flatten(input_lmks, 1)
                outputs = pred_model.forward(input_lmks)
                outputs = outputs.cpu().detach().numpy()
                simple_lmks = np.reshape(outputs, [2, 113])
                # print(simple_lmks)
                # print(H_static_face)
                # time.sleep(0.1)
                normalized_lmks = human2robot(simple_lmks, H_static_face)

            normalized_lmks_2 = reorder_lmks(normalized_lmks)

            input_d = torch.from_numpy(np.expand_dims(normalized_lmks_2, axis=0))
            input_d = torch.flatten(input_d, 1)
            cmds = model.forward(input_d.float())

            # return simple_lmks[0], simple_lmks[1], normalized_lmks[0], normalized_lmks[1], cmds.tolist()[0]
            return cmds.tolist()[0]



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
# frame_width, frame_height = 1920, 1080
frame_width, frame_height = 320, 480
channels = 3
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)
pcf = PCF(
    near=1,
    far=10000,
    frame_height=frame_height,
    frame_width=frame_width,
    fy=camera_matrix[1, 1],
)

face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

# import inverse model
PATH = "inverse_model/best_model_MSE.pt"

device = torch.device('cpu')
model = inverse_model()
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

model_path = "../Yuhang/log_folder/model3_lighter/best_model.pt"
device = torch.device('cpu')
pred_model = pred_model()
pred_model.load_state_dict(torch.load(model_path, map_location=device))
pred_model.eval()
torch.no_grad()
test_ID_logger = np.loadtxt("../Yuhang/pred_dataset/test/test_emoIdx.csv")


use_Camera = False
TESTID = 118
use_camera = False
SOCKET_control_robot = True

if SOCKET_control_robot == True:
    socket_env = remote_Env()

subj_ID = test_ID_logger[0][TESTID]
emo_ID = test_ID_logger[1][TESTID]
input_buffer = [0] * 4
success = 1
H_static_face = []
static_flag = 4
count = 0
threshold = 1
print(subj_ID,emo_ID )

video_path = "/Volumes/yuhang_ssd/Dataset(Face)/video/S%02d/%d.mp4"%(subj_ID,emo_ID)
cap = cv2.VideoCapture(video_path)

while success:


    success, r_image = cap.read()

    r_image.flags.writeable = False
    image = cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    cv2.imshow("Input", r_image)
    c_list = process_img2cmds(count)

    if SOCKET_control_robot == True:
        rec = socket_env.step(c_list)



    count +=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()


