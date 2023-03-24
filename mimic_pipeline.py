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
from lmks_data.face_geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
frame_width, frame_height = 1920, 1080
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
cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

fig = plt.figure()
widths = [1, 1]
heights = [2, 1]
spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[1, :])


def get_static_face():
    face_list = []
    for _ in range(20):
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            return 0, 0

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image = cv2.resize(image,(960,540))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Scanning", rgb_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
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
                face_list.append(simple_lmks)
    cv2.destroyAllWindows()

    return np.mean(face_list,axis=0)



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



# For webcam input:
def camera_data(i):
    time0 = time.time()
    success, image = cap.read()
    time2 = time.time()
    print("camera_frame_time:",time2-time0)
    if not success:
        print("Ignoring empty camera frame.")
        return 0, 0

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.resize(image,(960,540))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Input", rgb_image)
    print(i)
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

            distance = np.sum((simple_lmks-H_static_face)**2)

            # !!!!!!!!!! if it is not activate, static face output!
            normalized_lmks = human2robot(H_static_face, H_static_face)
            distance = np.sum((simple_lmks-H_static_face)**2)
            if distance > threshold:

                normalized_lmks = human2robot(simple_lmks, H_static_face)


            normalized_lmks_2 = reorder_lmks(normalized_lmks)

            input_d = torch.from_numpy(np.expand_dims(normalized_lmks_2, axis=0))
            input_d = torch.flatten(input_d, 1)
            cmds = model.forward(input_d.float())

            return simple_lmks[0], simple_lmks[1], normalized_lmks[0], normalized_lmks[1], cmds.tolist()[0]




def update(i):

    x0, y0, x, y, c_list = camera_data(i)
    rec = socket_env.step(c_list)
    time1 = time.time()
    # xi = [i for i in range(len(c_list))]
    # x_label = ['eye_0', 'eye_1', 'eye_2', 'eye_3', 'mouth_0', 'mouth_1', 'mouth_2', 'mouth_3', 'mouth_4', 'mouth_5', 'jaw']
    # ax1.cla()
    ax2.cla()
    # ax3.cla()
    # ax1.scatter(x0,y0,s=5,c='g')
    #
    # # Loop over data points; create box from errors at each point
    # xdata, ydata, xerror, yerror = R_static_face[0], R_static_face[1], robot_edge[:2], robot_edge[2:]
    # errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
    #             for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]
    #
    # # Create patch collection with specified colour/alpha
    # pc = PatchCollection(errorboxes, facecolor='red', alpha=0.5,
    #                     edgecolor='none')
    #
    # # Add collection to axes
    # ax2.add_collection(pc)
    ax2.scatter(x,y,s=5,c='k')
    # ax3.scatter(xi, c_list, marker='s', c='tab:orange')
    # plt.xticks(xi, x_label)
    # ax1.axes.xaxis.set_ticks([])
    # ax1.axes.yaxis.set_ticks([])
    ax2.axes.xaxis.set_ticks([])
    ax2.axes.yaxis.set_ticks([])
    # ax1.set_xlim(-10, 10)
    # ax1.set_ylim(-10, 10)
    # ax2.set_xlim(-10, 10)
    # ax2.set_ylim(-10, 10)
    # ax3.set_ylim(-1, 12)
    # ax3.set_ylim(-0.1, 1.1)
    # ax3.grid()
    #
    # ax1.set_title("Original face")
    # ax2.set_title("Normaiized Face")
    # ax3.set_title("Predicted commands")
    # fig.set_size_inches(8, 6)
    time2 = time.time()

    print(time2-time1)


if __name__ == "__main__":

    # import inverse model
    PATH = "inverse_model/best_model_MSE.pt"

    device = torch.device('cpu')
    model = inverse_model()
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    # import predictive model
    import sys
    sys.path.append("../Yuhang/")


    H_static_face = get_static_face()

    input_buffer = [H_static_face]*4
    socket_env = remote_Env()

    threshold = 5

    ani = FuncAnimation(plt.gcf(), update, interval=1)
    plt.show()
    cap.release()