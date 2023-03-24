import re
import time
import os
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import csv
import glob
from face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh



frame_width, frame_height = 320, 480  # face dataset
# frame_width, frame_height = 1280, 720  # emo robot
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture('video/002.mp4')

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

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def get_sub_folders(folder):
    folder_list = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    return folder_list


def image_to_matrix(img_folder, output_lmks_save_file_name,rendered_path = None):
    all_metric_landmarks = []
    with mp_face_mesh.FaceMesh(
        # plot version
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        # save version
            # max_num_faces=1,
            # # refine_landmarks=True,
            # static_image_mode=False,
            # min_detection_confidence=0.5,
            # min_tracking_confidence=0.5) as face_mesh:
        count = 0
        for name in sorted(glob.glob(img_folder + '/*.png'), key=numericalSort):
            print(name)
            image = cv2.imread(name)
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # to rgb (for human face)

            # frame_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # image = np.dstack((image,image,image))
            # frame_gray = cv2.equalizeHist(frame_gray)
            # image = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)

            # results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # save version

                    landmarks = np.array(
                        [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                    )
                    landmarks = landmarks.T
                    landmarks = landmarks[:, :468]

                    metric_landmarks, pose_transform_mat = get_metric_landmarks(
                        landmarks.copy(), pcf
                    )

                    # print(metric_landmarks)
                    all_metric_landmarks.append(metric_landmarks)


                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #         .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())
            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            # time.sleep(0.2)
            # if cv2.waitKey(5) & 0xFF == 27:
            #     break
            if rendered_path != None:
                cv2.imwrite(rendered_path+"/%d.png"%count,image)
            count +=1
    np.save(output_lmks_save_file_name, all_metric_landmarks)


def video_to_matrix(filename, output):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    all_metric_landmarks = []

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            # refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
                # continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            # image.flags.writeable = False
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            frame_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)
            image = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)

            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = np.array(
                        [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                    )
                    landmarks = landmarks.T

                    metric_landmarks, pose_transform_mat = get_metric_landmarks(
                        landmarks.copy(), pcf
                    )

                    # print(metric_landmarks)
                    all_metric_landmarks.append(metric_landmarks)

                    # print(len(metric_landmarks[0]))
                    # for xyz in metric_landmarks:
                    #     landmark = [lm.x, lm.y, lm.z]
                    #     writer.writerow(landmark)

                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #         .get_default_face_mesh_tesselation_style())
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #         .get_default_face_mesh_contours_style())
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_IRISES,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #         .get_default_face_mesh_iris_connections_style())
            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            # time.sleep(0.1)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    np.save(output, all_metric_landmarks)
    cap.release()


if __name__ == "__main__":
    
    # # one image
    # image_to_matrix("/Users/jionglin/Downloads/mouth_eye/0524_gau_mouth_2", "robot_data2/mouth_02")



    # read and process video and save .npy file into sourcedata
    data_path = "/Users/yuhang/Downloads/real_robodata/"
    image_path = data_path + "mouth1129/"
    try:
        os.mkdir(data_path+"/mouth_rendered/")
    except:
        pass
    image_to_matrix(image_path,
                    output_lmks_save_file_name = data_path + 'mouth_lmks.npy',
                    rendered_path = data_path+"/mouth_rendered/")
    # pass
    # transfer_video(i, j, frames_folder)




    # subject_folder = get_sub_folders(video_path)
    #
    # # read and process video and save .npy file into sourcedata
    # for i in range(len(subject_folder)):
    #     frames_folder = os.listdir(video_path + '/' + subject_folder[i])
    #     # print(frames_folder)
    #     matrix_path = os.path.join(out_path, subject_folder[i])
    #     # print(matrix_path)
    #     # os.mkdir(matrix_path)
    #
    #     for j in range(len(frames_folder)):
    #         filename = video_path + '/' + subject_folder[i] + '/' + frames_folder[j]
    #         output_name = frames_folder[j].split('.')[0]
    #         video_to_matrix(filename, matrix_path + '/' + output_name)
    #         # pass
    #         # transfer_video(i, j, frames_folder)

    print("end")
