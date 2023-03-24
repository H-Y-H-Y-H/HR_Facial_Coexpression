from re import T
import simplified_landmark_plus as sl
import numpy as np
import pandas as pd


eye_left = [11,8,10,6,9, 5,7,3,4,2, 12,24,18,19,20,21,22,25,1,0,23,13,14,15,16,17]
eye_right = [37,34,36,32,35, 31,33,29,30,28, 38,50,44,45,46,47,48,51,27,26,49,39,40,41,42,43]


mouth_center = [0,1,2,3,25]
mouth_left = [4,5,6,30,8,21,16,29,13, 12,11,10,31,9,18,15,28,14, 22,27,23,24,20,26,7,19,17,32]
mouth_right = [33,34,35,58,37,50,45,57,42, 41,40,39,59,38,47,44,56,43, 51,55,52,53,49,54,36,48,46,60]


def avg_lmks(eyes_list, mouth_list):
    # avg_eyes
    avg_eyes_list = []
    for eyes in eyes_list:
        avg_eyes_x = (-eyes[0, eye_left] + eyes[0, eye_right]) / 2
        avg_eyes_y = (eyes[1, eye_left] + eyes[1, eye_right]) / 2
        avg_eyes = np.array([avg_eyes_x, avg_eyes_y])
        avg_eyes_list.append(avg_eyes)

    # avg_mouth
    avg_mouth_list = []
    for mouth in mouth_list:
        xm = [.0,.0,.0,.0,.0]
        ym = []
        for id in mouth_center:
            ym.append(mouth[1][id])
        avg_mouth1 = np.array([np.array(xm), np.array(ym)])
        avg_mouth2_x = (-mouth[0, mouth_left] + mouth[0, mouth_right]) / 2
        avg_mouth2_y = (mouth[1, mouth_left] + mouth[1, mouth_right]) / 2
        avg_mouth2 = np.array([avg_mouth2_x, avg_mouth2_y])

        avg_mouth = np.concatenate((avg_mouth1, avg_mouth2), axis=1)
        avg_mouth_list.append(avg_mouth)

    return np.array(avg_eyes_list), np.array(avg_mouth_list)
    



def input_data_aug(eyes_data, mouth_data):
    print(eyes_data.shape, mouth_data.shape)

    all_faces = []
    for e_id in range(len(eyes_data)):
        for m_id in range(len(mouth_data)):
            print(e_id, m_id)

            combined_face_x = []
            combined_face_y = []
            for idx in sl.contour_idx:
                if idx in sl.eyes_idx:
                    combined_face_x.append(eyes_data[e_id][0][idx])
                    combined_face_y.append(eyes_data[e_id][1][idx])
                elif idx in sl.mouth_idx:
                    combined_face_x.append(mouth_data[m_id][0][idx])
                    combined_face_y.append(mouth_data[m_id][0][idx])

            combined_face = np.array([np.array(combined_face_x), np.array(combined_face_y)])
            all_faces.append(combined_face)
    
    return np.array(all_faces)


def labels_aug(eyes_commands, mouth_commands):
    all_cmds = []
    for e_id in range(len(eyes_commands)):
        for m_id in range(len(mouth_commands)):
            print(e_id, m_id)
            cmds = np.concatenate((eyes_commands[e_id], mouth_commands[m_id]))
            # print(eyes_commands[e_id], mouth_commands[m_id], cmds)
            all_cmds.append(cmds)
    
    return np.array(all_cmds)


def plot_static_face(static_face):
    pass


def half_mouth(mouth_data, seq=True):
    id_list = [0, 1, 2, 3, 4, 5, 10]
    if seq:
        return mouth_data[:, id_list]
    # return mouth_data[]




if __name__ == "__main__":

    data_path = "/Users/yuhang/Downloads/real_robodata/"

    matric_eyes = np.load(data_path + "eyes_lmks.npy")
    matric_mouth = np.load(data_path + "mouth_lmks.npy")


    simple_eyes_half = sl.mash_to_contour_half(sl.face_3d_to_2d(matric_eyes, seq=True), seq=True, eyes=True)
    simple_mouth_half = sl.mash_to_contour_half(sl.face_3d_to_2d(matric_mouth, seq=True), seq=True, mouth=True)
    print(matric_eyes.shape, matric_mouth.shape)
    print(simple_eyes_half.shape, simple_mouth_half.shape)

    avg_eyes, avg_mouth = avg_lmks(simple_eyes_half, simple_mouth_half)
    print(avg_eyes.shape, avg_mouth.shape)
    np.save(data_path + "full_eyes(1000x2x52).npy", simple_eyes_half)
    np.save(data_path + "full_mouth(1129x2x61).npy", simple_mouth_half)

    # np.save("./robot_data2/simple_eyes_half", simple_eyes_half)
    # np.save("./robot_data2/simple_mouth_half", simple_mouth_half)
    # sl.plot_2d("../../real_R_data/training_dataset/avg_mouth(1712x2x33).npy", contour=False)
    # sl.plot_2d_face(simple_eyes_half[1], text=False)

    # all_data_input = input_data_aug(matric_eyes, matric_mouth)
    # np.save("./robot_data2/all_combined.npy", all_data_input)
    # print(all_data_input.shape)

    # eyes_cmds=np.array(pd.read_csv('../../real_R_data/RAW_data/eyes_motor_cmds.csv', sep=' ', header=None))
    # mouth_cmd0=pd.read_csv('../../real_R_data/RAW_data/mouth_motor_cmd0.csv', sep=' ', header=None)
    # mouth_cmds1=pd.read_csv('../../real_R_data/RAW_data/mouth_motor_cmds1.csv', sep=' ', header=None)
    # mouth_cmds2=pd.read_csv('../../real_R_data/RAW_data/mouth_motor_cmds2.csv', sep=' ', header=None)

    # mouth_cmds = np.concatenate((mouth_cmd0, mouth_cmds1, mouth_cmds2), axis=0)
    # mouth_cmds2 = np.concatenate((mouth_cmds1, mouth_cmds2), axis=0)
    # mouth_cmds_half = half_mouth(mouth_cmds2)

    # print(mouth_cmds2.shape, mouth_cmds_half.shape, eyes_cmds.shape)

    # np.save("../../real_R_data/training_dataset/eyes_cmds(609,4).npy", eyes_cmds)
    # np.save("../../real_R_data/training_dataset/mouth_cmds(1712,7).npy", mouth_cmds_half)
    # for c in mouth_cmds:
        # if c[1] == c[9] and c[2] == c[8] and c[3] == c[7] and c[4] == c[6]:
        #     # print("yes")
        #     pass
        # else:
        #     print("Nooooooo")

    # all_cmds = labels_aug(eyes_cmds, mouth_cmds)
    # print(all_cmds.shape)
    # np.save("./robot_data2/all_cmds.npy", all_cmds)


    # half face landmarks



