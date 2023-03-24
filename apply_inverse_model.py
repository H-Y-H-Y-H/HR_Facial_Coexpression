import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import mediapipe as mp
import torch
from test_model import inverse_model
import glob
from normalization import human2robot, reorder_lmks, matric2simple, R_static_face, robot_edge



def H_lmks2R_cmds():

    data_type = 'label'
    # data_type = "test_result_lmks"
    p_data_pth = "../Yuhang/pred_dataset/"
    p_test_label_pth = p_data_pth + "test/%s"%data_type
    label_file_list = glob.glob(p_test_label_pth+"/*.npy")

    subID_list = np.loadtxt(p_data_pth + 'test/test_emoIdx.csv')[0] # first layer is the id
    cmds_list = []
    for i in range(len(label_file_list)):
        H_lmks = np.load(p_test_label_pth + "/%d.npy"%i)
        if data_type == "label":
            H_lmks = H_lmks[0]

        H_subID = subID_list[i]

        H_static_face = np.loadtxt(p_data_pth+ "H_static_face/static_face_lmks%d.csv"%H_subID)

        normalized_lmks = human2robot(H_lmks, H_static_face)
        normalized_lmks_2 = reorder_lmks(normalized_lmks)

        input_d = torch.from_numpy(np.expand_dims(normalized_lmks_2, axis=0))
        input_d = torch.flatten(input_d, 1)
        predict = model.forward(input_d.float())
        predict = predict.tolist()[0]
        cmds_list.append(predict)
    cmds_list = np.asarray(cmds_list)
    print(cmds_list.shape)

    # x_label = ['eye_0', 'eye_1', 'eye_2', 'eye_3', 'mouth_0', 'mouth_1', 'mouth_2', 'mouth_3', 'mouth_4', 'mouth_5', 'jaw']

    np.savetxt(p_data_pth+ "test/%s_cmds.csv"%data_type, np.asarray(cmds_list))
    print("finished")

def R_lmks2R_cmds():
    subj = "22"
    r_lmk = np.load("lmks_data/simple_static.npy")
    print(r_lmk.shape)
    p_data_pth = "audio_test/%s.npy"%subj
    lmks_list = np.load(p_data_pth)

    cmds_list = []
    for i in range(len(lmks_list)):
        R_lmks = lmks_list[i]
        print(R_lmks.shape)

        R_lmks = reorder_lmks(R_lmks)
        input_d = torch.from_numpy(np.expand_dims(R_lmks, axis=0))
        input_d = torch.flatten(input_d, 1)
        predict = model.forward(input_d.float())
        predict = predict.tolist()[0]
        cmds_list.append(predict)
        print(predict)
    cmds_list = np.asarray(cmds_list)
    print(cmds_list)

    # x_label = ['eye_0', 'eye_1', 'eye_2', 'eye_3', 'mouth_0', 'mouth_1', 'mouth_2', 'mouth_3', 'mouth_4', 'mouth_5', 'jaw']

    np.savetxt("audio_test/cmds%s.csv"%subj, np.asarray(cmds_list))


if __name__ == "__main__":
    PATH = "inverse_model/best_model_L1.pt"
    device = torch.device('cpu')
    model = inverse_model()
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()
    R_lmks2R_cmds()
    # H_lmks2R_cmds()


