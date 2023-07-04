import os
from cProfile import label
from cgitb import text
from turtle import color
import torch
import numpy as np
from torch import nn, seed
from torch.utils.data import Dataset, DataLoader
from model_v2 import inverse_model
import matplotlib.pyplot as plt
import random
random.seed(2022)
import time
# Check GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("start", device)

batchsize = 1 # 128
num_epoches = 1


class pred_data(Dataset):
    def __init__(self, input_data, label_data):
        self.input_data = input_data
        self.label_data = label_data

    def __getitem__(self, idx):

        input_data_sample = self.input_data[idx]
        label_data_sample = self.label_data[idx]
        input_data_sample = torch.from_numpy(input_data_sample).to(device, dtype=torch.float)
        label_data_sample = torch.from_numpy(label_data_sample).to(device, dtype=torch.float)
        sample = {"input": input_data_sample, "label": label_data_sample}
        return sample

    def __len__(self):
        return len(self.input_data)

def pecent_error(v_pred,v_gt):
    return abs((v_pred - v_gt)/v_gt)*100

def test_model():
    all_predict = []
    all_labels = []
    test_epoch_L  = []
    Loss_fun = nn.L1Loss(reduction='mean')
    n=0
    model.eval()
    loss_list = []
    print("test_model...")

    with torch.no_grad():
        c_time = time.time()

        for i, bundle in enumerate(test_dataloader):
            n += 1
            input_d, label_d = bundle["input"], bundle["label"]

            input_d = torch.flatten(input_d, 1)
            label_d = label_d[0].detach().cpu().numpy()
            pred_result = model.forward(input_d)[0].detach().cpu().numpy()

            all_predict.append(pred_result)
            all_labels.append(label_d)
            loss = pecent_error(pred_result, label_d)
            loss_list.append(loss)
    one_frame_time = (time.time() - c_time) / n
    np.savetxt(logger_pth + 'ours_prd.csv'   , np.asarray(all_predict))
    np.savetxt(logger_pth + 'ours_lbl.csv'   , np.asarray(all_labels))
    return all_predict, all_labels, loss_list, one_frame_time

def random_select_training_data_test():
    all_predict = []
    all_labels = []
    test_epoch_L  = []

    n = 0

    model.eval()
    loss_list = []
    print("random_select_training_data_test...")
    with torch.no_grad():
        c_time = time.time()
        for i, bundle in enumerate(test_dataloader):
            n += 1
            _, label_d = bundle["input"], bundle["label"]

            train_id = random.randint(0, 722400-1)

            input_d = torch.from_numpy(train_input_data[train_id].reshape(1,226)).float()

            # label_d = torch.flatten(label_d, 1)
            label_d = label_d[0].detach().cpu().numpy()
            pred_result = model.forward(input_d)
            pred_result= pred_result[0].detach().cpu().numpy()

            all_predict.append(pred_result)
            all_labels.append(label_d)
            loss = pecent_error(pred_result, label_d)
            loss_list.append(loss)

    np.savetxt(logger_pth + 'rdm_fac_prd.csv', np.asarray(all_predict))
    np.savetxt(logger_pth + 'rdm_fac_lbl.csv', np.asarray(all_labels))

    one_frame_time = (time.time() - c_time) / n
    return all_predict, all_labels, loss_list, one_frame_time


def select_nearest_lmks(num = 3, group_size = 100):
    all_predict = []
    all_labels = []
    start_id = random.randint(0, 722400-101)
    data_group = train_input_data[start_id:start_id+group_size]
    label_group = train_label_data[start_id:start_id+group_size]

    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    test_epoch_L  = []
    Loss_fun = nn.L1Loss(reduction='mean')
    n=0
    model.eval()
    loss_list = []
    print("select_nearest_lmks...")

    with torch.no_grad():
        c_time = time.time()
        for i, bundle in enumerate(test_dataloader):
            n += 1
            input_d, label_d = bundle["input"], bundle["label"]
            sum_list = [torch.sum(torch.abs(input_d-data_group[idx])) for idx in range(group_size)]
            select_idx_list = np.argsort(sum_list)[:num]

            pred_result_list = label_group[select_idx_list]

            pred_result = np.mean(pred_result_list, axis=0)

            label_d = label_d[0].detach().cpu().numpy()

            all_predict.append(pred_result)
            all_labels.append(label_d)
            loss = pecent_error(pred_result, label_d)
            loss_list.append(loss)

    one_frame_time = (time.time() - c_time) / n
    np.savetxt(logger_pth + 'nn_prd.csv'     , np.asarray(all_predict))
    np.savetxt(logger_pth + 'nn_lbl.csv'     , np.asarray(all_labels))
    return all_predict, all_labels, loss_list, one_frame_time



def plot_mean_std(all_p_list, all_l_list, ftime, record_path="./eval_pred/"):
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    # fig.tight_layout()
    # plt.style.use('bmh')
    record_5 = []

    mean_dist_list = []
    std_dist_list = []
    for t in range(4):
        all_p = all_p_list[t] # predicted results
        all_l = all_l_list[t] # ground truth -> label

        # all_dist = [[] for _ in range(11)]
        # avg_loss_list = []
        # for i in range(len(all_p)):
        #     loss_collect = []
        #     for j in range(11):
        #         cmd_loss = abs(all_p[i][0][j] - all_l[i][0][j])
        #         all_dist[j].append(cmd_loss)
        #         loss_collect.append(cmd_loss)
        #     avg_loss_list.append(np.mean(loss_collect))
        # np.savetxt(logger_pth + 'logger_%s.csv' % name_list[t], np.asarray(avg_loss_list))

        # mean_dist,std_dist = [],[]
        # for i in range(11):
        #     mean_dist.append(np.mean(loss_each_method[i]))
        #     std_dist.append(np.std(all_dist[i]))
        # print(name_list[t])
        # mean_i, std_i, max_i, min_i = np.mean(mean_dist), np.mean(std_dist), np.max(avg_loss_list), np.min(avg_loss_list)
        # mean_i, std_i, max_i, min_i = np.mean(avg_loss_list), np.std(avg_loss_list), np.max(avg_loss_list), np.min(avg_loss_list)
        # print('mean: ', mean_i, 'std: ', std_i)
        # print('max: ', max_i, 'min: ', min_i)
        # mean_dist_list.append(mean_i)
        # std_dist_list.append(std_i)
        # record_5.append(np.array([mean_i, std_i, max_i, min_i, int(1./ftime[t])]))


    x = [x for x in range(11)]
    x_label = ['eye0', 'eye1', 'eye2', 'eye3', 'mouth0', 'mouth1', 'mouth2', 'mouth3', 'mouth4', 'mouth5', 'jaw']
    ax[0,0].plot(x, mean_dist_list[0])
    ax[0,0].errorbar(x, mean_dist_list[0], yerr=std_dist_list[0], fmt='s')
    ax[0,0].set_title(name_list[0]+" (mean: {:.4f},  std: {:.4f}, fps: {})".format(np.mean(mean_dist_list[0]), np.mean(std_dist_list[0]), int(1./ftime[0])),size=10)

    ax[0,1].plot(x, mean_dist_list[1])
    ax[0,1].errorbar(x, mean_dist_list[1], yerr=std_dist_list[1], fmt='s')
    ax[0,1].set_title(name_list[1]+" (mean: {:.4f},  std: {:.4f}, fps: {})".format(np.mean(mean_dist_list[1]), np.mean(std_dist_list[1]), int(1./ftime[1])),size=10)

    ax[1,0].plot(x, mean_dist_list[2])
    ax[1,0].errorbar(x, mean_dist_list[2], yerr=std_dist_list[2], fmt='s')
    ax[1,0].set_title(name_list[2]+" (mean: {:.4f},  std: {:.4f}, fps: {})".format(np.mean(mean_dist_list[2]), np.mean(std_dist_list[2]), int(1./ftime[2])),size=10)

    ax[1,1].plot(x, mean_dist_list[3])
    ax[1,1].errorbar(x, mean_dist_list[3], yerr=std_dist_list[3], fmt='s')
    ax[1,1].set_title(name_list[3]+" (mean: {:.4f},  std: {:.4f}, fps: {})".format(np.mean(mean_dist_list[3]), np.mean(std_dist_list[3]), int(1./ftime[3])),size=10)

    for axi in ax.flat:
        axi.set_ylim(-0.1, 0.8)
        axi.set_xticks(x, x_label, fontsize=8)
        axi.tick_params(axis='y', labelsize=8)
    fig.suptitle('Commands loss evaluation')
    plt.show()
    np.savetxt(record_path + "cmd_eval_record.csv", np.array(record_5))

def inverse_map_cmds_eval(logger_pth):
    """
    1. random commands
    2. random face
    3. nearest neighbour 
    4. our method
    """
    error_rdmf_bl,error_nn_bl,error_rdmc_bl,error_ours = [],[],[],[]
    # 1
    rand_c_time = time.time()
    all_predict1 = np.random.rand(45200,11)
    frame_time1 = (time.time() - rand_c_time) / 45200

    # 2
    all_predict2, all_labels2, error_list2, frame_time2 = random_select_training_data_test()
    all_predict3, all_labels3, error_list3, frame_time3 = select_nearest_lmks()
    all_predict4, all_labels4, error_list4, frame_time4 = test_model()

    all_labels1 = np.copy(all_labels3)

    np.savetxt(logger_pth + 'rdm_cmd_prd.csv', np.asarray(all_predict1))
    np.savetxt(logger_pth + 'rdm_cmd_lbl.csv', np.asarray(all_labels1))






    # all_predict = [all_predict1, all_predict2, all_predict3, all_predict4]
    # all_labels = [all_labels1, all_labels2, all_labels3, all_labels4]
    #
    # frame_time = [frame_time1,frame_time2,frame_time3,frame_time4]
    # plot_mean_std(all_predict, all_labels, frame_time)



if __name__ == '__main__':

    mode = 1


    # obtain raw values:
    if mode ==0:
        data_path = "../real_R_data/new_version/training_dataset"
        eyes_d = np.load(data_path + "/full_eyes(1000x2x52).npy")
        mouth_d = np.load(data_path + "/full_mouth(1129x2x61).npy")
        eyes_c = np.load(data_path + "/eyes_cmds(1000x4).npy")
        mouth_c = np.load(data_path + "/mouth_cmds(1129x7).npy")


        import sys
        sys.path.insert(0, '../Jionglin/data_collection/')
        # from simplified_landmark_plus import face_3d_to_2d, mash_to_contour, mash_to_contour_half, plot_2d_face
        # insert at 1, 0 is the script path (or '' in REPL)
        sys.path.insert(1, '../Jionglin/train_map/')

        # from model_v1 import inverse_model
        from prepare_data import data_aug

        train_input_data, train_label_data, test_input_data, test_label_data = data_aug(eyes_d, mouth_d, eyes_c, mouth_c)
        test_dataset = pred_data(input_data = test_input_data, label_data=test_label_data)
        test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

        PATH = "inverse_model/best_model_MSE.pt"

        device = torch.device('cpu')
        model = inverse_model()
        model.load_state_dict(torch.load(PATH, map_location=device))

        """
        inverse_map commands evaluation
        """
        logger_pth = "inverse_model/paper_data_inverse_model/"
        name_list = [
            "Random_cmds", "Random_face", "Nearest_neighbour", "Inverse_model(cpu)"
        ]
        os.makedirs(logger_pth,exist_ok=True)
        inverse_map_cmds_eval(logger_pth)  # seed and save record version Dec 9 22

    # paper table and plot: calculate the error
    if mode ==1:
        # L1
        def loss_func(pred, label):
            return np.mean(abs(pred - label),axis=1) * 100

        logger_pth = 'inverse_model/paper_data_inverse_model/'

        rdm_cmd_prd = np.loadtxt(logger_pth + 'rdm_cmd_prd.csv')
        rdm_cmd_lbl = np.loadtxt(logger_pth + 'rdm_cmd_lbl.csv')
        rdm_fac_prd = np.loadtxt(logger_pth + 'rdm_fac_prd.csv')
        rdm_fac_lbl = np.loadtxt(logger_pth + 'rdm_fac_lbl.csv')
        nn_prd = np.loadtxt(logger_pth + 'nn_prd.csv' )
        nn_lbl = np.loadtxt(logger_pth + 'nn_lbl.csv' )
        ours_prd = np.loadtxt(logger_pth + 'ours_prd.csv')
        ours_lbl = np.loadtxt(logger_pth + 'ours_lbl.csv')

        loss0 = loss_func(rdm_cmd_prd, rdm_cmd_lbl)
        loss1 = loss_func(rdm_fac_prd, rdm_fac_lbl)
        loss2 = loss_func(nn_prd, nn_lbl)
        loss3 = loss_func(ours_prd, ours_lbl)

        loss_list = [loss0,loss1,loss2,loss3]

        for method_i in range(4):
            data = loss_list[method_i]
            mean_i, std_i,min_i, max_i  = np.mean(data), np.std(data), np.min(data), np.max(data)
            print(mean_i, ',', std_i,',', min_i, ',', max_i)


        fig, ax = plt.subplots()
        import matplotlib.ticker as mtick
        # Creating plot
        bp = ax.boxplot(loss_list, showfliers=False, labels = ["Random Commands", "Random Face", "Nearest Neighbour", "Our method"])
        ax.set_title('Evaluations of the Inverse Model')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        # ax.legend()
        plt.show()












