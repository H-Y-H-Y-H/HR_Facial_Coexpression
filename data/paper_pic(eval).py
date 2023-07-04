import json
import numpy as np
import matplotlib.pyplot as plt

def unit_measure():
    static_face_lmk_pth = "pred_dataset/H_static_face/static_face_lmks1.csv"
    lmk = np.loadtxt(static_face_lmk_pth)

    I0 = -1.59
    I1 = 2.38

    dist_list = []
    for i in range(len(lmk[0])):
        dist = abs(I0 - lmk[0][i]) + abs(I1 - lmk[1][i])

        dist_list.append(dist)

    plt.scatter(lmk[0], lmk[1])
    k = np.argmin(dist_list)
    print(2 * abs(lmk[0, k]))

    plt.scatter(lmk[0, k], lmk[1, k], c='r')
    plt.show()
# unit_measure()

def demo_14_figure():
    # frame timestamp: 25 fps
    test_list = [15, 20, 122, 160, 99, 181, 95, 154, 205, 204, 212, 188, 179, 185] # this is test id. not original id.
    test_id = np.loadtxt("pred_dataset/test/test_emoIdx.csv")
    real_test_list = test_id[1][test_list] # Here we got original id
    print(real_test_list)

    all_subject_gap = []
    for subjectID in range(1,46):
        json_path = "pred_dataset/selection_log/seg_data_S%02d.json"%(subjectID)
        with open(json_path, "r") as read_file:
            selection_data = json.load(read_file)
            emo_keys = selection_data.keys()
            for key in emo_keys:
                if int(key) in real_test_list:
                    print(test_list[len(all_subject_gap)])
                    print((selection_data[key][0]/25,selection_data[key][1]/25))
                    all_subject_gap.append((selection_data[key][0], selection_data[key][1]))

    print(all_subject_gap)

# demo_14_figure()

def evaluate_dataset():
    all_subject_gap = []
    for subjectID in range(1,46):
        json_path = "pred_dataset/selection_log/seg_data_S%02d.json"%(subjectID)
        with open(json_path, "r") as read_file:
            selection_data = json.load(read_file)
            emo_keys = selection_data.keys()
            dist = []
            for key in emo_keys:
                dist.append(selection_data[key][1] - selection_data[key][0])
                all_subject_gap.append((selection_data[key][1]-selection_data[key][0]))
            print(subjectID)
            print(np.mean(dist))
            print(np.std(dist))
            print(np.max(dist))
            print(np.min(dist))
            print("---------")

    print(all_subject_gap)
    avg_time = np.mean(all_subject_gap)
    std_time = np.std(all_subject_gap)
    print(avg_time)
    print(std_time)
    print(np.max(all_subject_gap))
    print(np.min(all_subject_gap))
    print(np.median(all_subject_gap))
    print(len(all_subject_gap))
    print('the average time for the facial expression in our data:',avg_time/25)
    print('the stand dev time for the facial expression in our data:',std_time/25 )

# evaluate_dataset()


def eval_pred_model_with_bl():
    bl1_loss_list = np.loadtxt("pred_dataset/test/bl1_loss.csv")
    bl2_loss_list = np.loadtxt("pred_dataset/test/bl2_loss.csv")
    pred_loss = np.loadtxt("pred_dataset/test/test_loss.csv")

    for data_name in [bl1_loss_list,bl2_loss_list,pred_loss]:
        print(np.mean(data_name),np.std(data_name),np.min(data_name),np.max(data_name))

        print("----")

    bl1_loss_list *= 9.0124
    bl2_loss_list *= 9.0124
    pred_loss *= 9.0124

    fig, ax = plt.subplots()

    data = [bl2_loss_list, bl1_loss_list ,pred_loss]
    # Creating plot
    bp = ax.boxplot(data, showfliers=False,labels = ['Random Search Baseline', 'Mimicry Baseline ','Our Method'])
    ax.set_title('Evaluations of the Predictive Model')
    # ax.legend()
    plt.show()

# The result is in pixel unit. See the excel for the further unit process to mm.
# eval_pred_model_with_bl()


