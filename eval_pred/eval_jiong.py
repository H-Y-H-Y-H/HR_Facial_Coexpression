from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import random
import math
random.seed(0)

def plot_eval(rand_cmds, result_cmds, label_cmds):
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))

    all_dist = [[] for _ in range(11)]
    for i in range(len(result_cmds)):
        for j in range(11):
            all_dist[j].append(abs(result_cmds[i][j] - label_cmds[i][j])) 
    
    mean_dist = []
    std_dist = []
    for i in range(11):
        mean_dist.append(np.mean(all_dist[i]))
        std_dist.append(np.std(all_dist[i])) 
    print('mean: ', mean_dist)
    print('std: ', std_dist)

    rand_dist = [[] for _ in range(11)]

    for i in range(len(rand_cmds)):
        for j in range(11):
            rand_dist[j].append(abs(rand_cmds[i][j] - label_cmds[i][j])) 
    
    rand_mean_dist = []
    rand_std_dist = []
    for i in range(11):
        rand_mean_dist.append(np.mean(rand_dist[i]))
        rand_std_dist.append(np.std(rand_dist[i])) 
    print('rand_mean: ', rand_mean_dist)
    print('rand_std: ', rand_std_dist)

    x = [x for x in range(11)]
    x_label = ['eye0', 'eye1', 'eye2', 'eye3', 'mouth0', 'mouth1', 'mouth2', 'mouth3', 'mouth4', 'mouth5', 'jaw']
    ax[0].plot(x, mean_dist)
    ax[0].errorbar(x, mean_dist, yerr=std_dist, fmt='s')
    ax[0].set_title("Predict model -> Imap -> cmds"+" (mean: {:.4f},  std: {:.4f})".format(np.mean(mean_dist), np.mean(std_dist)),size=10)

    ax[1].plot(x, rand_mean_dist)
    ax[1].errorbar(x, rand_mean_dist, yerr=std_dist, fmt='s')
    ax[1].set_title("Random cmds"+" (mean: {:.4f},  std: {:.4f})".format(np.mean(rand_mean_dist), np.mean(rand_std_dist)),size=10)

    for axi in ax.flat:
        axi.set_ylim(-0.1, 0.8)
        axi.set_xticks(x, x_label)
        axi.tick_params(axis='y', labelsize=8)
    fig.suptitle('Commands loss evaluation')
    plt.show()

def plot_each_lmks(dist_list1, dist_list2, dist_list3):
    x = [i for i in range(113)]
    y = [i*5 for i in range(23)]
    fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(12, 8))
    # hei = 0.8 # , height=hei
    axes[0].barh(x[:52], dist_list1[:52], align='center')
    axes[0].barh(x[52:], dist_list1[52:], align='center')
    axes[0].set(title='Random baseline')

    axes[1].barh(x[:52], dist_list2[:52], align='center')
    axes[1].barh(x[52:], dist_list2[52:], align='center')
    axes[1].set(title='Predict model')
    
    axes[2].barh(x[:52], dist_list3[:52], align='center', label="Eyes lmks")
    axes[2].barh(x[52:], dist_list3[52:], align='center', label="Mouth lmks")
    axes[2].set(title='I-map')
    axes[2].legend()

    for ax in axes.flat:
        ax.margins(0.03)
        # ax.grid(color='grey', linestyle='--')
        ax.set_xlim([0,0.5])

    # axes[0].invert_xaxis()
    axes[0].set(yticks=y, yticklabels=y)
    axes[0].set_xlabel("Landmark distance (robot matric face)")
    axes[0].set_ylabel("Landmark index")
    # axes[0].yaxis.tick_right()

    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.15)
    fig.suptitle("Average Loss on each Landmarks")
    plt.show()

def plot_avg_lmks(dist_list1, dist_list2, dist_list3):
    mean1, std1 = np.mean(dist_list1), np.std(dist_list1)
    mean2, std2 = np.mean(dist_list2), np.std(dist_list2)
    mean3, std3 = np.mean(dist_list3), np.std(dist_list3)

    x = [1,2,3]
    x_label = ["Random baseline", "Predict model", "I-map"]
    fig, ax = plt.subplots()
    ax.bar(x, [mean1, mean2, mean3])
    half = 0.3
    ax.errorbar(x, [mean1, mean2, mean3], yerr=[std1, std2, std3], fmt=" ", capsize=10, capthick=2, elinewidth=2, color="orange")

    ax.text(x[0]-half, mean1 + std1 + 0.03, "Mean: {:.4f}".format(mean1))
    ax.text(x[0]-half, mean1 + std1 + 0.01, "Std: {:.4f}".format(std1))

    ax.text(x[1]-half, mean2 + std2 + 0.03, "Mean: {:.4f}".format(mean2))
    ax.text(x[1]-half, mean2 + std2 + 0.01, "Std: {:.4f}".format(std2))

    ax.text(x[2]-half, mean3 + std3 + 0.03, "Mean: {:.4f}".format(mean3))
    ax.text(x[2]-half, mean3 + std3 + 0.01, "Std: {:.4f}".format(std3))

    ax.set_ylim([0,0.5])
    ax.set_xticks(x, x_label, fontsize=8)
    plt.title("I-map lmks evaluation, compare with human labels (normalized)")
    plt.show()

def pred_cmds_eval():
    """
    cmds evaluation
    """
    label_cmds = np.loadtxt("label_cmds.csv")
    test_result_lmks_cmds = np.loadtxt("test_result_lmks_cmds.csv")
    random_cmds = []
    for i in range(len(label_cmds)):
        random_cmds.append(np.random.rand(11))

    print(label_cmds.shape, test_result_lmks_cmds.shape)
    plot_eval(random_cmds, test_result_lmks_cmds, label_cmds)


def pred_lmks_eval():
    """
    pred lmks evaluation
    """
    # label_lmks = np.load("lmks/simple_label_reorder.npy")
    label_lmks = np.load("Imap_eval_data/simple_imap_reorder.npy")
    result_lmks = np.load("lmks/simple_result_reorder.npy")
    input_lmks = np.load("lmks/simple_input_reorder.npy")

    training_data = np.load("../../public/lmks_data/data_aug_lmks.npy")

    start_id = random.randint(0,len(training_data)-len(result_lmks)-1)

    random_lmks = training_data[start_id:start_id+len(result_lmks)]
    print(label_lmks.shape, result_lmks.shape, random_lmks.shape)

    dist_list_rand, dist_list_input, dist_list_pred = make_dist_data(random_lmks, input_lmks, result_lmks, label_lmks)
    print(np.max(dist_list_rand), np.min(dist_list_rand))
    print(np.max(dist_list_input), np.min(dist_list_input))
    print(np.max(dist_list_pred), np.min(dist_list_pred))
    with plt.style.context('seaborn'):
        plot_each_lmks(dist_list_rand, dist_list_input, dist_list_pred)
        plot_avg_lmks(dist_list_rand, dist_list_input, dist_list_pred)

def make_dist_data(lmks1, lmks2, lmks3, label_lmks):
    dist_list1 = [0] * 113
    dist_list2 = [0] * 113
    dist_list3 = [0] * 113

    for i in range(len(label_lmks)):
        for j in range(113):
            dist_list1[j] += math.sqrt((lmks1[i, 0, j]-label_lmks[i, 0, j])**2 + (lmks1[i, 1, j]-label_lmks[i, 1, j])**2) # sqrt(x^2+y^2)
            dist_list2[j] += math.sqrt((lmks2[i, 0, j]-label_lmks[i, 0, j])**2 + (lmks2[i, 1, j]-label_lmks[i, 1, j])**2)
            dist_list3[j] += math.sqrt((lmks3[i, 0, j]-label_lmks[i, 0, j])**2 + (lmks3[i, 1, j]-label_lmks[i, 1, j])**2)

    dist_list1 = np.array(dist_list1) / len(label_lmks)
    dist_list2 = np.array(dist_list2) / len(label_lmks)
    dist_list3 = np.array(dist_list3) / len(label_lmks)
    return dist_list1, dist_list2, dist_list3


def nearst_neighbour(h_label, neighbours):
    nearst_n = []
    for h in h_label:
        n_dist_list = []
        for n in neighbours:
            dist = 0
            for j in range(113):
                dist += math.sqrt((n[0, j]-h[0, j])**2 + (n[1, j]-h[1, j])**2)
            n_dist_list.append(dist)
        min_id = np.argmin(np.array(n_dist_list))
        nearst_n.append(neighbours[min_id])

    return np.array(nearst_n)


def imap_lmks_eval():
    """
    imap lmks eval
    """
    h_label_data = np.load("Imap_eval_data/simple_h_label_reorder.npy")
    imap_data = np.load("Imap_eval_data/simple_imap_reorder.npy")
    pred_data = np.load("lmks/simple_result_reorder.npy")

    training_data = np.load("../../public/lmks_data/data_aug_lmks.npy")
    neighbour_size = 500
    rand_idx = np.random.randint(len(training_data), size=len(h_label_data))
    rand_idx_nn = np.random.randint(len(training_data), size=neighbour_size)
    random_data = training_data[rand_idx, :, :]
    neighbours = training_data[rand_idx_nn, :, :]

    nearst_n = nearst_neighbour(h_label_data, neighbours)

    print(h_label_data.shape, imap_data.shape, random_data.shape, nearst_n.shape)

    dist_list_rand, dist_list_nn, dist_list_imap = make_dist_data(random_data, nearst_n, imap_data, h_label_data)
    print(np.min(dist_list_rand), np.max(dist_list_rand))
    print(np.min(dist_list_nn), np.max(dist_list_nn))
    print(np.min(dist_list_imap), np.max(dist_list_imap))

    with plt.style.context('seaborn'):
        plot_avg_lmks(dist_list_rand, dist_list_nn, dist_list_imap)
        plot_each_lmks(dist_list_rand, dist_list_nn, dist_list_imap)


if __name__ == "__main__":
    """
    EVALUATION:

    1. commands evaluation for inverse map:
    Run test_model.inverse_map_cmds_eval() in public folder

    2. commands evaluation for predictive model:
    Run pred_cmds_eval()

    3. landmarks evaluation for predictive model:
    Run pred_lmks_eval()

    4. landmarks evaluation for inverse map:
    Run imap_lmks_eval()
    """

    pred_cmds_eval()

    # pred_lmks_eval()

    # imap_lmks_eval()





