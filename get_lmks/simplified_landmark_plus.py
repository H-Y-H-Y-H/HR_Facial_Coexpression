import numpy as np
import matplotlib.pyplot as plt
import glob
import re


font = {'family': 'serif',
        'color':  'darkgreen',
        'weight': 'normal',
        'size': 16,
        }


original_contour_idx = [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78,
                        80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148,
                        149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185,
                        191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296,
                        297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362,
                        365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398,
                        400, 402, 405, 409, 415, 454, 466]

original_contour_connections = [(72, 124), (60, 45), (8, 0), (27, 5), (91, 94), (80, 96), (114, 113), (1, 57), (7, 66),
                                (5, 89), (104, 109), (68, 117), (118, 103), (70, 71), (81, 77), (117, 99), (59, 39),
                                (7, 1), (106, 47), (52, 51), (123, 92), (14, 33), (69, 127), (94, 85), (21, 36),
                                (124, 79), (51, 59), (126, 93), (116, 115), (23, 64), (44, 60), (87, 86), (9, 8),
                                (67, 118), (41, 42), (122, 91), (24, 25), (86, 125), (48, 49), (112, 120), (119, 102),
                                (65, 37), (33, 21), (74, 81), (98, 83), (107, 121), (37, 56), (92, 105), (105, 79),
                                (90, 122), (25, 26), (49, 50), (30, 62), (96, 82), (83, 95), (71, 72), (46, 40),
                                (36, 2), (99, 126), (16, 38), (88, 87), (47, 44), (125, 85), (55, 54), (82, 97),
                                (19, 15), (17, 43), (23, 32), (109, 110), (120, 101), (100, 78), (66, 55), (56, 6),
                                (0, 70), (26, 3), (38, 31), (89, 123), (2, 98), (61, 28), (115, 114), (110, 111),
                                (22, 18), (17, 63), (4, 90), (34, 20), (84, 80), (111, 101), (29, 61), (63, 10),
                                (11, 13), (76, 68), (121, 106), (40, 58), (93, 100), (3, 88), (6, 14), (58, 16),
                                (103, 104), (57, 41), (73, 75), (13, 12), (102, 108), (108, 107), (43, 30), (69, 67),
                                (75, 74), (28, 4), (42, 48), (50, 39), (31, 65), (20, 35), (32, 29), (53, 52),
                                (12, 19), (95, 76), (10, 9), (64, 24), (18, 34), (62, 27), (127, 116), (45, 46),
                                (78, 119), (54, 53), (113, 112)]

jaw_idx = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

left_eye_idx = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
right_eye_idx = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155]

left_eyebrow_idx = [293, 295, 296, 300, 334, 336, 276, 282, 283, 285]
right_eyebrow_idx = [65, 66, 70, 105, 107, 46, 52, 53, 55, 63]

contour_idx = sorted(jaw_idx + lips_idx + left_eyebrow_idx + right_eyebrow_idx + left_eye_idx + right_eye_idx)
mouth_idx = sorted(jaw_idx + lips_idx)
eyes_idx = sorted(left_eyebrow_idx + right_eyebrow_idx + left_eye_idx + right_eye_idx)

contour_connections = [(63, 109), (52, 38), (6, 0), (23, 4), (80, 83), (70, 84), (100, 99), (1, 49), (5, 58), (4, 78),
                       (90, 95), (103, 89), (61, 62), (71, 67), (51, 32), (5, 1), (92, 40), (45, 44), (108, 81),
                       (60, 112), (83, 74), (109, 69), (44, 51), (111, 82), (102, 101), (19, 56), (37, 52), (76, 75),
                       (7, 6), (59, 103), (34, 35), (107, 80), (20, 21), (75, 110), (41, 42), (98, 105), (104, 88),
                       (65, 71), (93, 106), (81, 91), (91, 69), (79, 107), (21, 22), (42, 43), (26, 54), (84, 72),
                       (62, 63), (39, 33), (13, 31), (77, 76), (40, 37), (110, 74), (48, 47), (72, 85), (16, 12),
                       (14, 36), (19, 28), (95, 96), (105, 87), (86, 68), (58, 48), (0, 61), (22, 2), (31, 27),
                       (78, 108), (53, 24), (101, 100), (96, 97), (18, 15), (14, 55), (3, 79), (29, 17), (73, 70),
                       (97, 87), (25, 53), (55, 8), (9, 11), (106, 92), (33, 50), (82, 86), (2, 77), (50, 13),
                       (89, 90), (49, 34), (64, 66), (11, 10), (88, 94), (94, 93), (36, 26), (60, 59), (66, 65),
                       (24, 3), (35, 41), (43, 32), (27, 57), (17, 30), (28, 25), (46, 45), (10, 16), (8, 7), (56, 20),
                       (15, 29), (54, 23), (112, 102), (38, 39), (68, 104), (47, 46), (99, 98)]

mesh2contour = {0: 0, 7: 1, 13: 2, 14: 3, 17: 4, 33: 5, 37: 6, 39: 7, 40: 8, 46: 9, 52: 10, 53: 11, 55: 12, 58: 13,
                61: 14, 63: 15, 65: 16, 66: 17, 70: 18, 78: 19, 80: 20, 81: 21, 82: 22, 84: 23, 87: 24, 88: 25, 91: 26,
                93: 27, 95: 28, 105: 29, 107: 30, 132: 31, 133: 32, 136: 33, 144: 34, 145: 35, 146: 36, 148: 37,
                149: 38, 150: 39, 152: 40, 153: 41, 154: 42, 155: 43, 157: 44, 158: 45, 159: 46, 160: 47, 161: 48,
                163: 49, 172: 50, 173: 51, 176: 52, 178: 53, 181: 54, 185: 55, 191: 56, 234: 57, 246: 58, 249: 59,
                263: 60, 267: 61, 269: 62, 270: 63, 276: 64, 282: 65, 283: 66, 285: 67, 288: 68, 291: 69, 293: 70,
                295: 71, 296: 72, 300: 73, 308: 74, 310: 75, 311: 76, 312: 77, 314: 78, 317: 79, 318: 80, 321: 81,
                323: 82, 324: 83, 334: 84, 336: 85, 361: 86, 362: 87, 365: 88, 373: 89, 374: 90, 375: 91, 377: 92,
                378: 93, 379: 94, 380: 95, 381: 96, 382: 97, 384: 98, 385: 99, 386: 100, 387: 101, 388: 102, 390: 103,
                397: 104, 398: 105, 400: 106, 402: 107, 405: 108, 409: 109, 415: 110, 454: 111, 466: 112}

s_eyes_idx = [mesh2contour[idx] for idx in eyes_idx]
s_mouth_idx = [mesh2contour[idx] for idx in mouth_idx]

def face_3d_to_2d(landmark, seq=False):
    """
    Turn the metric 3d face to 2d
    :param landmark: metric 3d landmark
    :param seq: whether input is a sequence
    :return: 2d landmark
    """
    if seq:
        return landmark[:, :2, :]
    return landmark[:2, :]


def mash_to_contour(landmark, seq=False):
    """
    Given face mash landmarks (468 points), reduce to contour only (128 points)
    :param landmark: face mesh landmarks
    :param seq: whether input is a sequence
    :return: contour landmarks
    """
    if seq:
        return landmark[:, :, contour_idx]
    return landmark[:, contour_idx]


def mash_to_contour_half(landmark, seq=False, eyes=False, mouth=False):
    """
    Given face mash landmarks (468 points), reduce to contour only (128 points)
    :param landmark: face mesh landmarks
    :param seq: whether input is a sequence
    :return: contour landmarks
    """
    half_idx = contour_idx
    
    if eyes:
        half_idx = sorted(left_eyebrow_idx + right_eyebrow_idx + left_eye_idx + right_eye_idx)
    if mouth:
        half_idx = sorted(jaw_idx + lips_idx)
    
    if seq:
        return landmark[:, :, half_idx]
    return landmark[:, half_idx]


def plot_2d_face(landmark, title=None, save=None, text = False):
    """
    Plot 2d landmarks
    :param landmark: 2d face landmarks
    :param title: title of the plot
    :param save: image filename if want to save to files
    """
    fig, ax = plt.subplots()
    ax.scatter(landmark[0], landmark[1], s=1)
    if text:
        n = [i for i in range(len(landmark[0]))]
        for j, txt in enumerate(n):
            ax.annotate(txt, (landmark[0][j], landmark[1][j]), fontsize=5)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(title)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    plt.close()


def plot_2d_contour(landmark, title=None, save=None):
    """
    Given 2d contour landmarks, plot 2d landmarks and contour connections
    :param landmark: 2d contour landmarks
    :param title: title of the plot
    :param save: image filename if want to save to files
    """
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(landmark[0], landmark[1], c='tab:blue', zorder=10)
    for c in contour_connections:
        point_1 = landmark[:, c[0]]
        point_2 = landmark[:, c[1]]
        plt.plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], c='tab:orange', zorder=0)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(title)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    plt.close()


# tools
numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def seq_glob(filepath, filetype='/*.npy'):
    file_list = []
    for name in sorted(glob.glob(filepath + filetype), key=numericalSort):
        file_list.append(name)

    return file_list

def plot_2d(lmks_path, contour=True):
    lmks = np.load(lmks_path)
    fig = plt.figure(figsize=(6, 6))
    fig.tight_layout()
    ax = plt.axes()
    for i in range(len(lmks)):
        x = lmks[i][0]
        y = lmks[i][1]
        ax.cla()
        ax.scatter(x,y,s=4)
        if contour:
            for c in contour_connections:
                point_1 = lmks[i][:, c[0]]
                point_2 = lmks[i][:, c[1]]
                plt.plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], c='tab:orange', zorder=0)
        plt.text(-10, 10, "Frame: "+str(i), fontdict=font)
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        plt.pause(0.0001)
    plt.show()
