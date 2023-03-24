import numpy as np
import sys
# insert at 1, 0 is the script path (or '' in REPL)

from lmks_data.simplified_landmark_plus import face_3d_to_2d, mash_to_contour, s_eyes_idx, s_mouth_idx

# lmks data
# R_static_face = np.load("./lmks_data/simple_static.npy")
# robot_edge = np.load("./lmks_data/robot_edge.npy")
R_static_face = np.load("./lmks_data/simple_static.npy") # new data
robot_edge = np.load("./lmks_data/robot_edge.npy")
# ratio_matrix = np.load("./lmks_data/matrix.npy")
ratio_matrix = np.ones((4,113))

# matric lmks to simple lmks
def matric2simple(matric):
    return mash_to_contour(face_3d_to_2d(matric))

# one frame version normalization
def human2robot(H_frame, H_static_face, K_m = 0.9):
    scaled_displace = (H_frame - H_static_face) * K_m
    clipped_displace = np.clip(scaled_displace, -robot_edge[[0, 2]], robot_edge[[1, 3]])
    return clipped_displace + R_static_face

# # one frame version normalization
# def human2robot(H_frame, H_static_face, K_m = 0.9):
#     # normalized_human   = H_frame.copy()
#     d_normalized_human = H_frame.copy()
#
#     dist = np.sum((H_frame - H_static_face)**2)
#     print(dist)
#     for i in range(113):
#         d_x = (H_frame - H_static_face)[0][i]
#         d_y = (H_frame - H_static_face)[1][i]
#
#
#         # normalize x
#         if d_x > 0:
#             if d_x * ratio_matrix[1][i] * K_m > robot_edge[1][i]:
#                 # set to x_max
#                 d_normalized_human[0][i] = robot_edge[1][i]
#             else:
#                 d_normalized_human[0][i] = d_x * ratio_matrix[1][i] * K_m
#         else:
#             if -d_x * ratio_matrix[0][i] * K_m > robot_edge[0][i]:
#                 # set to x_min
#                 d_normalized_human[0][i] = -robot_edge[0][i]
#             else:
#                 d_normalized_human[0][i] = d_x * ratio_matrix[0][i] * K_m
#         # normalize y
#         if d_y > 0:
#             if d_y * ratio_matrix[3][i] * K_m > robot_edge[3][i]:
#                 # set to y_max
#                 d_normalized_human[1][i] = robot_edge[3][i]
#             else:
#                 d_normalized_human[1][i] = d_y * ratio_matrix[3][i] * K_m
#         else:
#             if -d_y * ratio_matrix[2][i] * K_m > robot_edge[2][i]:
#                 # set to y_min
#                 d_normalized_human[1][i] = -robot_edge[2][i]
#             else:
#                 d_normalized_human[1][i] = d_y * ratio_matrix[2][i] * K_m
#
#     normalized_human = d_normalized_human + R_static_face
#
#
#     return normalized_human


# # one frame version normalization
# def human2robot(H_frame, H_static_face, K_m=0.9):
#     normalized_human = H_frame.copy()
#
#     for i in range(113):
#         d_x = (H_frame - H_static_face)[0][i]
#         d_y = (H_frame - H_static_face)[1][i]
#         # normalize x
#         if d_x > 0:
#             if d_x * ratio_matrix[1][i] * K_m > robot_edge[1][i]:
#                 # set to x_max
#                 normalized_human[0][i] = robot_edge[1][i] + R_static_face[0][i]
#             else:
#                 normalized_human[0][i] = d_x * ratio_matrix[1][i] * K_m + R_static_face[0][i]
#         else:
#             if -d_x * ratio_matrix[0][i] * K_m > robot_edge[0][i]:
#                 # set to x_min
#                 normalized_human[0][i] = -robot_edge[0][i] + R_static_face[0][i]
#             else:
#                 normalized_human[0][i] = d_x * ratio_matrix[0][i] * K_m + R_static_face[0][i]
#         # normalize y
#         if d_y > 0:
#             if d_y * ratio_matrix[3][i] * K_m > robot_edge[3][i]:
#                 # set to y_max
#                 normalized_human[1][i] = robot_edge[3][i] + R_static_face[1][i]
#             else:
#                 normalized_human[1][i] = d_y * ratio_matrix[3][i] * K_m + R_static_face[1][i]
#         else:
#             if -d_y * ratio_matrix[2][i] * K_m > robot_edge[2][i]:
#                 # set to y_min
#                 normalized_human[1][i] = -robot_edge[2][i] + R_static_face[1][i]
#             else:
#                 normalized_human[1][i] = d_y * ratio_matrix[2][i] * K_m + R_static_face[1][i]
#
#     return normalized_human


def reorder_lmks(norm1):
    # the data augment process will change the order of the lmks
    # this step is to ensure the input lmks order is the same as training data's lmks order
    normalized_lmks_eyes = norm1[:, s_eyes_idx]
    normalized_lmks_mouth = norm1[:, s_mouth_idx]
    norm2 = np.concatenate((normalized_lmks_eyes, normalized_lmks_mouth), axis=1)
    return norm2


if __name__ == "__main__":
    # print(np.load("./lmks_data/matrix.npy"))
    human_face = np.load('2.npy')
    human_static = np.load('s_1.npy')
    human_static = matric2simple(human_static)
    print(human_face.shape)
    print(human_static.shape)
    human_norm1 = human2robot(human_face[0], human_static)
    # human_norm2 = human2robot_new(human_face[0], human_static)
    # print(human_norm1 == human_norm2)