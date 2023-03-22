'''
This file is copied from https://github.com/xiaoaoran/polarmix
'''


import numpy as np
instance_classes_kitti = [1, 2, 3, 4, 5, 6, 7, 8]

def swap(pt1, pt2, start_angle, end_angle, label1, label2):
    # calculate horizontal angle for each point
    yaw1 = -np.arctan2(pt1[:, 1], pt1[:, 0])
    yaw2 = -np.arctan2(pt2[:, 1], pt2[:, 0])

    # select points in sector
    idx1 = np.where((yaw1>start_angle) & (yaw1<end_angle))
    idx2 = np.where((yaw2>start_angle) & (yaw2<end_angle))

    # swap
    pt1_out = np.delete(pt1, idx1, axis=0)
    pt1_out = np.concatenate((pt1_out, pt2[idx2]))
    pt2_out = np.delete(pt2, idx2, axis=0)
    pt2_out = np.concatenate((pt2_out, pt1[idx1]))

    label1_out = np.delete(label1, idx1)
    label1_out = np.concatenate((label1_out, label2[idx2]))
    label2_out = np.delete(label2, idx2)
    label2_out = np.concatenate((label2_out, label1[idx1]))
    assert pt1_out.shape[0] == label1_out.shape[0]
    assert pt2_out.shape[0] == label2_out.shape[0]

    return pt1_out, pt2_out, label1_out, label2_out

def rotate_copy(pts, labels, instance_classes, Omega):
    # extract instance points
    pts_inst, labels_inst = [], []
    for s_class in instance_classes:
        pt_idx = np.where((labels == s_class))
        pts_inst.append(pts[pt_idx])
        labels_inst.append(labels[pt_idx])
    pts_inst = np.concatenate(pts_inst, axis=0)
    labels_inst = np.concatenate(labels_inst, axis=0)

    # rotate-copy
    pts_copy = [pts_inst]
    labels_copy = [labels_inst]
    for omega_j in Omega:
        rot_mat = np.array([[np.cos(omega_j),
                             np.sin(omega_j), 0],
                            [-np.sin(omega_j),
                             np.cos(omega_j), 0], [0, 0, 1]])
        new_pt = np.zeros_like(pts_inst)
        new_pt[:, :3] = np.dot(pts_inst[:, :3], rot_mat)
        new_pt[:, 3] = pts_inst[:, 3]
        pts_copy.append(new_pt)
        labels_copy.append(labels_inst)
    pts_copy = np.concatenate(pts_copy, axis=0)
    labels_copy = np.concatenate(labels_copy, axis=0)
    return pts_copy, labels_copy

def polarmix(pts1, labels1, pts2, labels2, alpha, beta, instance_classes, Omega):
    pts_out, labels_out = pts1, labels1
    # swapping
    if np.random.random() < 0.5:
        pts_out, _, labels_out, _ = swap(pts1, pts2, start_angle=alpha, end_angle=beta, label1=labels1, label2=labels2)

    # rotate-pasting
    if np.random.random() < 1.0:
        # rotate-copy
        pts_copy, labels_copy = rotate_copy(pts2, labels2, instance_classes, Omega)
        # paste
        pts_out = np.concatenate((pts_out, pts_copy), axis=0)
        labels_out = np.concatenate((labels_out, labels_copy), axis=0)

    return pts_out, labels_out
