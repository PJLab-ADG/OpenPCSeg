'''
Inspired by https://arxiv.org/abs/2207.00026

LaserMix for Semi-Supervised LiDAR Semantic Segmentation
'''


import numpy as np


def lasermix_aug(xyzi_sup, label_sup, xyzi_unsup, label_unsup):
    'Generates one sample of data'   # for semantic-kitti
    # Mix sup and unsup for MixTeacher training
    global xyzil_mix_1
    xyzil_sup = np.concatenate((xyzi_sup,  label_sup), axis=1)  # [N, 5]
    xyzil_unsup = np.concatenate((xyzi_unsup,  label_unsup), axis=1)  # [N, 5]
    rho_sup = np.sqrt(xyzil_sup[:, 0] ** 2 + xyzil_sup[:, 1] ** 2)  # rho = (x^2 + y^2)^(1/2)
    rho_unsup = np.sqrt(xyzil_unsup[:, 0] ** 2 + xyzil_unsup[:, 1] ** 2)  # rho = (x^2 + y^2)^(1/2)

    phi_sup = np.arctan2(xyzil_sup[:, 1], xyzil_sup[:, 0])  # phi = arctan(y/x)
    phi_unsup = np.arctan2(xyzil_unsup[:, 1], xyzil_unsup[:, 0])  # phi = arctan(y/x)

    inc_sup = np.arctan2(xyzil_sup[:, 2], rho_sup)  # inc = arctan(z / (x^2 + y^2)^(1/2))
    inc_unsup = np.arctan2(xyzil_unsup[:, 2], rho_unsup)  # inc = arctan(z / (x^2 + y^2)^(1/2))

    strategy = 'mixture'

    if strategy == 'mixture':
        # strategies = ['rho3phi1', 'rho3phi2', 'rho4phi1', 'rho4phi2', 'rho5phi1', 'rho5phi2', 'rho6phi1', 'rho6phi2',]
        # strategies = ['inc3phi1', 'inc3phi2', 'inc4phi1', 'inc4phi2', 'inc5phi1', 'inc5phi2', 'inc6phi1', 'inc6phi2']
        strategies = ['inc3phi1', 'inc4phi1', 'inc5phi1', 'inc6phi1']
        strategy = np.random.choice(strategies, size=1)[0]

    if strategy == 'inc3phi1':

        xyzil_sup_p1 = xyzil_sup[inc_sup > -6.7 / np.pi * 180]
        xyzil_sup_p2 = xyzil_sup[np.logical_and(inc_sup <= -6.7 / np.pi * 180, inc_sup > -13.4 / np.pi * 180)]
        xyzil_sup_p3 = xyzil_sup[inc_sup <= -13.4 / np.pi * 180]

        xyzil_unsup_p1 = xyzil_unsup[inc_unsup > -6.7 / np.pi * 180]
        xyzil_unsup_p2 = xyzil_unsup[np.logical_and(inc_unsup <= -6.7 / np.pi * 180, inc_unsup > -13.4 / np.pi * 180)]
        xyzil_unsup_p3 = xyzil_unsup[inc_unsup <= -13.4 / np.pi * 180]

        # Mix sup and unsup in xyz coordinate
        xyzil_mix_1 = np.concatenate((xyzil_sup_p1, xyzil_unsup_p2, xyzil_sup_p3), axis=0)  # [N, 5]
        xyzil_mix_2 = np.concatenate((xyzil_unsup_p1, xyzil_sup_p2, xyzil_unsup_p3), axis=0)  # [N, 5]

    elif strategy == 'inc4phi1':

        xyzil_sup_p1 = xyzil_sup[inc_sup > -5.0 / np.pi * 180]
        xyzil_sup_p2 = xyzil_sup[np.logical_and(inc_sup <= -5.0 / np.pi * 180, inc_sup > -10.0 / np.pi * 180)]
        xyzil_sup_p3 = xyzil_sup[np.logical_and(inc_sup <= -10.0 / np.pi * 180, inc_sup > -15.0 / np.pi * 180)]
        xyzil_sup_p4 = xyzil_sup[inc_sup <= -15.0 / np.pi * 180]

        xyzil_unsup_p1 = xyzil_unsup[inc_unsup > -5.0 / np.pi * 180]
        xyzil_unsup_p2 = xyzil_unsup[np.logical_and(inc_unsup <= -5.0 / np.pi * 180, inc_unsup > -10.0 / np.pi * 180)]
        xyzil_unsup_p3 = xyzil_unsup[np.logical_and(inc_unsup <= -10.0 / np.pi * 180, inc_unsup > -15.0 / np.pi * 180)]
        xyzil_unsup_p4 = xyzil_unsup[inc_unsup <= -15.0 / np.pi * 180]

        # Mix sup and unsup in xyz coordinate
        xyzil_mix_1 = np.concatenate((xyzil_sup_p1, xyzil_unsup_p2, xyzil_sup_p3, xyzil_unsup_p4),
                                     axis=0)  # [N, 5]
        xyzil_mix_2 = np.concatenate((xyzil_unsup_p1, xyzil_sup_p2, xyzil_unsup_p3, xyzil_sup_p4),
                                     axis=0)  # [N, 5]

    elif strategy == 'inc5phi1':

        xyzil_sup_p1 = xyzil_sup[inc_sup > -4.0 / np.pi * 180]
        xyzil_sup_p2 = xyzil_sup[np.logical_and(inc_sup <= -4.0 / np.pi * 180, inc_sup > -8.0 / np.pi * 180)]
        xyzil_sup_p3 = xyzil_sup[np.logical_and(inc_sup <= -8.0 / np.pi * 180, inc_sup > -12.0 / np.pi * 180)]
        xyzil_sup_p4 = xyzil_sup[np.logical_and(inc_sup <= -12.0 / np.pi * 180, inc_sup > -16.0 / np.pi * 180)]
        xyzil_sup_p5 = xyzil_sup[inc_sup <= -16.0 / np.pi * 180]

        xyzil_unsup_p1 = xyzil_unsup[inc_unsup > -4.0 / np.pi * 180]
        xyzil_unsup_p2 = xyzil_unsup[np.logical_and(inc_unsup <= -4.0 / np.pi * 180, inc_unsup > -8.0 / np.pi * 180)]
        xyzil_unsup_p3 = xyzil_unsup[np.logical_and(inc_unsup <= -8.0 / np.pi * 180, inc_unsup > -12.0 / np.pi * 180)]
        xyzil_unsup_p4 = xyzil_unsup[np.logical_and(inc_unsup <= -12.0 / np.pi * 180, inc_unsup > -16.0 / np.pi * 180)]
        xyzil_unsup_p5 = xyzil_unsup[inc_unsup <= -16.0 / np.pi * 180]

        # Mix sup and unsup in xyz coordinate
        xyzil_mix_1 = np.concatenate((xyzil_sup_p1, xyzil_unsup_p2, xyzil_sup_p3, xyzil_unsup_p4, xyzil_sup_p5),
                                     axis=0)  # [N, 5]
        xyzil_mix_2 = np.concatenate(
            (xyzil_unsup_p1, xyzil_sup_p2, xyzil_unsup_p3, xyzil_sup_p4, xyzil_unsup_p5), axis=0)  # [N, 5]


    elif strategy == 'inc6phi1':

        xyzil_sup_p1 = xyzil_sup[inc_sup > -3.3 / np.pi * 180]
        xyzil_sup_p2 = xyzil_sup[np.logical_and(inc_sup <= -3.3 / np.pi * 180, inc_sup > -6.6 / np.pi * 180)]
        xyzil_sup_p3 = xyzil_sup[np.logical_and(inc_sup <= -6.6 / np.pi * 180, inc_sup > -9.9 / np.pi * 180)]
        xyzil_sup_p4 = xyzil_sup[np.logical_and(inc_sup <= -9.9 / np.pi * 180, inc_sup > -13.2 / np.pi * 180)]
        xyzil_sup_p5 = xyzil_sup[np.logical_and(inc_sup <= -13.2 / np.pi * 180, inc_sup > -16.5 / np.pi * 180)]
        xyzil_sup_p6 = xyzil_sup[inc_sup <= -16.5 / np.pi * 180]

        xyzil_unsup_p1 = xyzil_unsup[inc_unsup > -3.3 / np.pi * 180]
        xyzil_unsup_p2 = xyzil_unsup[np.logical_and(inc_unsup <= -3.3 / np.pi * 180, inc_unsup > -6.6 / np.pi * 180)]
        xyzil_unsup_p3 = xyzil_unsup[np.logical_and(inc_unsup <= -6.6 / np.pi * 180, inc_unsup > -9.9 / np.pi * 180)]
        xyzil_unsup_p4 = xyzil_unsup[np.logical_and(inc_unsup <= -9.9 / np.pi * 180, inc_unsup > -13.2 / np.pi * 180)]
        xyzil_unsup_p5 = xyzil_unsup[np.logical_and(inc_unsup <= -13.2 / np.pi * 180, inc_unsup > -16.5 / np.pi * 180)]
        xyzil_unsup_p6 = xyzil_unsup[inc_unsup <= -16.5 / np.pi * 180]

        # Mix sup and unsup in xyz coordinate
        xyzil_mix_1 = np.concatenate(
            (xyzil_sup_p1, xyzil_unsup_p2, xyzil_sup_p3, xyzil_unsup_p4, xyzil_sup_p5, xyzil_unsup_p6),
            axis=0)  # [N, 5]
        xyzil_mix_2 = np.concatenate(
            (xyzil_unsup_p1, xyzil_sup_p2, xyzil_unsup_p3, xyzil_sup_p4, xyzil_unsup_p5, xyzil_sup_p6),
            axis=0)  # [N, 5]

    Xyzi = xyzil_mix_1[:, :-1]
    Label = xyzil_mix_1[:, -1].reshape((-1, 1))
    return Xyzi, Label
