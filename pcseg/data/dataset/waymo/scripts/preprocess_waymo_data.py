'''
# In your python 3.6 environment:
rm -rf waymo-od > /dev/null
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
cd waymo-od && git branch -a
git checkout remotes/origin/master
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-6-0==1.4.3
'''


import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
import pickle

tf.enable_eager_execution()
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import os


def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      segmentation_labels: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


def get_point_cloud_with_label(frame, range_images, camera_projections, segmentation_labels, range_image_top_pose):
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, keep_polar_features=True)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=1, keep_polar_features=True)

    point_labels = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels)
    point_labels_ri2 = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels, ri_index=1)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)

    # point labels.
    point_labels_all = np.concatenate(point_labels, axis=0)
    point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)

    # labels and points
    assert len(point_labels_all_ri2) == len(points_all_ri2)
    assert len(point_labels_all) == len(points_all)

    return point_labels_all[:, 1], points_all[:point_labels_all.shape[0], :], point_labels_all_ri2[:,
                                                                              1], points_all_ri2[
                                                                                  :point_labels_all_ri2.shape[0], :]


def get_point_cloud_without_label(frame, range_images, camera_projections, segmentation_labels, range_image_top_pose):
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, keep_polar_features=True)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=1, keep_polar_features=True)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)

    # points
    return points_all, points_all_ri2


if __name__ == "__main__":
    waymo_root = './data_root/Waymo/raw_data/'  # raw_data in ['training', 'validation']
    if 'training' in waymo_root:
        save_dir = 'train'
    elif 'validation' in waymo_root:
        save_dir = 'val_with_label'
    for sub_root in os.listdir(waymo_root):
        if not sub_root.endswith(".tar"):  # ignore compressed waymo file
            root = os.path.join(waymo_root, sub_root)
            root_name = root.split('/')[-1]
            for sub_name in os.listdir(root):
                FILENAME = os.path.join(root, sub_name)
                if not sub_name == "LICENSE":
                    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
                    for frame_id, data in enumerate(dataset):
                        frame = open_dataset.Frame()
                        frame.ParseFromString(bytearray(data.numpy()))

                        if frame.lasers[0].ri_return1.segmentation_label_compressed:
                            npy_file_root_f = os.path.join(waymo_root, save_dir, 'first', root_name)
                            if not os.path.exists(npy_file_root_f):
                                os.makedirs(npy_file_root_f)
                            npy_file_root_s = os.path.join(waymo_root, save_dir, 'second', root_name)
                            if not os.path.exists(npy_file_root_s):
                                os.makedirs(npy_file_root_s)

                            npy_file_f = npy_file_root_f + '/' + sub_name[:-9] + "_" + str(frame_id) + '.npy'
                            npy_file_s = npy_file_root_s + '/' + sub_name[:-9] + "_" + str(frame_id) + '.npy'
                            (range_images, camera_projections, segmentation_labels, range_image_top_pose) = \
                                frame_utils.parse_range_image_and_camera_projection(frame)
                            points_labels_f, points_f, points_labels_s, points_s = get_point_cloud_with_label(frame,
                                                                                                              range_images,
                                                                                                              camera_projections,
                                                                                                              segmentation_labels,
                                                                                                              range_image_top_pose)
                            points_labels_f = points_labels_f.reshape(-1, 1)
                            points_labels_s = points_labels_s.reshape(-1, 1)

                            seg_points_f = np.concatenate((points_f, points_labels_f), axis=1)
                            assert seg_points_f.shape[1] == 7  # range, intensity, elogation, x, y,z ,label

                            seg_points_s = np.concatenate((points_s, points_labels_s), axis=1)
                            assert seg_points_s.shape[1] == 7
                            np.save(npy_file_f, seg_points_f)
                            np.save(npy_file_s, seg_points_s)
                            print(npy_file_f)
                            print(npy_file_s)
                            print('-' * 80)
