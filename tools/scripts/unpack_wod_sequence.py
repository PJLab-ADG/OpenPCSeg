'''
Unpacker code for a single sequence of Waymo Open Dataset
'''


import argparse
from tqdm import tqdm
import numpy as np
import os
import concurrent.futures as futures
import tensorflow as tf
from waymo_open_dataset.dataset_pb2 import Frame
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2
from pathlib import Path
from datetime import datetime
from math import sqrt
from scipy.spatial.transform import Rotation as R
import json
import pickle
import cv2

from pcseg.utils.waymo_utils import generate_labels, convert_range_image_to_point_cloud


def parse_config():
    parser = argparse.ArgumentParser(
        description='The unpack tool for Waymo Open Dataset')
    parser.add_argument(
        '--segment_path',
        type=str,
        required=True,
        help='The path of a tfrecord file')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='The path of output directory')
    parser.add_argument(
        '--num_worker',
        type=int,
        default=1,
        help='Parallelism')
    parser.add_argument(
        '--limit',
        type=int,
        default=100000,
        help='frame limit')
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='for development usage')
    args = parser.parse_args()
    return args


def process_single_frame(workload):
    frame_id, args, frame = workload

    # Labels
    pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
    annos = generate_labels(frame, pose)

    # process lidar data
    range_images, camera_projections, range_image_top_pose = \
        frame_utils.parse_range_image_and_camera_projection(frame)
    points, cp_points, points_in_NLZ_flag, points_intensity, points_elogation \
        = convert_range_image_to_point_cloud(
            frame, range_images, camera_projections,
            range_image_top_pose, ri_index=(0, 1))

    points, cp_points, points_in_NLZ_flag, points_intensity, points_elogation \
        = points[0], cp_points[0], points_in_NLZ_flag[0], points_intensity[0], points_elogation[0]
    vehicle_points = points
    points_in_NLZ_flag = points_in_NLZ_flag.reshape(-1, 1)
    points_intensity = np.tanh(points_intensity).reshape(-1, 1)
    points_elogation = np.tanh(points_elogation).reshape(-1, 1)
    concatenated_vehicle_points = np.concatenate(
        (vehicle_points, points_intensity, points_elogation), axis=1)
    concatenated_vehicle_points = \
        concatenated_vehicle_points[points_in_NLZ_flag.reshape(-1) == -1]
    
    # save points to file
    lidar_file_path = args.output_path_dict['lidar'] / f'{frame_id:010}.npy'
    np.save(str(lidar_file_path), concatenated_vehicle_points.astype(np.float32))

    return {
        "frame_id": frame_id,
    }


def main():
    args = parse_config()
    if os.path.basename(args.segment_path).split('.')[-1] != 'tfrecord':
        raise ValueError(f'segment has to be of tfrecord file')

    # prepare dirs
    args.output_path_dict = {
        'lidar': Path() / args.output_dir / 'LiDAR',
        'segment_meta': Path() / args.output_dir / 'segment_meta.json'
    }

    # mkdirs
    args.output_path_dict['lidar'].mkdir(parents=True, exist_ok=True)

    dataset = tf.data.TFRecordDataset(args.segment_path, compression_type='')
    frame_list = []
    for idx, data in enumerate(dataset):
        if args.limit > 0 and idx >= args.limit:
            break
        frame = Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frame_list.append(frame)
        if args.debug:
            if idx >= 9:
                break

    # process
    input_frame_list = [(idx, args, frame)
                        for idx, frame in enumerate(frame_list)]
    # multiprocessing?
    if args.num_worker == 1:
        result_list = list(
            tqdm(map(process_single_frame, input_frame_list), total=len(frame_list)))
    else:
        with futures.ThreadPoolExecutor(args.num_worker) as executor:
            result_list = list(
                tqdm(
                    executor.map(
                        process_single_frame,
                        input_frame_list),
                    total=len(frame_list)))

    segmenta_meta = {
        'segment_path': args.segment_path,
        'segment_basename': os.path.basename(args.segment_path),
        'frame_meta_list': result_list
    }

    if args.debug:
        import pdb
        pdb.set_trace()

    json.dump(
        segmenta_meta,
        open(
            args.output_path_dict['segment_meta'],
            'w'),
        indent=True)


if __name__ == '__main__':
    main()