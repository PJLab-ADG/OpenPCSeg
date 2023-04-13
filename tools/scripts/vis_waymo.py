import copy
import numpy as np
import argparse
try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')

COLOR_MAP = {
    0: (0, 118, 176),  # noise
    1: (234, 119, 191),  # car
    2: (0, 158, 57),  # truck
    3: (219, 218, 145),  # bus
    4: (196, 174, 210),  # othervehicle
    5: (202.254, 59),  # motorcyclist
    6: (253, 185, 210),  # 　bicylist
    7: (141, 222, 141),   # pedestrain
    8: (0, 185, 200),              # traffic-sign
    9: (0, 255, 104),            # traffic-light
    10: (169, 197, 229),               # pole
    11: (126, 126, 126),                # constrution cone
    12: (220, 32, 40),  # bicyle
    13: (0, 116, 250),  # motorcycle
    14: (135, 76, 65),  # building
    15: (255, 126, 47),  # vegetation
    16: (198, 198, 198),  # tree truck
    17: (255, 151, 150),            # curb
    18: (149, 217, 227),  # road
    19: (198, 152, 145),           # lane marker
    20: (255, 185, 125),  # other ground
    21: (186, 187, 56),  # walkable
    22: (150, 104, 185),  # sidewalk
}


def _draw_points(points,
                 vis,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 mode='xyzrgb'):
    """Draw points on visualizer.
    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    Returns:
        tuple: points, color of each point.
    """
    vis.get_render_option().point_size = points_size  # set points size
    # if isinstance(points, torch.Tensor):
    #     points = points.cpu().numpy()

    points = points.copy()
    pcd = geometry.PointCloud()
    mode = "xyzrgb"
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
        # normalize to [0, 1] for open3d drawing
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
            points_colors /= 255.0
    else:
        raise NotImplementedError

    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.add_geometry(pcd)

    o3d.io.write_point_cloud("eg.pcd", pcd)
    return pcd, points_colors


class Visualizer(object):
    r"""Online visualizer implemented with Open3d.
    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7], optional): 3D bbox
            (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            The 3D bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode).
            Default: None.
        save_path (str, optional): path to save visualized results.
            Default: None.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """

    def __init__(self,
                 points,
                 bbox3d=None,
                 save_path="pic",
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
        super(Visualizer, self).__init__()
        assert 0 <= rot_axis <= 2

        # init visualizer
        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])  # create coordinate frame
        self.o3d_visualizer.add_geometry(mesh_frame)

        self.points_size = points_size
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode
        self.seg_num = 0

        # draw points
        if points is not None:
            self.pcd, self.points_colors = _draw_points(
                points, self.o3d_visualizer, points_size, point_color, mode)

        # draw boxes
        if bbox3d is not None:
            _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors,
                         self.pcd, bbox_color, points_in_box_color, rot_axis,
                         center_mode, mode)

    def add_seg_mask(self, seg_mask_colors):
        """Add segmentation mask to visualizer via per-point colorization.
        Args:
            seg_mask_colors (numpy.array, shape=[N, 6]):
                The segmentation mask whose first 3 dims are point coordinates
                and last 3 dims are converted colors.
        """
        # we can't draw the colors on existing points
        # in case gt and pred mask would overlap
        # instead we set a large offset along x-axis for each seg mask
        self.seg_num += 1
        offset = (np.array(self.pcd.points).max(0)
                  - np.array(self.pcd.points).min(0))[0] * 1.2 * self.seg_num
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[offset, 0, 0])  # create coordinate frame for seg
        self.o3d_visualizer.add_geometry(mesh_frame)
        seg_points = copy.deepcopy(seg_mask_colors)
        seg_points[:, 0] += offset
        _draw_points(
            seg_points, self.o3d_visualizer, self.points_size, mode='xyzrgb')

    def show(self, save_path=None):
        """Visualize the points cloud.
        Args:
            save_path (str, optional): path to save image. Default: None.
        """

        self.o3d_visualizer.run()

        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path)

        self.o3d_visualizer.destroy_window()
        return


def show_rawdata(class_id, pc_path, label_path):
    raw_data = np.load(pc_path)[:, 0:3].astype(np.float32)
    raw_label = np.load(label_path).astype(np.int32)

    colors = []
    for label in raw_label:
        if label == class_id:
            colors.append((255, 0, 0))
            # print("yes")
        else:
            colors.append(COLOR_MAP[label])  # [::-1])
    colors = np.vstack(colors)
    points = raw_data
    color_points = np.concatenate((points, colors), axis=1)
    vis_er = Visualizer(color_points)
    vis_er.show(save_path="pic.ply")


def parse_config():
    parser = argparse.ArgumentParser(
        description='PCSeg visualization script version 0.1')

    # == general configs ==
    parser.add_argument('--pc_path', type=str,
                        help='a frame of unpacked WOD LiDAR data')
    parser.add_argument('--label_path', type=str,
                        help='PCSeg predictions for a frame in WOD')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_config()
    class_id = 100
    show_rawdata(class_id, args.pc_path, args.label_path)
