import cv2
import open3d as o3d
import numpy as np
import struct
from association.ransac_icp import RANSACParams, CheckerParams, ransac_match, exact_match

def get_q_matrix():
    # https://github.com/ros-perception/vision_opencv/blob/noetic/image_geometry/src/stereo_camera_model.cpp#L86

    # left_fx = left_camera_info.P[0]
    # left_fy = left_camera_info.P[5]
    # left_cx = left_camera_info.P[2]
    # left_cy = left_camera_info.P[6]
    # right_cx = right_camera_info.P[2]
    # base_line = -right_camera_info.P[3] / right_camera_info.P[0]

    # left_fx = 721.5377
    # left_fy = 721.5377
    # left_cx = 609.5593
    # left_cy = 172.854
    # right_cx = 609.5593
    # base_line = 339.5242 / 721.5377

    left_fx = 718.856
    left_fy = 718.856
    left_cx = 607.1928
    left_cy = 185.2157
    right_cx = 607.1928
    base_line = 337.2877 / 718.856

    Tx = -base_line
    Q = np.zeros((4, 4), dtype=np.float32)
    Q[0, 0] = left_fy * Tx
    Q[0, 3] = -left_fy * left_cx * Tx
    Q[1, 1] = left_fx * Tx
    Q[1, 3] = -left_fx * left_cy * Tx
    Q[2, 3] = left_fx * left_fy * Tx
    Q[3, 2] = -left_fy
    Q[3, 3] = left_fy * (left_cx - right_cx)

    return Q

def getpose(num):
    poses = []
    f = open('data/pose/00.txt', 'r')
    for i in range(num):
        pose_string = f.readline()
        pose_string = pose_string[:-1].split(' ')
        pose_tmp = np.array(pose_string)

        # y_roi = [
        #     [0,0,1,0], 
        #     [0,1,0,0], 
        #     [-1,0,0,0], 
        #     [0,0,0,1],
        # ]
        # y_roi = np.array(y_roi)

        pose = [
            [eval(pose_tmp[0]),eval(pose_tmp[1]),eval(pose_tmp[2]),eval(pose_tmp[3])], 
            [eval(pose_tmp[4]),eval(pose_tmp[5]),eval(pose_tmp[6]),eval(pose_tmp[7])],
            [eval(pose_tmp[8]),eval(pose_tmp[9]),eval(pose_tmp[10]),eval(pose_tmp[11])],
            [0,0,0,1]
        ]
        # pose = np.array(pose)

        # pose_t = np.dot(y_roi, pose)
        poses.append(pose)
    return poses

def xyzrgb_array_to_pointcloud2(projected_points, colors, max_depth, header=None):
    points_3d = []

    td_skip = 20
    lr_skip = 100

    if len(colors.shape) == 2:
        colors = np.dstack([colors, colors, colors])

    height, width = colors.shape[:2]
    for u in range(td_skip, height - td_skip):
        img_row = colors[u]
        projected_row = projected_points[u]
        for v in range(lr_skip, width - lr_skip):
            cur_projected_point = projected_row[v]
            x = cur_projected_point[2]
            y = -cur_projected_point[0]
            z = -cur_projected_point[1]

            if x < 0 or x > max_depth:
                continue

            cur_color = img_row[v]
            b = cur_color[0]
            g = cur_color[1]
            r = cur_color[2]

            float_formatter = lambda x: "%.4f" % x
            points_3d.append("{} {} {} {} {} {} 0\n".format
                (float_formatter(x), float_formatter(y), float_formatter(z),
                int(r), int(g), int(b)))
            # pt = [x, y, z, r, g, b]
            # points_3d.append(pt)

    return points_3d

def wirte_pcd(save_file, points):
    file = open(save_file , "w")
    file.write('''# .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z intensity
    SIZE 4 4 4 4
    TYPE F F F F
    COUNT 1 1 1 1
    WIDTH %d
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS %d
    DATA ascii
    %s
    ''' % (len(points),len(points), "".join(points)))
    file.close()

    print("Write into .pcd file Done.")

def write_ply(save_file, points):

    file = open(save_file , "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(points)))
    file.close()

    print("Write into .ply file Done.")

def show_point_cloud(pc_file):
    pcd = o3d.io.read_point_cloud(pc_file)
    o3d.visualization.draw_geometries([pcd])

def down_sample(pc_file):
    pcd = o3d.io.read_point_cloud(pc_file)
    # pcd_array = np.array([pcd])
    # print(pcd_array)
    pcd, idx_inliers = pcd.remove_radius_outlier(nb_points=10, radius=0.1)
    # pcd_array = np.array([pcd])
    # print(pcd_array)
    # print(pcd.size())
    # o3d.visualization.draw_geometries([pcd])
    # write_ply(pc_file[:-4] + '_down. ply', pcd)
    print(pc_file[:-4] + '_down.ply')
    pcd = o3d.io.write_point_cloud(pc_file[:-4] + '_down' + '.ply', pcd)

def main():

    f = open('list.txt', 'r')
    filelist = f.readlines()
    for index in filelist:
        save_file = index[:-5] + '.ply'
        pcd_file = index[:-5]  + '.pcd'
        _q_matrix = get_q_matrix()
        max_depth = 10

        color_img = cv2.imread('data/left/' + index[:-1] ,cv2.IMREAD_COLOR)

        # 这里读取的是*256的uint16,要先进行还原操作
        disp_img = cv2.imread('data/pred/'  + index[:-1], -1)
        disp_img = disp_img / 256

        # print(disp_img)

        depth = cv2.split(disp_img)[0]

        new_depth = depth.astype(np.float32)

        # print(disp_img)

        projected_points = cv2.reprojectImageTo3D(new_depth, _q_matrix)

        print(projected_points)

        pointcloud =  xyzrgb_array_to_pointcloud2(projected_points, color_img, max_depth )

        write_ply('data/pcd/' + save_file, pointcloud)
        wirte_pcd('data/pcd/' +pcd_file, pointcloud)

        down_sample('data/pcd/' + save_file)

        # show_point_cloud(save_file)

def test():
    radius = 0.5

    # load the map pcd and stereo pcd , map as the target and the stereo as the source
    map_pcd = o3d.io.read_point_cloud('data/pcd/kittimap.pcd')
    map_pcd_dense = map_pcd
    map_pcd  = map_pcd.voxel_down_sample(
        voxel_size = 0.3
    )
    map_pcd.estimate_normals(
         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
    )

    stereo_pcd = o3d.io.read_point_cloud('data/pcd/0000000000.ply')
    stereo_pcd_dense = stereo_pcd
    stereo_pcd  = stereo_pcd.voxel_down_sample(
        voxel_size = 0.3
    )
    stereo_pcd.estimate_normals(
         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
    )

    # build search trees:  创建kdtree 这里是想要将source 匹配到 target上面
    map_pcd, idx_inliers = map_pcd.remove_radius_outlier(nb_points=6, radius=radius)
    search_tree_map = o3d.geometry.KDTreeFlann(map_pcd)

    stereo_pcd, idx_inliers = stereo_pcd.remove_radius_outlier(nb_points=6, radius=radius)
    search_tree_stereo= o3d.geometry.KDTreeFlann(stereo_pcd)

    # o3d.visualization.draw_geometries([map_pcd, stereo_pcd])
     

    init_t = np.zeros((4, 4), dtype=np.float32)
    init_t[0,0] = 1
    init_t[1,1] = 1
    init_t[2,2] = 1
    init_t[0,3] = 2
    init_t[3,3] = 1

    # stereo_pcd.transform(init_t)

    # o3d.visualization.draw_geometries([map_pcd, stereo_pcd])

    # # exact ICP for refined estimation:  使用icp进行匹配
    final_result = exact_match(
        stereo_pcd, map_pcd, search_tree_map,
        init_t,
        0.5, 60
    )

    stereo_pcd_dense.transform(final_result.transformation)

    print(final_result.transformation)

    o3d.visualization.draw_geometries([map_pcd_dense, stereo_pcd_dense])

def joinmap():

    points= []

    y_roi = [
        [0,0,-1,0], 
        [0,1,0,0], 
        [1,0,0,0], 
        [0,0,0,1]
    ]

    # 读取文件列表
    f  = open('list.txt', 'r')
    filelist = f.readlines()

    # 获取文件中的位姿信息
    poses = getpose(len(filelist))
    # print(poses)

    for index in range(len(filelist)):
        pcd = o3d.io.read_point_cloud('data/pcd/'+filelist[index][: -5] + '_down' + '.ply')
        pcd.transform(y_roi)
        pcd.transform(poses[index])
        points.append(pcd)

    o3d.visualization.draw_geometries(points)
    




if __name__=='__main__':
    # main()
    # test()
    # down_sample('data/pcd/0000000000.ply')
    # show_point_cloud('data/pcd/0000000000_down.ply')
    # poses = getpose(1)
    # print(poses)
    joinmap()

 

