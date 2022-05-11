import open3d as o3d
import numpy as np
import struct



def main():
    f.open('pcdlist.txt', 'r')
    pcdlist = f.reandline()
    
    pose_file.open('data/pose/00.txt','r')
    poselist = pose_file.readline()

    for i in range(len(poselist)):
        post = poselist[i]
        q = mat(pos)

    for pcd_name in pcdlist:
        pcd = o3d.io.read_point_cloud(pcd_name)


if __name__ == '__main__':
    main()