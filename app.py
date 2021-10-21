import open3d as o3d
import cv2
import plyfile as pf
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Process
from PIL import Image
import os
from plyfile import PlyData, PlyElement
import pcl
import copy
import imageio
import math

import sklearn
# clustring

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py



# -------------------------- Experimenting & Visualizing -------------------------

def visualize(file):
    pcd1 = o3d.io.read_point_cloud(file)
    o3d.visualization.draw_geometries([pcd1])


# Function to view multiple 3D objects simultaneously
def multi_vis(func1, func2):
    p1 = Process(target=func1)
    p1.start()
    p2 = Process(target=func2)
    p2.start()
    p1.join()
    p2.join()


# read ply file and all of its fields
def read_ply_xyzrgb(filename):
    """ read XYZRGB point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 11], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
        vertices[:,6] = plydata['vertex'].data['nx']
        vertices[:,7] = plydata['vertex'].data['ny']
        vertices[:,8] = plydata['vertex'].data['nz']
        vertices[:,9] = plydata['vertex'].data['confidence']
        vertices[:,10] = plydata['vertex'].data['value']
    return vertices


def create_gif_for_clusters():
    general_path = '/Users/yousefesp/PycharmProjects/cloud_project/gif/'
    path_to_save = '/Users/yousefesp/PycharmProjects/cloud_project/gif/'
    all_times = os.listdir(general_path)
    all_times.sort()
    images = []
    for image_path in all_times:
        print(general_path+image_path)
        images.append(imageio.imread(general_path+image_path))
    imageio.mimsave(path_to_save + 'final.gif', images, duration=0.5)


def create_gif_for_satellites():
    general_path = '/Users/yousefesp/PycharmProjects/cloud_project/3views_results/'
    path_to_save = '/Users/yousefesp/PycharmProjects/cloud_project/'
    all_times = os.listdir(general_path)
    all_times.sort()
    for sat_num in range (1,4):
        images = []
        filenames = []
        for time in all_times:
            try:
                all_sat_images = os.listdir(general_path+time)
                for sat_image in all_sat_images:
                    if 'A'+str(sat_num) in sat_image:
                        filenames.append(general_path+time+'/'+sat_image)
            except NotADirectoryError:
                continue
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(path_to_save+'sat' + str(sat_num) + '.gif', images)

# ------------------------- Methods to calculate average height -------------------

def calc_avg_height_img(img):
    """
    Calculates average height - input is .ply
    """
    pcd1 = o3d.io.read_point_cloud(img)
    a = np.asarray(pcd1.points)
    sum1 = 0
    for element in a:
        # element[2] is the z coordinate (height)
        sum1 += element[2]
    # a.shape[0] is the number of elements in the array
    avg_height = sum1 / a.shape[0]
    return avg_height


def avg_height_graph(img_list):
    """
    Plots average height of the whole cloud
    """
    i = 0
    time_axis = []
    height_axis = []
    for img in img_list:
        time = i * 10
        time_axis.append(time)
        avg_h = calc_avg_height_img(img)
        height_axis.append(avg_h)
        i += 1
    return height_axis
    # plt.plot(time_axis, height_axis)
    # plt.xlabel('Time [sec]')
    # plt.ylabel('Average height [m]')
    # plt.show()


def calc_avg_height_arr(arr):
    """
    Calculates average height - input is np array
    """
    sum1 = 0
    for element in arr:
        sum1 += element[2]
    avg_height = sum1 / arr.shape[0]
    return avg_height


def split_pcl_avg_height(img):
    """
    Returns a list including the average height of each piece (9 pieces) for one image
    """
    pcd1 = o3d.io.read_point_cloud(img)
    a = np.array(pcd1.points)
    [x1, x2, x3] = split_to_3(a, 0)
    [x1_y1, x1_y2, x1_y3] = split_to_3(x1, 1)
    [x2_y1, x2_y2, x2_y3] = split_to_3(x2, 1)
    [x3_y1, x3_y2, x3_y3] = split_to_3(x3, 1)
    arr_list = [x1_y1, x1_y2, x1_y3, x2_y1, x2_y2, x2_y3, x3_y1, x3_y2, x3_y3]
    avg_list = []
    for arr in arr_list:
        avg_height = calc_avg_height_arr(arr)
        avg_list.append(avg_height)
    return avg_list



def avg_height_9():
    """
    Returns a matrix containing average heights (9 averages per image for each part) for all images
    """
    general_path = '/Users/yousefesp/PycharmProjects/cloud_project/3views_results/'
    img_list = []
    all_folders = os.listdir(general_path)
    for folder in all_folders:
        if 'time_' in folder:
            img_list.append(general_path+folder+'/scene_final/S.ply')
    height_matrix = np.zeros((9, 25))
    time_axis = []
    m = 0
    for img in img_list:
        list_a = split_pcl_avg_height(img)
        for i in range(9):
            height_matrix[i][m] = list_a[i]
        time = m*10
        time_axis.append(time)
        m += 1
    return height_matrix, time_axis


def split_to_3(a, index):
    """
    Splits array evenly into 3 groups based on x/y values (index = 0/1)
    :param a: array
    :param index: call with index 0 for x, index 1 for y
    :return: returns 3 arrays evenly split
    """
    # argsort() returns the indexes that would sort the array
    # a[:, 0].argsort() returns indexes that would sort the array based on the first element
    # a[said indexes] returns the whole cells (all elements) sorted based on the first element
    arr = a[a[:, index].argsort()]
    group1_lim = int(1 * a.shape[0] / 3)
    group2_lim = int(2 * a.shape[0] / 3)
    group3_lim = int(3 * a.shape[0] / 3)
    # places arr[0] to arr[group1_lim] in arr1
    arr1 = arr[0:group1_lim]
    arr2 = arr[group1_lim:group2_lim]
    arr3 = arr[group2_lim:group3_lim]
    return arr1, arr2, arr3


def graphs_9():
    """
    Plots the 9 graphs
    """
    [height_mat, time] = avg_height_9()
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            axs[2-j, i].plot(time, height_mat[i*3+j][:])
    plt.show()
    # draw each part separately
    # for i in range(9):
    #     plt.plot(time, height_mat[i][:])
    #     plt.show()

def modify_ply_top_heights():
    """
    takes a ply file and levels down anything below a specific height threshold
    """
    file_path = '/Users/yousefesp/PycharmProjects/cloud_project/keep_center.ply'
    plydata = PlyData.read(file_path)
    max_height = 0
    for point in plydata.elements[0].data:
        if point['z']>max_height:
            max_height=point['z']
    height_threshold = 0.8 * max_height
    for point in plydata.elements[0].data:
        if point['z']<height_threshold:
            point['z']=0
    el = PlyElement.describe(plydata['vertex'].data, 'vertex')
    PlyData([el], text=True).write('/Users/yousefesp/PycharmProjects/cloud_project/top_heights.ply')


def catch_biggest_cloud():
    file_path = '/Users/yousefesp/PycharmProjects/cloud_project/keep_center.ply'
    plydata = PlyData.read(file_path)
    points_ordered_by_x = plydata['vertex'].data
    points_ordered_by_x.sort()
    search_range = 15000
    threshold_of_y = 0.4
    eps = 0.3
    x=0
    print(type(points_ordered_by_x))
    for i in range(0,len(points_ordered_by_x)-search_range,50):
        range_of_x = points_ordered_by_x[i+search_range]['x']-points_ordered_by_x[i]['x']
        sum_of_y = 0
        count = 0
        for point in points_ordered_by_x[i:i+search_range]:
            if point['z'] > 0:
                sum_of_y += point['y']
        avg = sum_of_y/search_range
        for point in points_ordered_by_x[i:i + search_range]:
            if avg*(1-eps) <= point['y'] <= avg*(1+eps) and point['z']>0:
                count += 1
        if x<count/search_range:
            x = count/search_range
            print(x)
        if count/search_range >= threshold_of_y:
            plydata['vertex'].data = points_ordered_by_x[i:i+search_range]
            el = PlyElement.describe(plydata['vertex'].data, 'vertex')
            PlyData([el], text=True).write('/Users/yousefesp/PycharmProjects/cloud_project/main_cloud.ply')
            return

#------------------------------- Clustering -------------------------------

def create_clusters_from_ply(path, num_of_clusters, centroids):
    ply_path = path+'/S.ply'
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd_as_nparray = np.array(pcd.points)
    model = KMeans(n_clusters=num_of_clusters, init=centroids, max_iter=300, n_init=1, random_state=0)
    clusters_indices = model.fit_predict(pcd_as_nparray)
    plydata = PlyData.read(ply_path)
    for i in range(num_of_clusters):
        color = [i * j % 250 for j in [124, 213, 56]]
        cluster_points = plydata['vertex'].data[clusters_indices==i]
        for point in cluster_points:
            point['red'] = color[0]
            point['green'] = color[1]
            point['blue'] = color[2]
        el = PlyElement.describe(cluster_points, 'vertex')
        PlyData([el], text=True).write(path+'/cluster_'+str(i)+'.ply')

def combine_color_clusters(num_of_clusters):
    general_path = '/Users/yousefesp/PycharmProjects/cloud_project/3views_results/'
    ply_path_list = []
    all_folders = os.listdir(general_path)
    # collect all folders for ply path (not the file just the folder)
    for folder in all_folders:
        if 'time_' in folder:
            ply_path_list.append(general_path + folder + '/scene_final/')
    for ply_path in ply_path_list:
        pcd_combined = o3d.geometry.PointCloud()
        for i in range(num_of_clusters):
            curr_pcd_path = ply_path + 'cluster_' + str(i) + '.ply'
            curr_pcd = o3d.io.read_point_cloud(curr_pcd_path)
            pcd_combined += curr_pcd
        o3d.io.write_point_cloud(ply_path + 'colors.ply', pcd_combined)



def create_all_clusters(num_of_clusters):
    general_path = '/Users/yousefesp/PycharmProjects/cloud_project/3views_results/'
    ply_path_list = []
    all_folders = os.listdir(general_path)
    # collect all folders for ply path (not the file just the folder)
    for folder in all_folders:
        if 'time_' in folder:
            ply_path_list.append(general_path + folder + '/scene_final/')
    # go over all folders and generate clusters from the S.ply file in this folder
    centroids = 'k-means++'
    for ply_path in ply_path_list:
        create_clusters_from_ply(ply_path, num_of_clusters,centroids)
        centroids = []
        for i in range(num_of_clusters):
            centroids.append(get_cluster_center(ply_path+'cluster_'+str(i)+'.ply'))
        centroids = np.array(centroids)


def get_cluster_center(img):
    pcd = o3d.io.read_point_cloud(img)
    pcd_as_nparray = np.asarray(pcd.points)
    sum_x = 0
    sum_y = 0
    sum_z = 0
    for element in pcd_as_nparray:
        sum_x += element[0]
        sum_y += element[1]
        sum_z += element[2]
    x_center = sum_x / pcd_as_nparray.shape[0]
    y_center = sum_y / pcd_as_nparray.shape[0]
    z_center = sum_z / pcd_as_nparray.shape[0]
    return [x_center, y_center, z_center]


def same_cluster_avg_height(idx):
    # idx is the number of cluster we'd like to look at
    general_path = '/Users/yousefesp/PycharmProjects/cloud_project/3views_results/'
    ply_path_list = []
    all_folders = os.listdir(general_path)
    # collect all folders for ply path (not the file just the folder)
    for folder in all_folders:
        if 'time_' in folder:
            ply_path_list.append(general_path + folder + '/scene_final/')
    # sort folders by time
    ply_path_list.sort()
    # collect all ply files of the same cluster based on minimal euclidian distance between centroids
    first_cluster = ply_path_list[0] + '/cluster_' + str(idx) + '.ply'
    same_cluster_paths = []
    same_cluster_paths.append(first_cluster)
    first_cluster_center = get_cluster_center(first_cluster)
    ply_path_list.pop(0)
    prev_cluster_center = first_cluster_center
    for ply_path in ply_path_list:
        all_files = os.listdir(ply_path)
        all_clusters = []
        for file in all_files:
            if 'cluster' in file:
                all_clusters.append(file)
        min_dist = 999999
        for cluster in all_clusters:
            cluster_path = ply_path + cluster
            cluster_center = get_cluster_center(cluster_path)
            dist = calc_euclidian_dist(cluster_center, prev_cluster_center)
            if dist < min_dist:
                min_dist = dist
                closest_cluster = cluster_path
                temp = cluster_center
        prev_cluster_center = temp
        print(closest_cluster)
        same_cluster_paths.append(closest_cluster)
    # draw the average height graph of this cluster
    return avg_height_graph(same_cluster_paths)


def track_cluster_based_on_number(idx):
    general_path = '/Users/yousefesp/PycharmProjects/cloud_project/3views_results/'
    ply_path_list = []
    all_folders = os.listdir(general_path)
    # collect all folders for ply path (not the file just the folder)
    for folder in all_folders:
        if 'time_' in folder:
            ply_path_list.append(general_path + folder + '/scene_final/cluster_'+str(idx)+'.ply')
    return avg_height_graph(ply_path_list)


def clusters_avg_height():
    time = range (0,250,10)
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            axs[i, j].plot(time, track_cluster_based_on_number(i*3+j))
    plt.show()

def calc_euclidian_dist(list1, list2):
    dif_x = abs(list1[0] - list2[0])
    dif_y = abs(list1[1] - list2[1])
    dif_z = abs(list1[2] - list2[2])
    temp = dif_x**2 + dif_y**2 + dif_z**2
    return math.sqrt(temp)
# ------------------------------- MAIN -------------------------------
if __name__ == '__main__':
    #create_all_clusters(7)
    # combine_color_clusters(7)
    # filepath = '/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_colors.ply'
    # visualize(filepath)
    # plydata = PlyData.read(filepath)
    # print(plydata['vertex'].data['red'])

    #visualize('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_240_sec/scene_final/colors.ply')
    create_gif_for_clusters()
    #same_cluster_avg_height(2)

    # filep='/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_3.ply'
    # plydata = PlyData.read(filep)
    # for point in plydata.elements[0].data:
    #     point['red'] = 200
    #     point['green'] = 200
    #     point['blue'] = 0
    # el = PlyElement.describe(plydata['vertex'].data, 'vertex')
    # PlyData([el], text=True).write('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_3_special.ply')
    #visualize('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_3_special.ply')

    # plydata1 = PlyData.read('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_3_special.ply')
    # plydata2 = PlyData.read('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_2.ply')
    # points = np.concatenate(plydata1['vertex'], plydata2['vertex'])
    # o3d.geometry.PointCloud.cluster_dbscan()
    # el = PlyElement.describe(points, 'vertex')
    # PlyData([el], text=True).write('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/try1.ply')
    # visualize('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/try1.ply')

    # pcd1 = o3d.io.read_point_cloud('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_2.ply')
    # pcd2= o3d.io.read_point_cloud('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_3_special.ply')
    # pcd_combined = o3d.geometry.PointCloud()
    # pcd_combined = pcd1 + pcd2
    # o3d.io.write_point_cloud("/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/multiway_registration.ply", pcd_combined)
    #visualize('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/multiway_registration.ply')

    # print(get_cluster_center('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_020_sec/scene_final/cluster_4.ply'))
    # print(get_cluster_center('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_4.ply'))
    # print('####')
    # for i in range(7):
    #     print(get_cluster_center('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_010_sec/scene_final/cluster_'+str(i)+'.ply'))
    # print(calc_euclidian_dist(get_cluster_center('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_4.ply'), get_cluster_center('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_010_sec/scene_final/cluster_4.ply')))