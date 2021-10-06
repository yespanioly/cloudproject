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

# ----------------------------- Stuff for ICP -----------------------------------
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


def icp_3d(a, b, init_pose=(0,0,0), no_iterations = 13):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''

    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)

    # Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]), -np.sin(init_pose[2]), init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]), init_pose[1]],
                   [0,                    0,                   1          ]])

    src = cv2.transform(src, Tr[0:2])

    for i in range(no_iterations):
        # Find the nearest neighbours between the current source and the
        # destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto',
                                warn_on_equidistant=False).fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])

        # Compute the transformation between the current source
        # and destination cloudpoint
        T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
        # Transform the previous source and update the
        # current source cloudpoint
        src = cv2.transform(src, T)
        # Save the transformation from the actual source cloudpoint
        # to the destination
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
    return Tr[0:2]


# -------------------------- Experimenting & Visualizing -------------------------
def try_ply(img):
    plydata = pf.PlyData.read(img)
    print(plydata.elements)


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


def create_gif():
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

# ------------------------------- Stuff for SIFT ---------------------------------
def sift_f():
    # read images

    img1 = cv2.imread('/Users/yousefesp/Desktop/Photos/Screenshots/Screen Shot 2021-06-08 at 11.48.52.png')
    img2 = cv2.imread('/Users/yousefesp/Desktop/Photos/Screenshots/Screen Shot 2021-06-08 at 12.02.22.png')

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    print(len(matches))

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:10],img2, flags=2)
    plt.imshow(img3), plt.show()
    # cv2.imwrite('matches.jpg', img3)


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
    # general_path = '/Users/yousefesp/PycharmProjects/cloud_project/3views_results/'
    # img_list = []
    # all_folders = os.listdir(general_path)
    # for folder in all_folders:
    #     if 'time_' in folder:
    #         img_list.append(general_path+folder+'/scene_final/S.ply')
    i = 0
    time_axis = []
    height_axis = []
    for img in img_list:
        time = i * 10
        time_axis.append(time)
        avg_h = calc_avg_height_img(img)
        height_axis.append(avg_h)
        i += 1
    plt.plot(time_axis, height_axis)
    plt.xlabel('Time [sec]')
    plt.ylabel('Average height [m]')
    plt.show()


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

def modify_ply_keep_center():
    """
    takes a ply file and levels down the perimeter (keeps the center)
    """
    file_path = '/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/S.ply'
    plydata = PlyData.read(file_path)
    print(plydata)
    for point in plydata['vertex'].data:
        if not (-2000<=point['x']<=2000 and -2000<=point['y']<=2000):
            point['z'] = 0
    el = PlyElement.describe(plydata['vertex'].data, 'vertex')
    PlyData([el], text=True).write('/Users/yousefesp/PycharmProjects/cloud_project/keep_center.ply')


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
        cluster_points = plydata['vertex'].data[clusters_indices==i]
        el = PlyElement.describe(cluster_points, 'vertex')
        PlyData([el], text=True).write(path+'/cluster_'+str(i)+'.ply')


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


# tried and didn't work
def determine_biggest_clusters():
    general_path = '/Users/yousefesp/PycharmProjects/cloud_project/3views_results/'
    ply_path_list = []
    all_folders = os.listdir(general_path)
    for folder in all_folders:
        if 'time_' in folder:
            ply_path_list.append(general_path + folder + '/scene_final')
    biggest_clusters = []
    for path in ply_path_list:
        biggest_cluster = ''
        max_points = 0
        for i in range(7):
            cluster_path = path+'/cluster_'+str(i)+'.ply'
            plydata = PlyData.read(cluster_path)
            if len(plydata.elements[0].data) > max_points:
                max_points = len(plydata.elements[0].data)
                biggest_cluster = cluster_path
        biggest_clusters.append(biggest_cluster)
    avg_height_graph(biggest_clusters)


def same_cluster_avg_height(idx):
    general_path = '/Users/yousefesp/PycharmProjects/cloud_project/3views_results/'
    ply_path_list = []
    all_folders = os.listdir(general_path)
    # collect all folders for ply path (not the file just the folder)
    for folder in all_folders:
        if 'time_' in folder:
            ply_path_list.append(general_path + folder + '/scene_final/')
    ply_path_list.sort()
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
        min_dist = 99999999
        for cluster in all_clusters:
            cluster_path = ply_path + cluster
            cluster_center = get_cluster_center(cluster_path)
            dist = calc_euclidian_dist(cluster_center, prev_cluster_center)
            if dist < min_dist:
                min_dist = dist
                closest_cluster = cluster_path
                temp = cluster_center
        prev_cluster_center = temp
        same_cluster_paths.append(closest_cluster)
    avg_height_graph(same_cluster_paths)


def calc_euclidian_dist(list1, list2):
    dif_x = abs(list1[0] - list2[0])
    dif_y = abs(list1[1] - list2[1])
    dif_z = abs(list1[2] - list2[2])
    temp = dif_x**2 + dif_y**2 + dif_z**2
    return math.sqrt(temp)
# ------------------------------- MAIN -------------------------------
if __name__ == '__main__':
    # create_all_clusters(7)
    visualize('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_4.ply')
    #same_cluster_avg_height(4)
    # print(get_cluster_center('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_020_sec/scene_final/cluster_4.ply'))
    # print(get_cluster_center('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_4.ply'))
    # print('####')
    # for i in range(7):
    #     print(get_cluster_center('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_010_sec/scene_final/cluster_'+str(i)+'.ply'))
    # print(calc_euclidian_dist(get_cluster_center('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_000_sec/scene_final/cluster_4.ply'), get_cluster_center('/Users/yousefesp/PycharmProjects/cloud_project/3views_results/time_010_sec/scene_final/cluster_4.ply')))