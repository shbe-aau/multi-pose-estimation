import torch
from DatasetGeneratorOpenGL import DatasetGenerator
from Pipeline import Pipeline
from Encoder import Encoder
from Model import Model
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.tools import *
from utils.utils import *
from scipy.spatial.transform import Rotation
from PIL import Image

def pointToMat(point):
    r = Rotation.from_euler('yz', point['spherical']) # select point wanted for comparison here
    R = r.as_matrix()
    return torch.from_numpy(R)

base_directory = './output/paper-models/10views/obj10/'
vis_directory = base_directory + 'visualization/'

def main():
    device = torch.device("cuda:0")
    num_datapoints = 1984
    views = 10
    load_points = True
    prepareDir(vis_directory)
    # Need to replace sundermeyer-random with something where we can
    # use predetermined poses, to plot each arch to visualize
    datagen = DatasetGenerator("",
                            ["./data/cad-files/ply-files/obj_10.ply"],
                            375,
                            num_datapoints,
                            "not_used",
                            device,
                            "sundermeyer-random",
                            random_light=False)

    # load model
    encoder = Encoder("./data/obj1-18/encoder.npy").to(device)
    encoder.eval()
    checkpoint = torch.load("./output/paper-models/10views/obj10/models/model-epoch199.pt")

    model = Model(num_views=views).cuda()
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    pipeline = Pipeline(encoder, model, device)

    # around x
    shiftx = np.eye(3, dtype=np.float)
    theta = np.pi / num_datapoints
    #theta = np.pi / 3
    shiftx[1,1] = np.cos(theta)
    shiftx[1,2] = -np.sin(theta)
    shiftx[2,2] = np.cos(theta)
    shiftx[2,1] = np.sin(theta)
    # around y
    shifty = np.eye(3, dtype=np.float)
    theta = np.pi / num_datapoints
    shifty[0,0] = np.cos(theta)
    shifty[0,2] = -np.sin(theta)
    shifty[2,2] = np.cos(theta)
    shifty[2,0] = np.sin(theta)
    # around z
    shiftz = np.eye(3, dtype=np.float)
    theta = np.pi / num_datapoints
    shiftz[0,0] = np.cos(theta)
    shiftz[0,1] = -np.sin(theta)
    shiftz[1,1] = np.cos(theta)
    shiftz[1,0] = np.sin(theta)
    predicted_poses = []
    predicted_poses_raw = []
    R_conv = np.eye(3, dtype=np.float)
    #R_conv = np.array([[ 0.5435,  0.1365,  0.8283],
    #                   [ 0.6597,  0.5406, -0.5220],
    #                   [-0.5190,  0.8301,  0.2037]])
    #R_conv = np.array([[-0.7132,  0.0407,  0.6998],
    #                   [ 0.1696, -0.9586,  0.2287],
    #                   [ 0.6802,  0.2818,  0.6767]])
    #R_conv = np.array([[-0.9959,  0.0797,  0.0423],
    #                   [ 0.0444,  0.0249,  0.9987],
    #                   [ 0.0786,  0.9965, -0.0283]])

    if load_points:
        # Try with points from the sphere
        points = np.load('./output/depth/spherical_mapping_obj10_1_500/points.npy', allow_pickle=True)
        num_datapoints = len(points)
        #print(points[0])
        #points = [points[0]]*num_datapoints
        #print(points[1])

        Rin = []
        for point in points:
            Rin.append(pointToMat(point))
            #Rin.append(np.matmul(pointToMat(point), shiftx))

    else:
        Rin = []
        for i in range(num_datapoints):
            # get data from fixed R and T vectors
            R_conv = np.matmul(R_conv, shiftx)
            R = torch.from_numpy(R_conv)
            Rin.append(R)

    t = torch.tensor([0.0, 0.0, 375])
    # Generate images
    data = datagen.generate_image_batch(Rin = Rin, tin = t, augment = False)

    for i, image in enumerate(data["images"]):
        im = Image.fromarray((image * 255).astype(np.uint8))
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save("{}{}.jpeg".format(vis_directory,i))

    # run images through model
    # Predict poses
    output = pipeline.process(data["images"])

    # evaluate how output confidence and each view changes with input pose
    plot_confidences(output.detach().cpu().numpy())
    if load_points:
        plot_flat_landscape(points, output[:,0:views].detach().cpu().numpy())

    rotation_matrices = []
    for i in range(views):
        start = views + i*6
        end = views + (i + 1)*6
        curr_poses = output[:,start:end]
        matrices = compute_rotation_matrix_from_ortho6d(curr_poses)
        euler_angles = compute_euler_angles_from_rotation_matrices(matrices)
        print(matrices.shape)
        print(matrices[0:3])
        print(euler_angles.shape)
        print(euler_angles[0:3])
        exit()

def plot_confidences(predicted_poses):
    num_datapoints = len(predicted_poses)
    views = int(len(predicted_poses[0])/7)

    x = np.linspace(0, np.pi, num_datapoints)
    #fig, ax = plt.subplots(int(views/2),2)
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    fig.tight_layout(rect=(0,0,0.95,1),pad=1.0)
    for i in range(views):
        #fig.add_subplot(int(views/2),2,i+1)
        plt.plot(x,predicted_poses[:,i], label=i)
        plt.ylim([-0.01, 1.01])
        plt.xlim([-0.01, 1.01])

    plt.legend(loc="upper right")
    # plt.show()
    fig.savefig(vis_directory + 'confidence.png', bbox_inches='tight')

def plot_flat_landscape(points_in, conficences):
    angles = [point['spherical'] for point in points_in]
    # shift for clearer image
    shift_radians = 1.1
    for i in range(len(angles)):
        temp = (angles[i][1] + shift_radians)
        angles[i][1] = temp if temp < 2*np.pi else temp-2*np.pi
    theta, phi = zip(*angles)

    from scipy.spatial import Voronoi, voronoi_plot_2d

    #print(angles[0])
    angles = np.array(angles)
    angles[:,[0, 1]] = angles[:,[1, 0]]
    #print(angles[0])
    vor = Voronoi(angles)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # plot Voronoi diagram, and fill finite regions with color mapped from losses
    fig = voronoi_plot_2d(vor, show_points=False, show_vertices=False, line_alpha=0, s=1)
    ax = fig.add_subplot(1,1,1)
    fig.set_size_inches(45, 15)
    fig.tight_layout(rect=(0,0,0.95,1),pad=1.0)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    #plt.axis('off')
    plt.ylim([min(theta), max(theta)])
    plt.xlim([min(phi), max(phi)])
    print(len(vor.point_region))
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            view = np.argmin(conficences[r,:])
            plt.fill(*zip(*polygon), color=colors[view])

    fig.savefig(vis_directory + 'confidence_landscape.png', bbox_inches='tight', dpi=fig.dpi)

if __name__ == '__main__':
    main()
