import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
#from collections import defaultdict
import resource, sys

def shared_points(cell1, cell2):
    shared = []
    for p in cell1:
        if p in cell2:
            shared.append(p)
    return shared

def plot_boundaries(regions, neighbours, sVoronoi, ax):
    # plot the boundary of each region by finding a cell on the outer border and folloing the edge until it loops (assumes no hole topology)
    print(regions[0])
    print(neighbours[0])
    print(sVoronoi.regions[0])
    print(sVoronoi.vertices[0])
    p1 = sVoronoi.vertices[sVoronoi.regions[50][0]]
    p2 = sVoronoi.vertices[sVoronoi.regions[50][1]]
    p3 = sVoronoi.vertices[sVoronoi.regions[50][2]]

    #ax.scatter(p1[0]*1.1, p1[1]*1.1, p1[2]*1.1, c='magenta', s=10)
    #ax.scatter(p2[0]*1.1, p2[1]*1.1, p2[2]*1.1, c='magenta', s=10)
    #ax.scatter(p3[0]*1.1, p3[1]*1.1, p3[2]*1.1, c='magenta', s=10)

    for region in regions:
        border = []
        bx = []
        by = []
        bz = []
        set = regions[region]['region_set']
        # find member of set with a non-same-set neightbour
        start = None
        next = None
        current = None
        for index in set:
            cell = sVoronoi.regions[index]
            neighs = neighbours[index]
            for neigh in neighs:
                if neigh not in set:
                    #a = (sVoronoi.points[index]*1.1)
                    #ax.scatter(a[0], a[1], a[2], c='magenta', s=10)
                    pts = shared_points(cell, sVoronoi.regions[neigh])
                    start = pts[0]
                    next = pts[1]
                    current = index
                    border.append(index)
                    bx = [sVoronoi.vertices[pts[0]][0]*1.1, sVoronoi.vertices[pts[1]][0]*1.1]
                    by = [sVoronoi.vertices[pts[0]][1]*1.1, sVoronoi.vertices[pts[1]][1]*1.1]
                    bz = [sVoronoi.vertices[pts[0]][2]*1.1, sVoronoi.vertices[pts[1]][2]*1.1]
                    break
        i = 0
        print()
        while next is not start:
            i += 1
            #print("{} {}".format(next, start))
            #cell = sVoronoi.regions[current]
            neighs = neighbours[current]
            # find which in-neighbour shares point "next"
            for neigh in neighs:
                if neigh in set and next in sVoronoi.regions[neigh]:# and (neigh is start or neigh not in border):
                    cell = sVoronoi.regions[neigh]
                    neighs2 = neighbours[neigh]
                    for neigh2 in neighs2:
                            if neigh2 not in set:
                                border.append(neigh)
                                #print("{} {}".format(current,neighs))
                                #print("{} {}".format(next,start))
                                #print(border)
                                current = neigh
                                pts = shared_points(cell, sVoronoi.regions[neigh2])
                                if next not in pts:
                                    #print("Error, missing point in neightbour")
                                    continue
                                for p in pts:
                                    if p is not next:
                                        next = p
                                bx.append(sVoronoi.vertices[p][0]*1.1)
                                by.append(sVoronoi.vertices[p][1]*1.1)
                                bz.append(sVoronoi.vertices[p][2]*1.1)
                                ax.scatter(bx, by, bz, c='magenta', s=2)
                                plt.show()
                                continue
            break
        print(border)
        ax.scatter(bx, by, bz, c='magenta', s=2)


# recursively
def steepest_region_for(current, neighbours, losses, steepest, found=set(), current_path=set(), regions={}):
    current_path |= {current}

    if current in found:
        for region in regions:
            if current in regions[region]['region_set']:
                regions[region]['region_set'] |= current_path
                found |= {current}
                return found, regions, steepest

    found |= {current}

    # global index of neighbour with the lowest loss
    index = neighbours[current][np.argmin(losses[neighbours[current]])]
    # if current har smaller loss than than lowest neighbour this it the minima, and has not been found before this
    if losses[index] > losses[current] or (losses[index] >= losses[current] and index > current):
        regions[current] = {'index': current, 'min loss': losses[current], 'region_set': current_path}
        return found, regions, steepest
    steepest[current] = index
    return steepest_region_for(index, neighbours, losses, steepest, found, current_path, regions)

def steepest_region(neighbours, losses):
    resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
    sys.setrecursionlimit(10**6)

    steepest = [-1]*len(losses)

    found = set()
    regions = {}
    for i in range(len(neighbours)):
        if i not in found:
            found_temp, regions_temp, steepest = steepest_region_for(i, neighbours, losses, steepest, found=found, current_path=set(), regions=regions)
            found |= found_temp
            regions.update(regions_temp)
    return regions, steepest

def voronoi_plot(points_in, losses, path, cmap):
    from matplotlib import colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    from scipy.spatial import SphericalVoronoi
    from mpl_toolkits.mplot3d import proj3d
    # get input points in correct format
    cart = [point['cartesian'] for point in points_in]
    points = np.array(cart)
    center = np.array([0, 0, 0])
    radius = 1
    # calculate spherical Voronoi diagram
    sv = SphericalVoronoi(points, radius, center)
    # sort vertices (optional, helpful for plotting)
    sv.sort_vertices_of_regions()
    # generate plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot the unit sphere for reference (optional)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='y', alpha=0.1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')

    # normalize and map losses to colormap
    mi = min(losses)
    ma = max(losses)

    norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    loss_color = [mapper.to_rgba(l) for l in losses]

    # indicate Voronoi regions (as Euclidean polygons)
    for i in range(len(sv.regions)):
        region = sv.regions[i]
        random_color = colors.rgb2hex(loss_color[i])
        polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
        polygon.set_color(random_color)
        ax.add_collection3d(polygon)

    fig.savefig(os.path.join(path, "landscape.png"), dpi=fig.dpi)

    # determine neighbours
    neighbours = []
    for i in range(len(sv.regions)):
        vertices = sv.regions[i]
        neighbours_for_i = []
        for j in range(len(sv.regions)):
            neigh = False
            shared = 0
            vert2 = sv.regions[j]
            for vert in vertices:
                if vert in vert2:
                    shared += 1
                    if shared >= 2:
                        neigh = True
                        break
            if neigh and i != j:
                neighbours_for_i.append(j)
        neighbours.append(neighbours_for_i)

    regions, steepest_neighbour = steepest_region(neighbours, losses)
    #plot_boundaries(regions, neighbours, sv, ax)

    temp = []
    for i in range(len(neighbours)):
        point1 = sv.points[i]
        point2 = points[steepest_neighbour[i]]
        temp.append(point2-point1)

    x1, y1, z1 = zip(*sv.points)
    dx, dy, dz = zip(*temp)

    col = [[0, 0, 0, 0.5] for i in range(len(neighbours))]
    #ax.quiver(x1, y1, z1, dx, dy, dz, length=0.02, normalize=True, colors=col) # uncomment this get directional arrows as well

    num_sets = len(regions)
    colors = cm.rainbow(np.linspace(0, 1, num_sets))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='y', alpha=0.1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')

    # indicate Voronoi regions (as Euclidean polygons)
    i = 0
    centers = []
    for reg in regions:
        set = regions[reg]['region_set']
        centers.append(reg)
        for index in set:
            region = sv.regions[index]
            polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
            polygon.set_color(colors[i])
            ax.add_collection3d(polygon)
        i += 1

    x_p, y_p, z_p = zip(*sv.points)
    x_p = [x_p[i] for i in centers]
    y_p = [y_p[i] for i in centers]
    z_p = [z_p[i] for i in centers]
    ax.scatter(x_p, y_p, z_p, c='k', s=1)
    fig.savefig(os.path.join(path, "regions.png"), dpi=fig.dpi)
    plt.show()

def plot_points(points, losses):
    cart = [point['cartesian'] for point in points]
    x, y, z = zip(*cart)
    #fig = plt.figure()

    ma = max(losses)
    mi = min(losses)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    print(z)

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=losses, s=700)
    plt.show()

    #norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma, clip=True)
    #mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)

    #loss_color = [mapper.to_rgba(l) for l in losses]


    #norm = matplotlib.colors.Normalize(mi, ma)
    #m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    #m.set_array([])
    #fcolors = m.to_rgba(losses)

    #X, Y = np.meshgrid(x, y)    # 50x50
    #Z = np.outer(z.T, z)        # 50x50

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_trisurf(x, y, z, rstride=1, cstride=1, color=losses, shade=0, cmap=cm.Greys_r)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, shade=0)

def plot_flat_landscape(points_in, losses, path, cmap):
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
    #fig = voronoi_plot_2d(vor)
    #plt.scatter(theta, phi)

    minima = min(losses)
    maxima = max(losses)

    # normalize chosen colormap
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap) #cm.jet

    #loss_color = [mapper.to_rgba(l) for l in losses]

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
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(losses[r]))

    mapper.set_array(losses)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "2%", pad="1%")
    cbar = plt.colorbar(mapper, cax=cax)
    cbar.ax.tick_params(labelsize=60)

    #print(*list(zip(losses,points_in)), sep = "\n")
    index = list(range(len(losses)))

    sorted_points = np.array([(x,y,z) for x,y,z in sorted(zip(losses,points_in,index), key=lambda tup: tup[0])])
    last = 1
    x = []
    y = []
    for i in range(last):
        pt = sorted_points[i][1]['spherical']
        #print(sorted_points[i])
        x.append(pt[0])
        y.append(pt[1])
    #plt.plot(x, y,'ro')
    #for i in range(10):
        #print(sorted_points[i])

    # print(angles[494])
    # print(angles[972])
    # print(angles[530])
    # print(angles[1012])
    plt.show()

    fig.savefig(os.path.join(path, "flat_landscape.png"), dpi=fig.dpi)

def main():
    import faulthandler
    faulthandler.enable()
    #np.set_printoptions(threshold=sys.maxsize)
    path = sys.argv[1]
    points = np.load(os.path.join(path, 'points.npy'), allow_pickle=True)
    losses = np.load(os.path.join(path, 'losses.npy'), allow_pickle=True)

    cmap = cm.get_cmap('jet', 256)
    first = 550
    total = 1000
    newcolors = np.vstack((cmap(np.linspace(0, 0.2, first)),
                       cmap(np.linspace(0.2, 1, total-first))))
    newcmp = ListedColormap(newcolors)
    plot_flat_landscape(points, losses, path, newcmp)
    voronoi_plot(points, losses, path, newcmp)


if __name__ == '__main__':
    main()
