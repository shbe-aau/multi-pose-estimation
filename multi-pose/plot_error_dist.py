import json
import numpy as np
from tabulate import tabulate
from os import listdir
import matplotlib.pyplot as plt

objects = ["{:02d}".format(i+1) for i in range(30)]


def process_object(obj, data_split, approach, tau, thres, sort=False):
    # Load VSD from json
    base_path = "/home/hampus/Dropbox/PhD_stuff/results/"
    json_path = base_path + data_split + "/obj" + obj + "/eval/" + approach + "-obj" + obj + "_tless-" + data_split + "-primesense/error=vsd_ntop=-1_delta=15.000_tau=" + tau + "/matches_th=" + thres + "_min-visib=0.100.json"
    data = []
    try:
        with open(json_path) as f:
            data = json.load(f)
        print("Loaded a data vector of length: ", len(data))
    except:
        print("something went wrong!")

    #exit()

    pruned = []
    for d in data:
        if d['obj_id'] == int(obj) and d['valid']:
            # print(d)
            #if d['error'] == -1: # turn one element lists into values instead
            #    print(d)
            if d['error'] != -1: # turn one element lists into values instead
                d['error'] = d['error'][0]
            if d['error_norm'] != -1: # turn one element lists into values instead
                d['error_norm'] = d['error_norm'][0]
            pruned.append(d)

    #pruned = np.array(pruned)
    #pruned = np.where(pruned<0, 1, pruned)

    if sort:
        pruned = sorted(pruned, key = lambda i: i['error'])
    #print(pruned)
    print("Relevant values:                 {}".format(len(pruned)))
    return [ent['error'] for ent in pruned]

objects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
           "11", "12", "13", "14", "15", "16", "17", "19", "20",
           "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]

           #"11", "12", "13", "14", "15", "16", "17", "18", "19", "20",

objects1 = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
objects2 = ["11", "12", "13", "14", "15", "16", "17", "19", "20"]
objects3 = ["21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]

object = "23"

# ours = np.array(process_object(object, "test", "6views-depth-max30-199epochs", "0.500", "0.500", True))
# sund = np.array(process_object(object, "test", "sundermeyer", "0.500", "0.500", True))
# #process_object("20", "test", "sundermeyer")
# ours = np.where(ours<0, 1, ours)
# sund = np.where(sund<0, 1, sund)
#
# fig, ax = plt.subplots()
#
# ax.plot(ours, label='ours')
# ax.plot(sund, label='sund')
# legend = ax.legend(loc='upper center')
# plt.show()

fig, ax = plt.subplots(3, figsize=(20, 30))

#plt.axis('off')
#fig.axes.axis('tight')
#fig.axes.get_xaxis().set_visible(False)
#fig.axes.get_yaxis().set_visible(False)

fig.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import numpy.matlib as npm

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.matlib.repmat(np.linspace(0, 1, int(N/4)),4,1).flatten()
vals[:, 1] = np.matlib.repmat(np.linspace(0, 1, int(N/2)),2,1).flatten()
vals[:, 2] = np.linspace(0, 1, N)
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#cm = ListedColormap(vals)
cm = plt.get_cmap('tab10')

from cycler import cycler

sort_before_diff = False
tau = "0.500"
thres = "0.500"

for i, list in enumerate([objects1, objects2, objects3]):
    NUM_COLORS = 10 #len(list)
    ax[i].set_prop_cycle(cycler('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]) + cycler(linestyle=['-', '--', '-', '--', '-', '--', '-', '--', '-', '--', ]))
    for o in list:
        print("Comparing object {}".format(o))
        ours = np.array(process_object(o, "test", "6views-depth-max30-199epochs", tau, thres, sort_before_diff))
        sund = np.array(process_object(o, "test", "sundermeyer", tau, thres, sort_before_diff))
        ours = np.where(ours<0, 1, ours)
        sund = np.where(sund<0, 1, sund)

        diff = np.subtract(sund, ours)
        if not sort_before_diff:
            diff = np.sort(diff)
        else:
            shrink = 0.8
            diff = np.where(diff>1, diff-shrink, diff)
            diff = np.where(diff<-1, diff+shrink, diff)
        x = np.linspace(0, 1, len(diff))
        ax[i].plot(x, diff, label=o)
        ax[i].axis('tight')
        ax[i].axhline(y=0, color='grey', alpha=0.5)

        legend = ax[i].legend(loc='upper center', ncol=10)
plt.show()
