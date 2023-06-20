import json
import numpy as np
import cv2
import os

from utils.utils import *

class DatasetGenerator(torch.utils.data.Dataset):

    def __init__(self, pbr_path, obj_ids):
        self.pbr_path = pbr_path
        self.data = self.load_pbr_dataset(self.pbr_path)
        self.data_combined = []

        # Create combined list with all objects
        for new_id,old_id in enumerate(obj_ids):
            curr_obj = self.data[int(old_id)]
            for i,_ in enumerate(curr_obj):
                curr_obj[i]['obj_id'] = new_id
            self.data_combined += curr_obj


    def __len__(self):
        return len(self.data_combined)

    def __getitem__(self, idx):
        curr_item = self.data_combined[idx]
        img = self.sample2img(curr_item)
        obj_id = curr_item['obj_id']
        R_gt = curr_item['R']

        # Convert R matrix from opengl to pytorch format
        xy_flip = np.eye(3, dtype=np.float)
        xy_flip[0,0] = -1.0
        xy_flip[1,1] = -1.0
        R_conv = np.transpose(np.array(R_gt).reshape(3,3))
        R_conv = np.dot(R_conv,xy_flip)
        return img, (obj_id, R_conv)

    def random_sample(self, obj_id):
        random_ind = np.random.randint(0, len(data[obj_id]))
        sample = self.data[obj_id][random_ind]
        return sample

    def sample2img(self, sample):
        img_bgr = cv2.imread(sample['img_name'])
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)[:,:,:3] # .astype(float)
        img = img.astype(np.float32) / 255.0
        bb_xywh = sample['BB_vis']
        cropped = extract_square_patch(img, (bb_xywh))
        return cropped

    def get_dirs(self, path):
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    def load_pbr_dataset(self, pbr_path):
        dirs = self.get_dirs(pbr_path)

        data_all = {}
        for d in dirs:
            dir_path = os.path.join(pbr_path, d)
            curr_data = self.load_pbr_dir(dir_path)

            print("Loading - {0}".format(dir_path))

            for k in curr_data.keys():
                if(k not in data_all):
                    data_all[k] = curr_data[k]
                else:
                    data_all[k] += curr_data[k]
            #break
        return data_all

    def load_pbr_dir(self, dir_path):
        gt_path_Rs = os.path.join(dir_path, 'scene_gt.json')
        gt_path_BB = os.path.join(dir_path, 'scene_gt_info.json')

        # Dict of objects
        # each with an R and BB
        data = {}

        # Open both files
        f_Rs = open(gt_path_Rs)
        f_BB = open(gt_path_BB)

        gt_data_Rs = json.load(f_Rs)
        gt_data_BB = json.load(f_BB)

        # Iterate through scene images
        for k in gt_data_Rs.keys():
            curr_scene_Rs = gt_data_Rs[k]
            curr_scene_BB = gt_data_BB[k]

            # Iterate through each object in the scene
            for o,_ in enumerate(curr_scene_Rs):
                curr_obj_Rs = curr_scene_Rs[o]
                curr_obj_BB = curr_scene_BB[o]

                obj_id = curr_obj_Rs['obj_id']
                if obj_id not in data:
                    data[obj_id] = []

                curr_obj = {}
                curr_obj['obj_id'] = obj_id
                curr_obj['R'] = curr_obj_Rs['cam_R_m2c']
                curr_obj['t'] = curr_obj_Rs['cam_t_m2c']
                curr_obj['BB_gt'] = curr_obj_BB['bbox_obj']
                curr_obj['BB_vis'] = curr_obj_BB['bbox_visib']
                img_name = str("rgb/{0:06d}.jpg").format(int(k))
                curr_obj['img_name'] = os.path.join(dir_path, img_name)

                if(curr_obj_BB['visib_fract'] < 0.5):
                        continue

                data[obj_id].append(curr_obj)
        return data


if __name__ == "__main__":
    data_gen = DatasetGenerator('data/train_pbr/', [5, 10])

    # for i in data_gen.data_combined.keys():
    #     print("Obj: {0} - num: {1}".format(i,len(data_gen.data_combined[i])))
    #     #sample = random_sample(data, i)
    #     #print(sample)
    #     #img = sample2img(sample)
    #     #cv2.imwrite("obj{0}.png".format(i), img)


    for i in data_gen:
        img, gt = i
        if(gt[0] == 5):
                print(gt)
