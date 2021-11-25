import torch
import numpy as np
import cv2
from Encoder import Encoder
from Model import Model

#import onnxruntime as rt

"""class Pipeline():
    def __init__(self, encoder, model, device):
        self.encoder = encoder
        self.model = model
        self.device = device

    # Input: x = images as list of numpy arrays
    # Output: y = pose as 6D representation
    def process(self, images):
        # Disable gradients for the encoder
        with torch.no_grad():fnatest_img_pathme
            # Convert images to AE codes
            codes = []
            for img in images:
                # Normalize image
                img_max = np.max(img)
                img_min = np.min(img)
                img = (img - img_min)/(img_max - img_min)

                # Run image through encoder
                img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).to(self.device)
                #print(img.shape)
                code = self.encoder(img.float())
                code = code.detach().cpu().numpy()[0]
                norm_code = code / np.linalg.norm(code)
                codes.append(norm_code)

        # Predict poses from the codes
        batch_codes = torch.tensor(np.stack(codes), device=self.device, dtype=torch.float32)
        predicted_poses = self.model(batch_codes)
        return predicted_poses
"""

# local copy to avoid relying on utils due to name clash
def loadCheckpoint(model_path):
    # Load checkpoint and parameters
    checkpoint = torch.load(model_path)

    # Load model
    num_views = int(checkpoint['model']['l3.bias'].shape[0]/(6+1))
    model = Model(num_views=num_views).cuda()

    model.load_state_dict(checkpoint['model'])

    print("Loaded the checkpoint: \n" + model_path)
    return model, num_views

# to be used if we just transmit coordinates, not crops
"""def extract_square_patch(scene_img, bb_xywh, pad_factor=1.2,resize=(128,128),
                         interpolation=cv2.INTER_NEAREST,black_borders=False):

        x, y, w, h = np.array(bb_xywh).astype(np.int32)
        size = int(np.maximum(h, w) * pad_factor)

        left = int(np.maximum(x+w/2-size/2, 0))
        right = int(np.minimum(x+w/2+size/2, scene_img.shape[1]))
        top = int(np.maximum(y+h/2-size/2, 0))
        bottom = int(np.minimum(y+h/2+size/2, scene_img.shape[0]))

        scene_crop = scene_img[top:bottom, left:right].copy()

        if black_borders:
            scene_crop[:(y-top),:] = 0
            scene_crop[(y+h-top):,:] = 0
            scene_crop[:,:(x-left)] = 0
            scene_crop[:,(x+w-left):] = 0

        scene_crop = cv2.resize(scene_crop, resize, interpolation = interpolation)
        return scene_crop"""

# batch*n
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]

    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3

    return out

# again copy to avoid importing utils
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3

    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

class Inference():
    def __init__(self):
        detector_path = "/home/hampus/vision/yolov3/runs/train/exp4/weights/best.pt"
        #detector_path = "./data/yolo/detector.pt"
        detector_model = "/home/hampus/vision/yolov3"
        encoder_weights = "./data/encoder/obj1-18/encoder.npy"
        model_path = "./output/test/models/model-epoch1.pt"

        """detector = torch.hub.load(detector_model, 'custom', source='local', path=detector_path)
        #detector = torch.jit.load(detector_path)
        #detector = torch.hub.load('ultralytics/yolov3', "yolov3", classes=30, force_reload=True, pretrained=False)
        #detector = torch.hub.load('https://github.com/HampusAstrom/yolov3', 'custom', path='./runs/train/exp4/weights/best.pt')
        #detector = ct.models.MLModel("/home/hampus/vision/yolov3/runs/train/exp4/weights/best.mlmodel")

        #from models.yolo import Model
        #yaml_path='models/yolov5m.yaml'
        #new_weights='weights/yolov5m_resave.pt'
        #detector = Model(yaml_path).to(device)
        checkpoint = torch.load(detector_path)#['model']
        detector.load_state_dict(checkpoint['model'].state_dict())
        detector = detector.autoshape()
        detector.names = checkpoint.names
        #print(detector)"""

        """detector = rt.InferenceSession("/home/hampus/vision/yolov3/runs/train/exp4/weights/best.onnx", None)

        input_name = detector.get_inputs()[0].name
        print("input name", input_name)
        input_shape = detector.get_inputs()[0].shape
        print("input shape", input_shape)
        input_type = detector.get_inputs()[0].type
        print("input type", input_type)
        output_name = detector.get_outputs()[0].name
        print("output name", output_name)
        output_shape = detector.get_outputs()[0].shape
        print("output shape", output_shape)
        output_type = detector.get_outputs()[0].type
        print("output type", output_type)"""

        device = torch.device("cuda:0")

        encoder = Encoder(encoder_weights).to(device)
        encoder.eval()

        model, num_views = loadCheckpoint(model_path)
        model = model.eval() # Set model to eval mode

        self.device = device
        #self.detector = detector
        self.encoder = encoder
        self.model = model
        self.num_views = num_views

    # Input: x = image as numpy arrays
    # Output: y = pose as 6D representation
    # run once for each
    def process_single(self, image):
        # Disable gradients for the encoder
        with torch.no_grad():
            # detect object in scene, get bboxes, crop and pass to AE for each object
            #pred = self.detector.predict(scene_image)
            #print(type(scene_image))
            #print(scene_image.shape)
            #pred = self.detector.run([self.detector.get_outputs()[0].name], {self.detector.get_inputs()[0].name: scene_image.astype(np.float32)})[0]
            #pred = detector(scene_img)

            resize=(128,128)
            interpolation=cv2.INTER_NEAREST

            img = cv2.resize(image, resize, interpolation = interpolation)

            # Convert images to AE codes
            codes = []

            # Normalize image
            img_max = np.max(img)
            img_min = np.min(img)
            img = (img - img_min)/(img_max - img_min)

            # Run image through encoder
            img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).to(self.device)
            #print(img.shape)
            code = self.encoder(img.float())
            code = code.detach().cpu().numpy()[0]
            norm_code = code / np.linalg.norm(code)

        # Predict poses from the codes
        batch_codes = torch.tensor(np.stack([norm_code]), device=self.device, dtype=torch.float32)
        predicted_poses = self.model(batch_codes)
        confs = predicted_poses[:,:self.num_views]
        # which pose has highest conf
        index = torch.argmax(confs)
        pose_start = self.num_views # +3 if with translation
        pose_end = pose_start + 6
        curr_pose = predicted_poses[:,pose_start:pose_end]
        Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
        return Rs_predicted.cpu().detach().numpy()


if __name__ == '__main__':

    inference = Inference()

    test_img_path = '/home/hampus/vision/AugmentedAutoencoder/multi-pose/detection_data/images/1.png'
    bgr = cv2.imread(test_img_path)

    pred = inference.process_single(bgr)
    print(pred)
