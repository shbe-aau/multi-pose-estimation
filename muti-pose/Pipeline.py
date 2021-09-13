import torch
import numpy as np

class Pipeline():
    def __init__(self, encoder, model, device):
        self.encoder = encoder
        self.model = model
        self.device = device

    # Input: x = images as list of numpy arrays
    # Output: y = pose as 6D representation
    def process(self, images):
        # Disable gradients for the encoder
        with torch.no_grad():
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
