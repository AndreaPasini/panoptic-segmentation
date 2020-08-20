from __future__ import absolute_import, division, print_function
import os
import PIL.Image as pil
import cv2
import torch
from torchvision import transforms
from monodepth import networks
from monodepth.utils import download_model_if_doesnt_exist

"""
Depth prediction from 2D mono images.
Repository: https://github.com/nianticlabs/monodepth2
"""

class Monodepth:
    model_dir = './monodepth/models'

    def __init__(self):
        """
        Initialize the depth prediction model.
        """
        model_name = "mono_640x192"

        download_model_if_doesnt_exist(self.model_dir, model_name)
        encoder_path = os.path.join(self.model_dir, model_name, "encoder.pth")
        depth_decoder_path = os.path.join(self.model_dir, model_name, "depth.pth")

        # LOADING PRETRAINED MODEL
        self.encoder = networks.ResnetEncoder(18, False)
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        self.loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        self.loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(self.loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval();

    
    def predict(self, image):
        """
        Inference from a single image.
        :param image: must follow the format  (BGR)
        :return: 2D matrix with labels (panoptic COCO ids)
        """
        original_height, original_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = pil.fromarray(image)
        
        feed_height = self.loaded_dict_enc['height']
        feed_width = self.loaded_dict_enc['width']
        input_image_resized = image.resize((feed_width, feed_height), pil.LANCZOS)
        with torch.no_grad():
            input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
            features = self.encoder(input_image_pytorch)
            outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        # back to original size
        disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        return disp_resized_np