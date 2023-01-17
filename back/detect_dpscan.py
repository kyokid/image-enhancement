import numpy as np
import torch
from PIL import Image
from dpscan.model import *
from dpscan.utils import *

class DPScan(object):
    def __init__(self):
        ckpt = torch.load("./dpscan/dpscan_saved_weights.pth.tar")

        self.G = Net(ckpt['opts']).cuda()
        self.G.load_state_dict(ckpt['G'])
        self.G.eval()
        self.img_size = (1072,720)

    def __call__(self, pil_img):
        with torch.no_grad():
            if self.img_size[0] != -1:
                pil_img = pil_img.resize(self.img_size, resample=Image.BICUBIC)

            tensor_img = self.totensor(np.array(pil_img)).cuda()
            tensor_out = self.G(tensor_img)

            tensor_img = tensor_img.cpu()
            tensor_out   = tensor_out.cpu()

            tensor_out = (tensor_out + 1)/2
            tensor_out = np.array(tensor_out[0,:,:,:].clamp(0,1).numpy().transpose(1,2,0) * 255.0, dtype=np.uint8)

        return tensor_out

    @staticmethod
    def totensor(tmp):
        tmp = tmp / 255.0
        tmp = (tmp - 0.5)/0.5
        tmp = tmp.transpose((2, 0, 1))
        return torch.from_numpy(tmp).unsqueeze(0).float()
