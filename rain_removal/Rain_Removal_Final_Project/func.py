import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch.autograd import Variable


def get_mask(dg_img, img):
    # downgraded image - image
    mask = np.fabs(dg_img - img)
    # threshold under 30
    mask[np.where(mask < (30.0 / 255.0))] = 0.0
    mask[np.where(mask > 0.0)] = 1.0
    # avg? max?
    # mask = np.average(mask, axis=2)
    # mask = np.max(mask, axis=2)
    mask = np.mean(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    return mask

def torch_variable(x, is_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_train:
        return Variable(
            torch.from_numpy(np.array(x)), requires_grad=True
        ).to(device)
    else:
        # Replace volatile=True with torch.no_grad()
        with torch.no_grad():
            return Variable(torch.from_numpy(np.array(x))).to(
                device
            )
