# PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image

# Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
import os
import time

# Models lib
from models import *

# Metrics lib
from metrics import calc_psnr, calc_ssim, calc_lpips


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="baseline", help="baseline, dsconv"
    )
    parser.add_argument("--mode", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument("--ckpt_path", type=str, default="./weights/baseline_gen.pkl")
    
    args = parser.parse_args()
    return args


def align_to_four(img):
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    return img


def predict(image, time_list):
    img = transforms.Resize((512, 512), antialias=True)(image)
    img = transforms.ToTensor()(img)
    img = img#.cuda()
    img = img.unsqueeze(0)
    start_time = time.time()
    out = model(img)[-1]
    end_time = time.time()
    print(f"Time taken for prediction: {end_time - start_time} seconds")
    time_list.append(end_time - start_time)
    out = out.detach().cpu().numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :] * 255.0

    return out, time_list


if __name__ == "__main__":
    args = get_args()

    if args.model == "dsconv":
        model = DSConvGenerator()#.cuda()
    else:
        model = Generator()#.cuda()
    model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device('cpu')))
    model.eval()

    if args.mode == "demo":
        if not os.path.exists(args.output_dir):
            print("Output directory does not exist. Creating it now...")
            os.makedirs(args.output_dir)
        input_list = sorted(os.listdir(args.input_dir))
        num = len(input_list)
        time_list = []
        for i in range(num):
            print("Processing image: %s" % (input_list[i]))
            img = Image.open(os.path.join(args.input_dir, input_list[i])).convert("RGB")
            width, height = img.size
            result, _ = predict(img, time_list)
            result = result.astype(np.uint8)
            result = Image.fromarray(result)
            result = result.resize((width, height))
            img_name = input_list[i].split(".")[0]
            path = os.path.join(args.output_dir, img_name + ".jpg")
            result.save(path)

    elif args.mode == "test":
        input_list = sorted(os.listdir(args.input_dir))
        gt_list = sorted(os.listdir(args.gt_dir))
        num = len(input_list)
        cumulative_psnr = 0
        cumulative_ssim = 0
        cumulative_lpips = 0
        time_list = []
        for i in range(num):
            print("Processing image: %s" % (input_list[i]))
            img = Image.open(os.path.join(args.input_dir, input_list[i])).convert("RGB")
            gt = Image.open(os.path.join(args.gt_dir, gt_list[i])).convert("RGB")
            gt = transforms.Resize((512, 512))(gt)
            gt = np.array(gt, dtype=np.uint8)
            result, time_list = predict(img, time_list)
            result = np.array(result, dtype="uint8")
            cur_psnr = calc_psnr(result, gt)
            cur_ssim = calc_ssim(result, gt)
            cur_lpips = calc_lpips(gt, result)
            print(
                "PSNR is %.4f and SSIM is %.4f and LPIPS is %.4f"
                % (cur_psnr, cur_ssim, cur_lpips)
            )
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
            cumulative_lpips += cur_lpips
        print(
            "In testing dataset, PSNR is %.4f and SSIM is %.4f and LPIPS is %.4f"
            % (cumulative_psnr / num, cumulative_ssim / num, cumulative_lpips / num)
        )
        print(
            f"Average time taken for prediction: {sum(time_list) / len(time_list)} seconds"
        )

    else:
        print("Mode Invalid!")
