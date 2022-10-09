import os
from decimal import Decimal
import re
import cv2
import csv
import time


        ###########################################################
        ##### image_root : your model inference results' path   ###
        ##### gt_root : gt files' path                          ###
        ###########################################################

gt_root = 'output/kvasir/gt/'
image_root = 'output/kvasir/pred/'

def criteria(root_outputs, root_test):
    dice_sum = 0
    acc_sum = 0
    count = 0

    files_outpus = os.listdir(root_outputs)
    for f1 in files_outpus:
        count += 1
        cmd = r"./EvaluateSegmentation " + \
              root_test + f1 + " " + root_outputs + f1 + r" -thd 0.5 -use DICE"
        f = os.popen(cmd)
        all = f.read()
        nums = re.findall(r'[0-9]+\.?[0-9]*', all)
        f.close()

        dice = Decimal(nums[1])
        dice_sum += dice

    dice = round(Decimal(dice_sum / count), 4)
    print("dice score")
    print(dice)

if __name__ == "__main__":
    criteria(gt_root, image_root)
