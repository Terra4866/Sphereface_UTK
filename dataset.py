from __future__ import print_function, division

import torch
import torchvision
from torch.utils.data import DataLoader
from skimage import io, transform
import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def landmark_load(path):
      f = open(path, 'r')
      lines = f.readlines()
      landmark_frame = []
      for line in lines:
            line_tmp = line.split(' ')[1:-1]
            line_tmp = [int(i) for i in line_tmp]
            line_tmp = np.array(line_tmp)
            line_tmp = line_tmp.reshape(-1, 2)
            str_tmp = line.split(' ')[0] + '.chip.jpg'
            #str_tmp = line.split(' ')[0]
            tmp = []
            tmp.append(str_tmp)
            tmp.append(line_tmp)
            landmark_frame.append(tmp)
      return landmark_frame



class UTK_DS(torch.utils.data.Dataset):
      
      def __init__(self, txt, root_dir, label, transform = None):
            self.root_dir = root_dir
            self.transform = transform
            self.landmarks_frame = landmark_load(txt)
            self.label = label

      def __len__(self):
           return len(self.landmarks_frame)

      def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.landmarks_frame[idx][0])
            image = Image.open(img_name).convert('RGB')
            landmarks = self.landmarks_frame[idx][1]
            label_tmp = self.landmarks_frame[idx][0].split('_')
            label_age = int(label_tmp[0])
            label_gender = int(label_tmp[1])
            if len(label_tmp) == 4:
                  label_race = int(label_tmp[2])
            else:
                  print(label_tmp[2])
            
            if self.label == "age":
                label = label_age
            elif self.label == "gender":
                label = label_gender
            elif self.label == "race":
                label = label_race


            if self.transform:
                  image = self.transform(image)

            return (image, label)

      
