import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, count, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.count = count
        # list all file in that folder
        #self.images = os.listdir(image_dir)
        # MODIFICATIONS
        # int augmentation array
        self.images_augment = []
        self.mask_augment = []
        # get names and put in an array
        image_names = [f for f in os.listdir(image_dir) if '.jpg' in f]
        mask_names = [f for f in os.listdir(mask_dir) if '.jpg' in f]
        # sort the names in order (since names are everywhere)
        image_names.sort()
        mask_names.sort()

        # augmentation
        for i in range(0, len(image_names)):
            # get images from name
            image_ = np.array(Image.open(self.image_dir + image_names[i]).convert("RGB"))
            # grayscale so L instead of RGB
            #print(i)
            mask_ = np.array(Image.open(self.mask_dir + mask_names[i]).convert("L"), dtype=np.float32)
            #since we use sigmoid at the end, it should be 0 vs 1 not 0 vs 255
            mask_[mask_ == 255.0] = 1.0


            # transform each image 10 times
            if self.transform is not None:
                for j in range(0,count):
                    augmentations = self.transform(image=image_, mask=mask_)
                    image = augmentations["image"]
                    mask = augmentations["mask"]
                    self.images_augment.append(image)
                    self.mask_augment.append(mask)
        # empty name


    def __len__(self):
        #return len(self.images)
        # MODIFICATIONS
        return len(self.images_augment)

    def __getitem__(self, index):
        # /home/ + h.jpg = /home/h.jpg
        # img_path = os.path.join(self.image_dir, self.images[index])
        # mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        #mask_path = os.path.join(self.mask_dir, self.images[index])
        #image = np.array(Image.open(img_path).convert("RGB"))
        # grayscale so L instead of RGB
        #mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # since we use sigmoid at the end, it should be 0 vs 1 not 0 vs 255
        #mask[mask == 255.0] = 1.0

        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"]

        # MODIFICATIONS

        return self.images_augment[index], self.mask_augment[index]


