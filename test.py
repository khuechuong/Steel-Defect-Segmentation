import cv2
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils_test import plot_examples
import os
from PIL import Image
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
# dir = 'data/Images_val/'
# image1 = Image.open(dir +'frame1080.jpg')
#
# image2 = Image.open(dir+'frame2040.jpg')
# image3 = Image.open(dir+'frame2700.jpg')
# image4 = Image.open(dir+'frame2880.jpg')
# image5 = Image.open(dir+'frame540.jpg')
#
# #resize, first image
# image1 = image1.resize((512, 512))
# image1_size = image1.size
# image2 = image2.resize((512, 512))
# image3 = image3.resize((512, 512))
# image4 = image4.resize((512, 512))
# image5 = image5.resize((512, 512))
# new_image = Image.new('RGB',(5*image1_size[0], image1_size[1]), (255,255,255))
# new_image.paste(image1,(0,0))
# new_image.paste(image2,(image1_size[0],0))
# new_image.paste(image3,(2*image1_size[0],0))
# new_image.paste(image4,(3*image1_size[0],0))
# new_image.paste(image5,(4*image1_size[0],0))
# new_image.show()
# new_image.save('merged_image_.jpg')

dir = "unet_val_checkpoint/"
image_names = [f for f in os.listdir(dir) if '.pth.tar' in f]
image_names.sort()
print (len(image_names))
acc = []
TP = []
TN = []
epoch = list(range(1,101))
max_epoch = -1
string = ""
for i in range(1,101):
    for a in image_names:
        b = a.split('_')
        #d = float(b[3]) + float(b[4])/100 + float((b[5])[:-8])
        c = b[2]
        if i == int(c):
            string = a
    b = string.split('_')
    TP.append(float(b[3]))
    acc.append(float(b[4])/100 )
    TN.append(float((b[5])[:-8]))

plt.plot(acc, 'r')
plt.plot(TP, 'g')
plt.plot(TN, 'b')
plt.title("Epoch progress")
plt.ylabel("Measurements percentage")
plt.xlabel("Epoch")
plt.legend(['Accuracy','True Positive','True Negative'],loc="lower right")
plt.show()



# count = 0
# for i in mask_:
#     for j in i:
#         if j == 255:
#             count = count+1
# print(count)
#mask_ = np.array(Image.open(self.mask_dir + mask_names[i]).convert("L"), dtype=np.float32)
# image = Image.open(dir+"frame30.jpg").convert("RGB")
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.show()


# transform = A.Compose(
#     [
#         A.RandomCrop(width=572, height=572),
#         A.Rotate(limit=[-180,180], p=0.9),
#         A.HorizontalFlip(p=0.5)
#     ]
# )
#
# images_list = [image]
# image = np.array(image)
# for i in range(15):
#     augmentations = transform(image=image)
#     augmented_image = augmentations["image"]
#     images_list.append(augmented_image)
#
# plot_examples(images_list)

# image_names = [f for f in os.listdir(dir) if '.jpg' in f]
# image_list = []
#
# image_names.sort()
# #print(image_names)
#
# for i in range(0,len(image_names)):
#     image = Image.open(dir+image_names[i]).convert("RGB")
#     image_list.append(image)
# print(len(image_list))
# filter_config =(4,8,16)
# n_init = 3
# encod_confi = (n_init,) + filter_config # (3,4,8,16)
# print(encod_confi)
# final = sum(filter_config) + filter_config[0] #4+8+16+4
# print(final)