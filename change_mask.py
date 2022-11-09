import os
from PIL import Image
from matplotlib import pyplot as plt
#dir = "data/Masks/"
dir = "data/Masks_val/"
# dir = "data/Masks_test/"
image_names = [f for f in os.listdir(dir) if '.jpg' in f]
print(image_names)
for name in image_names:
    img = Image.open(dir + name)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(im)
    # plt.show()
    white = (255,255,255)
    black = (0,0,0)
    pixels = img.load()

    for i in range(img.size[0]):    # for every col:
        for j in range(img.size[1]):    # For every row
            if pixels[i,j][0] >230 and pixels[i,j][1] < 60 and pixels[i,j][2] < 60:
                pixels[i,j] = (255,255,255)
            else:
                pixels[i, j] = (0,0,0)
    #img.show()
    # plt.figure(figsize=(10, 10))
    # plt.imshow(newim)
    # plt.show()
    img.save(dir+name)
