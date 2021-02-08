import xml.etree.ElementTree as ET
from os import listdir, remove, rename
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

IMAGE_DIR = "Dataset_7/best/images"
ANNOT_DIR = "Dataset_7/best/annotations"
AUGMENTED_DIR_IMG = "Dataset_7/best/augmented"
AUGMENTED_DIR_ANN = "Dataset_7/best/augmented"

index = 3073
for i in range(0, 1):
    for filename in listdir(ANNOT_DIR):
        tree = ET.parse(ANNOT_DIR + "/" + filename)
        root = tree.getroot()
        imagename = root.find("filename")
        sz = root.find("size")
        hg = sz.find("height")
        wd = sz.find("width")

        img = Image.open(IMAGE_DIR + "/" + imagename.text)
        image = np.array(img)

        #image_au = image[np.newaxis, ...]

        if(image.shape[2] != 3):
            image = image[...,:3]

        bnds = root.findall(".//bndbox")
        bb_list = []
        for bb in bnds:
            xmin = bb.find("xmin")
            ymin = bb.find("ymin")
            xmax = bb.find("xmax")
            ymax = bb.find("ymax")

            bb_list.append(BoundingBox(x1=int(xmin.text), y1=int(ymin.text), x2=int(xmax.text), y2=int(ymax.text)))
        bbs = BoundingBoxesOnImage(bb_list, shape=image.shape)

        import random
        rnd = random.randint(0, 5)
        if rnd == 0:
            seq = iaa.Sequential([
                iaa.Fliplr(0.7),
                iaa.RemoveSaturation(0.5),
                iaa.ScaleX((1.3, 1.4))
            ])
        elif rnd == 1:
            seq = iaa.Sequential([
                iaa.Fliplr(0.8),
                iaa.ChangeColorTemperature((1100, 7000)),
                iaa.ScaleX((1.3, 1.4))
            ])
        elif rnd == 2:
            seq = iaa.Sequential([
                iaa.Fliplr(0.7),
                iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True),
                iaa.ScaleX((1.3, 1.4))
            ])
        elif rnd == 3:
            seq = iaa.Sequential([
                iaa.Fliplr(0.8),
                iaa.AllChannelsHistogramEqualization()
            ])
        elif rnd == 4:
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.ScaleX((1.3, 1.7))
            ])
        elif rnd == 6:
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.ScaleY((1.3, 1.7))
            ])
        elif rnd == 5:
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.ScaleX((1.2, 1.5)),
                iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
            ])


        # Augment BBs and images.
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

        # print coordinates before/after augmentation (see below)
        # use .x1_int, .y_int, ... to get integer coordinates
        '''for i in range(len(bbs.bounding_boxes)):
            before = bbs.bounding_boxes[i]
            after = bbs_aug.bounding_boxes[i]
            print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                i,
                before.x1, before.y1, before.x2, before.y2,
                after.x1, after.y1, after.x2, after.y2)
                )'''

        # image with BBs before/after augmentation (shown below)
        #image_before = bbs.draw_on_image(image, size=2)
        #image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

        #plt.imshow(image_before)
        #plt.show()

        #plt.imshow(image_after)
        #plt.show()

        bnds = root.findall(".//bndbox")
        i = 0
        for bb in bnds:
            after = bbs_aug.bounding_boxes[i]
            xmin = bb.find("xmin")
            if after.x1 < 0:
                after.x1 = 1
            xmin.text = str(int(after.x1))

            ymin = bb.find("ymin")
            if after.y1 < 0:
                after.y1 = 1
            ymin.text = str(int(after.y1))

            xmax = bb.find("xmax")
            if after.x2 > int(wd.text):
                after.x2 = int(wd.text)-1
            xmax.text = str(int(after.x2))

            ymax = bb.find("ymax")
            if after.y2 > int(hg.text):
                after.y2 = int(hg.text)-1
            ymax.text = str(int(after.y2))
            i += 1

        j = imagename.text.find(".")
        ext = imagename.text[j:]
        img_to_save = Image.fromarray(image_aug)
        img_to_save.save(AUGMENTED_DIR_IMG + "/" + str(index) + ext)

        imagename.text = str(index) + ext
        tree.write(AUGMENTED_DIR_ANN + "/" + str(index) + ".xml")
        index += 1