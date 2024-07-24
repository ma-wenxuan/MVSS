import torch.cuda
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class SAM(object):
    def __init__(self):
        super(SAM).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.sam = sam_model_registry["vit_h"](checkpoint="/home/mwx/Downloads/sam_vit_h_4b8939.pth").to(self.device)
        # self.sam = sam_model_registry["vit_l"](checkpoint="/home/mwx/Downloads/sam_vit_l_0b3195.pth").to(self.device)
        self.sam = sam_model_registry["vit_b"](checkpoint="/home/mwx/Downloads/sam_vit_b_01ec64.pth").to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(model=self.sam)
        # self.mask_generator = SamAutomaticMaskGenerator(model=self.sam,
        #                                                 points_per_side=32,
        #                                                 pred_iou_thresh=0.88,
        #                                                 stability_score_thresh=0.95,
        #                                                 crop_n_layers=0,
        #                                                 crop_n_points_downscale_factor=1,
        #                                                 min_mask_region_area=5000,)

    def segment(self, image):
        masks = self.mask_generator.generate(image)
        # plt.figure(figsize=(20, 20))
        # plt.imshow(image)
        # seg = self.show_anns(masks)
        # plt.axis('off')
        # plt.imshow(seg)
        # plt.show()
        return self.seg2label(masks)
        # return seg


    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
        return img

    def seg2label(self, masks):
        label = np.zeros((480, 640))
        masks.sort(key=lambda x: x['area'], reverse=True)
        for i, mask in enumerate(masks):
            label[mask['segmentation']] = i
        return torch.from_numpy(label).unsqueeze(0)


if __name__ == "__main__":
    model = SAM()
    image = np.array(Image.open('image_66.png'))
    masks = model.segment(image=image)
    plt.imshow(masks[0])
    plt.show()