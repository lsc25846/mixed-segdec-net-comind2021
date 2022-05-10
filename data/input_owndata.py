import numpy as np
import pickle
import os
from data.dataset import Dataset
from config import Config
'''
    問題:需要有沒有標記的樣本，不然程式會出錯(dataset.py line 49)
    可能需要重新標記
    無標記樣本需要有一定數量程式才能動
'''
class OwnDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(OwnDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_contents(self):
        pos_samples, neg_samples = [], []

        #data_points = read_split(self.cfg.NUM_SEGMENTED, self.kind)
        """
        讀取資料夾下的所有影像
        """
        datalist=os.listdir(os.path.join(self.path,self.kind.lower()))
        for part in datalist:
            #讀取資料夾中所有檔案
            if "GT" not in part:
            #如果不是GT，抓取影像
                part=part.replace('.png','')
                image_path = os.path.join(self.path, self.kind.lower(), f"{part}.png")
                seg_mask_path = os.path.join(self.path, self.kind.lower(), f"{part}_GT.png")
            else:
            #如果是GT，跳過
                continue
            '''
            imagesize需要去config裡面新增
            '''
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)
            is_segmented=True
            if positive:
                image = self.to_tensor(image)
                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                pos_samples.append((image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, part))
            else:
                image = self.to_tensor(image)
                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, part))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)
        print("pos,neg",self.num_pos,self.num_neg)
        self.init_extra()
