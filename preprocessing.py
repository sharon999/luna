import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
from scipy.ndimage import zoom, rotate
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import warnings
import pandas as pd

DATA_DIR = None
LABEL_FOLDER = None
AGGREGATED_BBOX_FILE = '/content/drive/My Drive/Luna16/prep_subset1/aggregated_bboxes.npy'

# פונקציות נוספות שהיו קיימות...

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import warnings

class DataBowl3Detector(Dataset):
    def __init__(self, split, config, phase='train', anchors=None):
        """
        :param split: רשימת קבצים מהספליט
        :param config: קובץ קונפיגורציה
        :param phase: שלב (train, val, test)
        :param anchors: רשימת Anchors לשימוש
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        if phase == 'train':
            base_folder = config['datadir_train']
        elif phase == 'val':
            base_folder = config['datadir_val']
        else:
            base_folder = config['datadir_test'] if 'datadir_test' in config else config['datadir_val']

        self.filenames = [os.path.join(base_folder, f'{idx}_cropped.npy') for idx in split]

        self.sample_bboxes = []
        self.input_size = config['crop_size']  # גודל קלט מהקונפיגורציה
        self.anchors = anchors  # שמירת ה-anchors ב-Init

        for idx in split:
            label_path = os.path.join(base_folder, f'{idx}_label.npy')

            if os.path.exists(label_path):
                try:
                    bboxes = np.load(label_path)
                except Exception as e:
                    print(f"❌ INIT: failed to load {label_path}: {e}")
                self.sample_bboxes.append(bboxes)
            else:
                self.sample_bboxes.append([])

    def normalize_labels(self, labels):
        """
        מנרמל את ה-Labels לקואורדינטות יחסיות לגודל הקלט.
        :param labels: מערך התוויות
        :return: תוויות מנורמלות
        """
        if labels.size == 0:
            return labels  # אין צורך לנרמל אם אין תוויות
        normalized_labels = labels.copy()
        normalized_labels[:, 1:4] /= np.array(self.input_size)  # קואורדינטות יחסיות לגודל הקלט
        normalized_labels[:, 4] /= max(self.input_size)  # נורמליזציה של הקוטר לגודל המקסימלי של הקלט
        return normalized_labels

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        try:
            imgs = np.load(filename)
        except Exception as e:
            raise RuntimeError(f"❌ CROP load failed for {filename}: {e}")


        if len(imgs.shape) != 3:
            raise ValueError(f"Expected 3D image [depth, height, width], but got {imgs.shape}")

        label_path = filename.replace('_cropped.npy', '_label.npy')
        #print(filename)
        if os.path.exists(label_path):
            labels = np.load(label_path)
            #print("yeslabel")
        else:
            labels = np.array([])  # הגדר תוויות ריקות כ-array ריק
            #print("nollabel")
        # טיפול במקרה של תוויות ריקות
        if labels.size == 0:
            #print("changelempty")
            # תוויות ריקות (אין נודול) – ערך ברירת מחדל עם תווית שלילית
            labels = np.zeros((1, 5))  # צורה לדוגמה: (1, [is_nodule, X, Y, Z, size])
        #print(f"Original Labels (Before Normalization) for {filename}: {labels}")

        # נורמליזציה ל-labels
        #normalized_labels = self.normalize_labels(labels)
        normalized_labels = labels

        #print(f"Normalized Labels for {filename}: {normalized_labels}")

        # נורמליזציה לנתוני התמונה
        #sample = (imgs.astype(np.float32) - 128) / 128
        #print(f"Image Min: {sample.min()}, Image Max: {sample.max()}")

        #sample = torch.from_numpy(sample[np.newaxis, ...])
        #print(f"📢 Dataset - Image Min: {imgs.min()}, Max: {imgs.max()}, Mean: {imgs.mean()}")

        sample = torch.from_numpy(imgs[np.newaxis, ...].astype(np.float32))

        updated_bboxes_tensor = torch.from_numpy(normalized_labels)

        return sample, normalized_labels, updated_bboxes_tensor,filename

    def __len__(self):
        return len(self.filenames)

