import os, json

import cv2
import numpy as np

from torch.utils.data import Dataset


class Img2DesignDataset(Dataset):

    def __init__(self, path: str, split: str):
        self.data = []
        self.data_path = os.path.join(path, split)
        with open(os.path.join(self.data_path, 'metadata.jsonl'), 'rt') as f:
            for line in f:
                sample = json.loads(line.strip())
                if sample['tgt_file_name'] in {
                    "0c4f4c36-7b0a-496c-b170-340929f5b7ab.source.16x9.zh-Hans-6-thumbnail.png",
                    "fc57077c-be39-4e37-b8fd-e692edd90c19.source.16x9.zh-Hans-4-thumbnail.png"
                }:
                    continue
                self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['src_file_name']
        target_filename = item['tgt_file_name']
        prompt = 'a high-quality presentation slide background image, with ' + item['decoration_prompt'] + ' decoration'

        source = self.load_rgb_image_cv2(os.path.join(self.data_path, source_filename))
        target = self.load_rgb_image_cv2(os.path.join(self.data_path, target_filename))
        if source is None or target is None:
            print(item['tgt_file_name'])
            return None

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.resize(source, (512, 512,), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (512, 512,), interpolation=cv2.INTER_AREA)

        # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0
        # TODO: Consider to perform data augmentation (horizontal flip, random center crop). This can be important for avoiding overfiting
        source = (source.astype(np.float32) / 127.5) - 1.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        return dict(jpg=target, txt=prompt, hint=source, name=target_filename)
    
    def load_rgb_image_cv2(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2: # Grayscale image
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3: # RGB image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif img.shape[2] == 4: # RGBA image
            alpha_channel = img[:, :, 3]
            rgb_channels = img[:, :, :3]
            white_background = np.full_like(rgb_channels, 255)
            alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
            alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)
            img = alpha_factor * rgb_channels + (1 - alpha_factor) * white_background
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            raise ValueError("Input image must be either grayscale (2D) or RGB (3D) or RGBA (4D).")


if __name__ == '__main__':
    dataset = Img2DesignDataset('/data/text2design/preprocess/img2design/theme_v1.0', 'test')
    num = 0
    for item in dataset:
        num += 1
    print(num)
