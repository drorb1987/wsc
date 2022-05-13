import torch
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import os

LABELS_TO_NUMS = {'Angle': 0,
                  'Box': 1,
                  'Circle': 2,
                  'Closeup': 3,
                  'Crowd': 4,
                  'Other': 5}

NUMS_TO_LABELS = {val: key for key, val in LABELS_TO_NUMS.items()}

class WscDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.img_dir, self.img_labels.label.iloc[index], self.img_labels.image.iloc[index])
        image = read_image(img_path)
        label = self.img_labels.label.iloc[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        return image, label


def make_csv():
    """
    Check if csv files of the data are existed, if not create them
    """
    for tt in ["train", "test"]:
        annotations_file = f"{tt}\\labeled_data.csv"
        if not os.path.exists(annotations_file):
            imgs = glob.glob(f"{tt}\*\*")
            labels = []
            images = []
            for img in imgs:
                parse_img = img.split('\\')
                labels.append(parse_img[1])
                images.append(parse_img[2])
            df = pd.DataFrame({"label": labels, "image": images})
            df.to_csv(annotations_file)

def main():
    make_csv()
    training_data = WscDataset('train/labeled_data.csv', 'train', transform=transforms.Resize([720, 1280]))
    validation_data = WscDataset('test/labeled_data.csv', 'test', transform=transforms.Resize([720, 1280]))
    training_loader = DataLoader(dataset=training_data, batch_size=4, shuffle=True, num_workers=0)
    validation_loader = DataLoader(dataset=validation_data, batch_size=4, shuffle=False, num_workers=0)



if __name__ == '__main__':
    main()
