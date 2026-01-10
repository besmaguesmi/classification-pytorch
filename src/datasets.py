import os

from PIL import Image
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, root_dir, transform=None, mode=None):
        """
        Args:
            root_dir (string): Directory with subdirectories for each class.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Auto-detect classes from subdirectories
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.mode = mode
        self.imgs = self._make_dataset()

    def _make_dataset(self):
        imgs = []

        valid_extensions = {'.png', '.jpg', '.jpeg'}  # Set of valid file extensions

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for fname in os.listdir(class_dir):
                if any(fname.lower().endswith(ext) for ext in valid_extensions):
                    path = os.path.join(class_dir, fname)
                    imgs.append((path, class_idx))

        return imgs

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Get the item at index `idx` from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple (image, label, filename)
        """
        # Fetch the image path and label
        img_path, label = self.imgs[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
