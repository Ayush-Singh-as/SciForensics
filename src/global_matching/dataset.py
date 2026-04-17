from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image

# from manipulations import RandomText
import manipulations


# See https://pytorch.org/docs/stable/torchvision/ for description of transforms
default_manipulations = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomPerspective(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.CenterCrop(128),
        manipulations.RandomText(p=0.5),
        manipulations.RandomRect(p=0.5),
        manipulations.RandomErase(p=0.25),
        transforms.ColorJitter(brightness=0.2),
        manipulations.JPEGCompression(quality_range=(30, 95), p=0.4),
        manipulations.GaussianNoise(std_range=(5, 25), p=0.3),
        manipulations.ColorShift(brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3), p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

default_transforms = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize(256),
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


class SimulatedDataset(Dataset):
    IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp")

    def __init__(
        self,
        image_dirs,
        transforms=default_transforms,
        manipulations=default_manipulations,
    ):
        if not isinstance(image_dirs, (list, tuple)):
            image_dirs = [image_dirs]
            
        self.fnames = []
        for d in image_dirs:
            root = Path(d)
            if not root.exists():
                continue
            
            # Use rglob so images nested inside subdirectories are found.
            # The BBBC038 dataset layout is: <hash>/images/<hash>.png,
            # so a flat glob("*.png") would return 0 files.
            for ext in self.IMAGE_EXTENSIONS:
                self.fnames.extend(root.rglob(ext))

        # Exclude mask directories (e.g. polimi_western_blots/*/mask/*)
        self.fnames = [f for f in self.fnames if "mask" not in f.parts]
        if len(self.fnames) == 0:
            raise FileNotFoundError(
                f"No images found under the provided directories: {image_dirs}. "
                "Check that the datasets have been downloaded and placed at the correct paths."
            )
        self.idx = np.arange(len(self))  # for random selection
        self.transforms = transforms
        self.manipulations = manipulations

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        anchor = Image.open(self.fnames[index]).convert("RGB")

        diff_idx = np.random.choice(self.idx[self.idx != index])
        diff = Image.open(self.fnames[diff_idx]).convert("RGB")

        return (
            self.transforms(anchor),
            self.manipulations(anchor),
            self.manipulations(diff),
        )


if __name__ == "__main__":
    dataset = SimulatedDataset("data/train/bbbc038")
    to_img = transforms.ToPILImage()

    n = 5
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(n, 3, gridspec_kw={"wspace": 0, "hspace": 0}, squeeze=True)
    for i in range(n):
        for j in range(3):
            ax[i, j].axis("off")
        og, man, sim = dataset[i]
        ax[i, 0].imshow(to_img(og), cmap="gray")
        ax[i, 1].imshow(to_img(man), cmap="gray")
        ax[i, 2].imshow(to_img(sim), cmap="gray")
    plt.tight_layout()
    plt.show()
