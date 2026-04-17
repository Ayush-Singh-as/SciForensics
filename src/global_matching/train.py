import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from dataset import SimulatedDataset
from model import Model, triplet_loss, triplet_acc

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="max number of epochs")
parser.add_argument(
    "--patience",
    type=int,
    default=50,
    help="number of epochs without improvement before stopping",
)
parser.add_argument("--bs", type=int, default=128, help="size of batches")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument(
    "--train_dir",
    type=str,
    default="data/train",
    help="training data root (auto-discovers all sub-dataset folders)",
)
parser.add_argument(
    "--valid_dir",
    type=str,
    default="data/valid/bbbc038",
    help="validation data location",
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_combined_dataset(root_dir: str) -> ConcatDataset:
    """Auto-discover all sub-dataset directories under root_dir and combine
    them into a single ConcatDataset.  Empty directories are skipped
    gracefully."""
    root = Path(root_dir)
    datasets = []

    # If root_dir itself contains images (not just sub-folders), treat it as a
    # single dataset directory.
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not subdirs:
        subdirs = [root]

    for subdir in subdirs:
        try:
            ds = SimulatedDataset(str(subdir))
            print(f"  ✓ {subdir.name}: {len(ds)} images")
            datasets.append(ds)
        except FileNotFoundError:
            print(f"  ✗ {subdir.name}: empty or no images, skipping")

    if not datasets:
        raise FileNotFoundError(
            f"No usable datasets found under '{root_dir}'. "
            "Ensure at least one sub-directory contains images."
        )

    combined = ConcatDataset(datasets)
    print(f"  Combined training set: {len(combined)} images total")
    return combined


def train(opt):
    writer = SummaryWriter()  # tensorboard

    print(f"Loading training data from '{opt.train_dir}'...")
    train_dataset = build_combined_dataset(opt.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=opt.bs, shuffle=True)

    print(f"Loading validation data from '{opt.valid_dir}'...")
    valid_loader = DataLoader(SimulatedDataset(opt.valid_dir), batch_size=opt.bs)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    counter = 0  # epochs since improvement
    best_loss = float("inf")

    print("train_loss\tvalid_loss\tvalid_acc")

    for epoch in range(opt.n_epochs):
        model.train()
        train_loss = 0.0
        size = len(train_loader.dataset)

        # train step
        for i, (anchor_imgs, same_imgs, diff_imgs) in enumerate(train_loader):
            optimizer.zero_grad()
            anchor = model(anchor_imgs.to(device))
            same = model(same_imgs.to(device))
            diff = model(diff_imgs.to(device))

            loss = triplet_loss(anchor, same, diff)
            train_loss += loss.item() * anchor.size(0)
            loss.backward()  # backprop
            optimizer.step()
        train_loss /= size

        # validation step
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            total_accuracy = 0.0
            size = len(valid_loader.dataset)
            for i, (anchor_imgs, same_imgs, diff_imgs) in enumerate(valid_loader):
                anchor = model(anchor_imgs.to(device))
                same = model(same_imgs.to(device))
                diff = model(diff_imgs.to(device))

                loss = triplet_loss(anchor, same, diff)
                valid_loss += loss.item() * anchor.size(0)
                total_accuracy += triplet_acc(anchor, same, diff) * anchor.size(0)
        valid_loss /= size
        total_accuracy /= size

        print(f"{train_loss:.3f}\t{valid_loss:.3f}\t{total_accuracy:.4f}")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/valid", valid_loss, epoch)
        writer.add_scalar("acc/valid", total_accuracy, epoch)
        writer.flush()

        # early stopping
        if valid_loss < best_loss:
            counter = 0
            print("new best loss, saving checkpoint...")
            torch.save(model.state_dict(), f"models/checkpoint_{epoch}.pth")
            torch.save(model.state_dict(), f"models/weights.pth")
            best_loss = valid_loss
        else:
            counter += 1

        if counter > opt.patience:
            print(f"{opt.patience} epochs without improvement, exiting")
            break


if __name__ == "__main__":
    opt = parser.parse_args()
    train(opt)
