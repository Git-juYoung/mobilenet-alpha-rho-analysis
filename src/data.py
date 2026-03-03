from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets


def build_datasets(data_root, train_tf, eval_tf):
    data_root = Path(data_root)

    train_set = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_full = datasets.ImageFolder(data_root / "val", transform=eval_tf)

    return train_set, val_full


def split_val_test(val_full):
    val_len = len(val_full) // 2
    test_len = len(val_full) - val_len

    val_set, test_set = random_split(val_full, [val_len, test_len])
    return val_set, test_set


def build_train_val_dataloaders(train_dataset, val_dataset, batch_size, num_workers, pin_memory):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_test_dataloader(test_dataset, batch_size, num_workers, pin_memory):
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader