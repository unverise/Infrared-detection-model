import os
import random


def _read_id_list(txt_path):
    ids = []
    with open(txt_path, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                ids.append(name)
    return ids


def _resolve_split_txt(root, dataset, split_method):
    """
    Support both layouts:
      - DNANet original:  <root>/<dataset>/<split_method>/train.txt|test.txt
      - RDANet flat:      <root>/<dataset>/trainval.txt|test.txt
    """
    dataset_dir = os.path.join(root, dataset)
    flat_train = os.path.join(dataset_dir, 'trainval.txt')
    flat_test = os.path.join(dataset_dir, 'test.txt')
    nested_train = os.path.join(dataset_dir, split_method, 'train.txt')
    nested_test = os.path.join(dataset_dir, split_method, 'test.txt')

    if os.path.isfile(flat_train) and os.path.isfile(flat_test):
        return flat_train, flat_test
    if os.path.isfile(nested_train) and os.path.isfile(nested_test):
        return nested_train, nested_test
    raise FileNotFoundError(
        'Cannot find split files under {}. Tried flat trainval/test and '
        '{}/train|test.'.format(dataset_dir, split_method)
    )


def subsample_ids(ids, max_samples, seed, split):
    if max_samples is None or max_samples <= 0 or max_samples >= len(ids):
        return ids
    rng = random.Random(int(seed) + (0 if split == 'train' else 1))
    picked = rng.sample(ids, max_samples)
    picked.sort()
    return picked


def load_dataset(root, dataset, split_method, subset_size=0, subset_seed=42):
    train_txt, test_txt = _resolve_split_txt(root, dataset, split_method)
    train_img_ids = _read_id_list(train_txt)
    val_img_ids = _read_id_list(test_txt)

    before_train, before_val = len(train_img_ids), len(val_img_ids)
    train_img_ids = subsample_ids(train_img_ids, subset_size, subset_seed, 'train')
    val_img_ids = subsample_ids(val_img_ids, subset_size, subset_seed, 'test')
    if len(train_img_ids) < before_train or len(val_img_ids) < before_val:
        print(
            '[{}] subset train {}/{} | test {}/{} (seed={})'.format(
                dataset,
                len(train_img_ids), before_train,
                len(val_img_ids), before_val,
                subset_seed,
            )
        )
    return train_img_ids, val_img_ids, test_txt


def load_param(channel_size, backbone):
    if channel_size == 'one':
        nb_filter = [4, 8, 16, 32, 64]
    elif channel_size == 'two':
        nb_filter = [8, 16, 32, 64, 128]
    elif channel_size == 'three':
        nb_filter = [16, 32, 64, 128, 256]
    elif channel_size == 'four':
        nb_filter = [32, 64, 128, 256, 512]

    if   backbone == 'resnet_10':
        num_blocks = [1, 1, 1, 1]
    elif backbone == 'resnet_18':
        num_blocks = [2, 2, 2, 2]
    elif backbone == 'resnet_34':
        num_blocks = [3, 4, 6, 3]
    elif backbone == 'vgg_10':
        num_blocks = [1, 1, 1, 1]
    return nb_filter, num_blocks