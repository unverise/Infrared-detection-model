"""Single-image visualization for RDANet predictions."""

from argparse import ArgumentParser
import os
import os.path as osp

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from model.rdanet import RDANet, ResidualBlock


DATASET_STATS = {
    'IRSTD-1k': ([0.343, 0.343, 0.343], [0.230, 0.230, 0.230]),
    'NUDT-SIRST': ([0.424, 0.424, 0.424], [0.218, 0.218, 0.218]),
    'NUAA-SIRST': ([0.449, 0.448, 0.449], [0.224, 0.224, 0.224]),
}

PGSM_CFG = {
    'IRSTD-1k': dict(
        mem_slots=[256, 128, 64, 32],
        patch=[8, 8, 8, 4],
        topk=[8, 8, 8, 4],
        drop_out=[0, 0, 0, 0.0],
        use_overlap=[False, False, False, False],
    ),
    'NUDT-SIRST': dict(
        mem_slots=[256, 128, 64, 32],
        patch=[16, 8, 8, 4],
        topk=[16, 8, 8, 4],
        drop_out=[0, 0, 0, 0.0],
        use_overlap=[False, False, False, False],
    ),
    'NUAA-SIRST': dict(
        mem_slots=[128, 64, 32, 16],
        patch=[16, 8, 8, 4],
        topk=[16, 8, 8, 4],
        drop_out=[0, 0, 0, 0.0],
        use_overlap=[False, False, False, False],
    ),
}


def parse_args():
    parser = ArgumentParser(description='Visualize RDANet on a single image')
    parser.add_argument('--dataset-name', type=str, default='IRSTD-1k',
                        choices=list(DATASET_STATS.keys()))
    parser.add_argument('--data-root', type=str, default='./datasets')
    parser.add_argument('--weight-path', type=str, default='./weights/IRSTD-1k.pkl')
    parser.add_argument('--image-name', type=str, default='XDU189',
                        help='image id without extension, e.g. XDU189')
    parser.add_argument('--image-path', type=str, default=None,
                        help='optional absolute/relative path to a custom image')
    parser.add_argument('--mask-path', type=str, default=None,
                        help='optional GT mask path; auto-resolved for dataset images')
    parser.add_argument('--base-size', type=int, default=512)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--factor', type=int, default=2)
    parser.add_argument('--block-factor', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='logit threshold; 0 matches evaluation (sigmoid>0.5)')
    parser.add_argument('--save-dir', type=str, default='./vis')
    return parser.parse_args()


def resolve_paths(args):
    if args.image_path is not None:
        img_path = args.image_path
        stem = osp.splitext(osp.basename(img_path))[0]
        mask_path = args.mask_path
    else:
        stem = args.image_name
        dataset_dir = osp.join(args.data_root, args.dataset_name)
        img_path = osp.join(dataset_dir, 'images', stem + '.png')
        mask_path = args.mask_path or osp.join(dataset_dir, 'masks', stem + '.png')
        if not osp.isfile(mask_path):
            mask_path = None
    if not osp.isfile(img_path):
        raise FileNotFoundError(f'Image not found: {img_path}')
    return img_path, mask_path, stem


def build_model(args, device):
    cfg = PGSM_CFG[args.dataset_name]
    model = RDANet(
        3,
        args.base_size,
        args.layers,
        args.factor,
        args.block_factor,
        block=ResidualBlock,
        **cfg,
    )
    try:
        state = torch.load(args.weight_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess(img_pil, dataset_name, base_size):
    mean, std = DATASET_STATS[dataset_name]
    img_resized = img_pil.resize((base_size, base_size), Image.BILINEAR)
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])(img_resized).unsqueeze(0)
    return img_resized, tensor


def overlay_mask(rgb, binary_mask, color=(255, 0, 0), alpha=0.55):
    out = rgb.copy()
    color_arr = np.array(color, dtype=np.float32)
    m = binary_mask.astype(bool)
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return out


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_path, mask_path, stem = resolve_paths(args)

    img_pil = Image.open(img_path).convert('RGB')
    img_show, tensor = preprocess(img_pil, args.dataset_name, args.base_size)

    model = build_model(args, device)
    with torch.no_grad():
        logits = model(tensor.to(device)).squeeze().cpu().numpy()

    prob = torch.sigmoid(torch.from_numpy(logits)).numpy()
    pred_bin = (logits > args.threshold).astype(np.uint8)

    rgb = np.array(img_show.convert('RGB'))
    pred_overlay = overlay_mask(rgb, pred_bin, color=(255, 0, 0), alpha=0.55)

    gt_bin = None
    if mask_path is not None and osp.isfile(mask_path):
        gt = Image.open(mask_path).resize((args.base_size, args.base_size), Image.NEAREST)
        gt_bin = (np.array(gt.convert('L')) > 0).astype(np.uint8)

    n_cols = 4 if gt_bin is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    axes = np.atleast_1d(axes)

    axes[0].imshow(rgb)
    axes[0].set_title('Input')
    axes[0].axis('off')

    col = 1
    if gt_bin is not None:
        axes[col].imshow(gt_bin, cmap='gray', vmin=0, vmax=1)
        axes[col].set_title('Ground Truth')
        axes[col].axis('off')
        col += 1

    axes[col].imshow(prob, cmap='hot', vmin=0, vmax=1)
    axes[col].set_title('Prediction (prob)')
    axes[col].axis('off')
    col += 1

    axes[col].imshow(pred_overlay)
    axes[col].set_title('Overlay (red=pred)')
    axes[col].axis('off')

    if gt_bin is not None:
        inter = (pred_bin * gt_bin).sum()
        union = ((pred_bin + gt_bin) > 0).sum()
        iou = float(inter) / float(union + 1e-8)
        fig.suptitle(f'{stem}  |  IoU={iou:.4f}', fontsize=12)
    else:
        fig.suptitle(stem, fontsize=12)

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = osp.join(args.save_dir, f'{stem}_vis.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # also save raw prediction mask
    pred_path = osp.join(args.save_dir, f'{stem}_pred.png')
    Image.fromarray((pred_bin * 255).astype(np.uint8)).save(pred_path)

    print(f'Saved visualization: {osp.abspath(out_path)}')
    print(f'Saved prediction mask: {osp.abspath(pred_path)}')
    if gt_bin is not None:
        print(f'Image IoU: {iou:.4f}')


if __name__ == '__main__':
    main()
