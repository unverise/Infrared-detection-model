from argparse import ArgumentParser
import bisect
import copy
import os
import os.path as osp
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import AdamW, Adagrad, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.rdanet import RDANet, ResidualBlock
from utils.data import IRSTDDataset
from utils.losses import AverageMeter, IoULoss
from utils.metric import PD_FA, ROCMetric, SamplewiseSigmoidMetric, SigmoidMetric


def parse_args():
    parser = ArgumentParser(description='Train or evaluate RDANet')

    parser.add_argument(
        '--dataset-name',
        type=str,
        default='NUAA-SIRST',
        choices=['IRSTD-1k', 'NUDT-SIRST', 'NUAA-SIRST'],
        help='dataset used for training or evaluation',
    )
    parser.add_argument('--data-root', type=str, default='./datasets',
                        help='root directory containing the dataset folders')
    parser.add_argument('--save-root', type=str, default='./runs',
                        help='directory used to save training outputs')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--eta-min', type=float, default=1e-5)

    parser.add_argument('--base-size', type=int, default=512)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--multi-gpus', action='store_true')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='training checkpoint used when resuming')

    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['AdamW', 'Adam', 'Adagrad'])

    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--factor', type=int, default=2)
    parser.add_argument('--block-factor', type=int, default=1)

    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test', 'fast-screen'])
    parser.add_argument('--weight-path', type=str, default=None,
                        help='checkpoint used in test mode')

    # ---- fast-screen / efficiency knobs ----
    parser.add_argument(
        '--screen-datasets',
        type=str,
        nargs='+',
        default=['IRSTD-1k', 'NUAA-SIRST'],
        help='datasets used by fast-screen mode',
    )
    parser.add_argument(
        '--subset-size',
        type=int,
        default=None,
        help='randomly keep at most N images for train and val (None = full set)',
    )
    parser.add_argument(
        '--subset-seed',
        type=int,
        default=42,
        help='seed for deterministic subset sampling across A/B runs',
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='run validation every N epochs (last epoch always evaluated)',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoader workers; use 0 only if debugging on Windows',
    )
    parser.add_argument(
        '--screen-epochs',
        type=int,
        default=300,
        help='epochs used when --mode fast-screen (override --epochs)',
    )
    parser.add_argument(
        '--screen-subset-size',
        type=int,
        default=150,
        help='per-split sample size in fast-screen (capped by available images)',
    )

    args = parser.parse_args()
    return args


def apply_fast_screen_defaults(args):
    """Configure a short A/B screening protocol without touching full-train defaults."""
    args.mode = 'train'
    args.epochs = args.screen_epochs
    args.subset_size = args.screen_subset_size
    if args.eval_interval < 5:
        args.eval_interval = 5
    # Keep cosine schedule in sync with the shorter run.
    args._scheduler_tmax = min(600, max(1, args.epochs))
    print(
        '[fast-screen] datasets={}, subset<= {}, epochs={}, eval_interval={}, seed={}'.format(
            args.screen_datasets,
            args.subset_size,
            args.epochs,
            args.eval_interval,
            args.subset_seed,
        )
    )
    print(
        '[fast-screen] Note: NUAA-SIRST test has only 86 images; '
        'val subset will use min(requested, available).'
    )
    return args


def irstd_collate(batch):
    """Stack image/mask; keep original-size `temp` as a tuple (sizes may differ)."""
    imgs, masks, temps = zip(*batch)
    return torch.stack(imgs, 0), torch.stack(masks, 0), temps


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.mode = args.mode

        valset = IRSTDDataset(args, mode='val')

        if self.mode == 'train':
            trainset = IRSTDDataset(args, mode='train')
            self.train_loader = Data.DataLoader(
                trainset,
                args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                pin_memory=True,
                collate_fn=irstd_collate,
            )
            self.val_loader = Data.DataLoader(
                valset,
                args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=irstd_collate,
            )
            print(
                f'[{args.dataset_name}] train={len(trainset)} val={len(valset)} '
                f'batch={args.batch_size}'
            )
        else:
            self.val_loader = Data.DataLoader(
                valset,
                1,
                num_workers=args.num_workers,
                shuffle=False,
                collate_fn=irstd_collate,
            )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if device.type != 'cuda':
            print('WARNING: CUDA not available, training on CPU will be very slow')

        if args.dataset_name == 'IRSTD-1k':
            mem_slots = [256, 128, 64, 32]
            patch = [8, 8, 8, 4]
            topk = [8, 8, 8, 4]
            drop_out = [0, 0, 0, 0.0]
            use_overlap = [False, False, False, False]
        elif args.dataset_name == 'NUDT-SIRST':
            mem_slots = [256, 128, 64, 32]
            patch = [16, 8, 8, 4]
            topk = [16, 8, 8, 4]
            drop_out = [0, 0, 0, 0.0]
            use_overlap = [False, False, False, False]
        elif args.dataset_name == 'NUAA-SIRST':
            mem_slots = [128, 64, 32, 16]
            patch = [16, 8, 8, 4]
            topk = [16, 8, 8, 4]
            drop_out = [0, 0, 0, 0.0]
            use_overlap = [False, False, False, False]
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset_name}")

        model = RDANet(
            3,
            args.base_size,
            args.layers,
            args.factor,
            args.block_factor,
            block=ResidualBlock,
            mem_slots=mem_slots,
            patch=patch,
            topk=topk,
            drop_out=drop_out,
            use_overlap=use_overlap,
        )
        model.to(device)

        if args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use ' + str(torch.cuda.device_count()) + ' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        print('use ' + str(torch.cuda.device_count()) + ' gpus')

        self.model = model

        if args.optimizer == 'Adagrad':
            self.optimizer = Adagrad(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.06
            )
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == 'Adam':
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
            )

        tmax = getattr(args, '_scheduler_tmax', 600)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=tmax, eta_min=args.eta_min
        )
        self.scheduler_tmax = tmax

        self.loss_fun = IoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = SigmoidMetric()
        self.nIoU = SamplewiseSigmoidMetric(1)
        self.ROC = ROCMetric(1, 10)
        self.best_iou = 0

        self.BIN_LABELS = ['[1,15]', '[16,31]', '[32,63]', '[64,127]', '[128,255]', '[256,∞)']
        self.AREA_EDGES = [15, 31, 63, 127, 255]

        self.bin_metrics = [SigmoidMetric() for _ in range(6)]
        self.bin_counts = [0 for _ in range(6)]

        if args.mode == 'train':
            if args.checkpoint_path is not None:
                checkpoint_path = osp.abspath(args.checkpoint_path)
                check_folder = osp.dirname(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_iou = checkpoint['iou']
                self.save_folder = check_folder
            else:
                run_name = args.dataset_name + '-%s' % (
                    time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                )
                self.save_folder = osp.join(args.save_root, run_name)
                os.makedirs(self.save_folder, exist_ok=False)
                dict_args = vars(args)
                args_key = list(dict_args.keys())
                args_value = list(dict_args.values())
                with open(self.save_folder + '/train_log.txt', 'w') as f:
                    now = datetime.now()
                    f.write("time:--")
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    f.write(dt_string)
                    f.write('\n')
                    for i in range(len(args_key)):
                        f.write(args_key[i])
                        f.write(':--')
                        f.write(str(args_value[i]))
                        f.write('\n')
            self.writer = SummaryWriter(log_dir=osp.join(self.save_folder, 'tensorboard'))
        if args.mode == 'test':
            if args.weight_path is None:
                raise ValueError('--weight-path is required in test mode')
            try:
                state_dict = torch.load(
                    args.weight_path, map_location=self.device, weights_only=True
                )
            except TypeError:
                state_dict = torch.load(args.weight_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()

        for i, (data, mask, temp) in enumerate(tbar):
            data = data.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            pred = self.model(data)
            loss = self.loss_fun(pred, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), data.size(0))
            tbar.set_description(
                'Epoch %d, lr %.6f, loss %.4f'
                % (epoch, self.optimizer.param_groups[0]['lr'], losses.avg)
            )
        self.writer.add_scalar('Loss/loss', losses.avg, epoch)

        if epoch < self.scheduler_tmax:
            self.scheduler.step()

    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.nIoU.reset()

        self.PD_FA.reset()
        self.ROC.reset()
        tbar = tqdm(self.val_loader)
        count = 0
        with torch.no_grad():
            for i, (data, mask, temp) in enumerate(tbar):
                data = data.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                pred = self.model(data)
                count = count + data.size(0)

                pred = pred.cpu()
                mask = mask.cpu()
                self.mIoU.update(pred, mask)
                self.nIoU.update(pred, mask)

                if self.mode == 'test':
                    from scipy.ndimage import label

                    # collate keeps temps as a tuple (HxW may differ across images)
                    temp_list = temp if isinstance(temp, (list, tuple)) else [temp]
                    for bi, t in enumerate(temp_list):
                        if torch.is_tensor(t):
                            t_np = t.detach().cpu().numpy().squeeze()
                        else:
                            t_np = np.asarray(t).squeeze()
                        lbl, num = label(t_np)
                        if num == 0:
                            area = 0
                        else:
                            cnt = np.bincount(lbl.ravel())[1:]
                            area = int(cnt.max())
                        if area >= 1:
                            idx = bisect.bisect_right(self.AREA_EDGES, area)
                            self.bin_metrics[idx].update(pred[bi : bi + 1], mask[bi : bi + 1])
                            self.bin_counts[idx] += 1

                    self.PD_FA.update(pred, mask)
                    self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()
                _, n_IoU = self.nIoU.get()

                tbar.set_description(
                    'Epoch %d, IoU %.4f, nIoU %.4f' % (epoch, mean_IoU, n_IoU)
                )

            _, mean_IoU = self.mIoU.get()
            _, n_IoU = self.nIoU.get()

            if self.mode == 'train':
                iou = mean_IoU
                self.writer.add_scalar('IoU/mean', mean_IoU, epoch)
                self.writer.add_scalar('IoU/nIoU', n_IoU, epoch)
                if iou > self.best_iou:
                    self.best_iou = iou
                    all_states = {
                        "net": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        "epoch": epoch,
                        "iou": self.best_iou,
                    }
                    torch.save(self.model.state_dict(), self.save_folder + '/weight.pkl')
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write(
                            '{} - {:04d}\t - IoU {:.4f}\t - nIoU {:.4f}\n'.format(
                                time.strftime(
                                    '%Y-%m-%d-%H-%M-%S', time.localtime(time.time())
                                ),
                                epoch,
                                mean_IoU,
                                n_IoU,
                            )
                        )
                    torch.save(all_states, self.save_folder + '/best_checkpoint.pkl')
                # Save resume checkpoint less often during screening.
                interval = max(1, int(self.args.eval_interval))
                if epoch % interval == 0 or epoch + 1 >= self.args.epochs:
                    all_states = {
                        "net": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        "epoch": epoch,
                        "iou": self.best_iou,
                    }
                    torch.save(all_states, self.save_folder + '/checkpoint.pkl')

            elif self.mode == 'test':
                FA, PD = self.PD_FA.get(count)
                true_positive_rate, false_positive_rate, _, _ = self.ROC.get()
                print('mIoU: ' + str(mean_IoU) + '\n')
                print('nIoU:', n_IoU)
                print('Pd: ' + str(PD[0]) + '\n')
                print('Fa: ' + str(FA[0] * 1000000) + '\n')
                print('tp:', true_positive_rate)
                print('fp:', false_positive_rate)
                print('\n# Bin-wise IoU:')
                for k, metric in enumerate(self.bin_metrics):
                    _, mIoU_k = metric.get()
                    print(
                        f'{self.BIN_LABELS[k]:>8} | num={self.bin_counts[k]:4d} | '
                        f'mIoU={float(mIoU_k):.4f}'
                    )


def run_training(args):
    trainer = Trainer(args)
    for epoch in range(trainer.start_epoch, args.epochs):
        trainer.train(epoch)
        do_eval = (
            ((epoch + 1) % max(1, args.eval_interval) == 0)
            or (epoch + 1 == args.epochs)
            or (epoch == trainer.start_epoch)
        )
        if do_eval:
            trainer.test(epoch)
    trainer.writer.close()
    print(
        f'[{args.dataset_name}] done. best IoU={trainer.best_iou:.4f} '
        f'save={trainer.save_folder}'
    )
    return trainer.best_iou, trainer.save_folder


def run_fast_screen(args):
    args = apply_fast_screen_defaults(args)
    screen_root = osp.join(
        args.save_root,
        'fast-screen-%s' % time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
    )
    os.makedirs(screen_root, exist_ok=True)
    summary_path = osp.join(screen_root, 'screen_summary.txt')

    results = []
    with open(summary_path, 'w') as f:
        f.write('fast-screen summary\n')
        f.write(f'subset_size={args.subset_size} seed={args.subset_seed}\n')
        f.write(f'epochs={args.epochs} eval_interval={args.eval_interval}\n\n')

    for name in args.screen_datasets:
        if name not in ['IRSTD-1k', 'NUDT-SIRST', 'NUAA-SIRST']:
            raise ValueError(f'Unsupported screen dataset: {name}')
        ds_args = copy.copy(args)
        ds_args.dataset_name = name
        ds_args.save_root = screen_root
        ds_args.checkpoint_path = None
        print('\n' + '=' * 60)
        print(f'[fast-screen] start dataset={name}')
        print('=' * 60)
        best_iou, save_folder = run_training(ds_args)
        results.append((name, best_iou, save_folder))
        with open(summary_path, 'a') as f:
            f.write(f'{name}\tbest_IoU={best_iou:.4f}\t{save_folder}\n')

    print('\n[fast-screen] finished:')
    for name, best_iou, save_folder in results:
        print(f'  {name}: best IoU={best_iou:.4f} -> {save_folder}')
    print(f'summary: {summary_path}')


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'fast-screen':
        run_fast_screen(args)
    elif args.mode == 'train':
        if not hasattr(args, '_scheduler_tmax'):
            args._scheduler_tmax = 600
        run_training(args)
    else:
        trainer = Trainer(args)
        trainer.test(1)
