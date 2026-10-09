from model.utils import *

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='DNANet',
                        help='model name: DNANet')

    # parameter for DNANet
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--deep_supervision', type=str, default='True', help='True or False (model==DNANet)')


    # data and pre-process
    parser.add_argument('--dataset', type=str, default='NUDT-SIRST',
                        help='dataset name: NUDT-SIRST, NUAA-SIRST, IRSTD-1k, SIRST-V2, NUST-SIRST')
    parser.add_argument('--st_model', type=str, default='NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS',
                        help='result folder name under ./result/, e.g. NUAA-SIRST_DNANet_xx_wDS')
    parser.add_argument('--model_dir', type=str,
                        default = 'NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar',
                        help    = 'checkpoint path relative to ./result/')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--root', type=str, default='../RDANet-main/datasets',
                        help='dataset root that contains <dataset>/images|masks|*.txt')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='50_50, 10000_100(for NUST-SIRST); ignored when trainval.txt exists')
    parser.add_argument('--subset_size', type=int, default=0,
                        help='if >0, only evaluate this many test images')
    parser.add_argument('--subset_seed', type=int, default=42,
                        help='random seed used by --subset_size')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train (default: 110)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')


    args = parser.parse_args()

    # the parser
    return args