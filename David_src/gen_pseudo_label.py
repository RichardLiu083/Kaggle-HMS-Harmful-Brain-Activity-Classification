import numpy as np
import torch
import torch.nn as nn
import model
import data_loader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import os
from utils import check_path, seed_everything, check_mlflow_server, get_transform, sliding_pred
import mlflow
from trainer import Trainer
from copy import deepcopy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for kaggle HMS dataset.')
    # dataset
    parser.add_argument('--path_dataset', type=str, default='G:\dataset\HMS', help='Path to the dataset')
    parser.add_argument('--name_fold_csv', type=str, default='train_spectrogram.csv', help='Path to Richard\'s csv file')
    parser.add_argument('--name_csv', type=str, default='train.csv', help='Path to dataset training csv file')
    parser.add_argument('--length-process', type=str, default='slide', choices=['random_crop', 'first', 'slide', 'resize'],
                        help='The way to process the spectrogram\'s length(width)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loader')
    parser.add_argument('--cat-type', type=str, default='x', choices=['x', 'ch', 'y'],
                        help="The way to concat ['LL', 'RL', 'RP', 'LP'] images, default as x.")
    parser.add_argument('--crop-size_spec', nargs=2, type=int, default=[100, 300], help='Size of the cropped spectrogram')
    parser.add_argument('--crop-size_eeg', nargs=2, type=int, default=[50, 256], help='Size of the cropped eeg spectrogram')
    parser.add_argument('--slide_wd_ratio', type=float, default=0.5, help='The ratio of sliding window')
    parser.add_argument('--fold-num', type=int, default=0, help='The number of i-th fold for validation')
    parser.add_argument('--network-type', type=str, default='cnn', choices=['cnn', 'transformer'],
                        help='The type of network')
    parser.add_argument('--cnn-type', type=str, default='b0', choices=['b0', 'v2s', 'mobilenet', 'convnext-tiny'],
                        help='The type of CNN')
    parser.add_argument('--use-mean', action='store_true', help='Use mean to aggregate the sliding window prediction and original gt')
    parser.add_argument('--gen-all', action='store_true', help='Generate pseudo label for all')
    # dataset cache
    parser.add_argument('--cache_dir', type=str, default='cache', help='Path to the cache directory')
    # network
    parser.add_argument('--use-attention', action='store_true', help='Use attention in the network')
    parser.add_argument('--use-mask', action='store_true', help='Use mask in the input of network')
    # training
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--path-weights', type=str, default='proj/cnn-b0-f0/weights/best.pt', help='Path to the weight file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    crop_size_sp = args.crop_size_spec
    crop_size_eeg = args.crop_size_eeg
    seed_everything(9527)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dataset
    dataset_kwargs = {
        'dataset_path': args.path_dataset,
        'fold_csv': args.name_fold_csv,
        'csv_path': args.name_csv,
        'cache_dir': args.cache_dir,
        'cache': True,
        'flush_cache': False,
        'cat_type': args.cat_type,
        'slide_wd_ratio': args.slide_wd_ratio,
        'crop_size_sp': crop_size_sp,
        'crop_size_eeg': crop_size_eeg,
        'mixin_ratio': 0,
        'mask_w_ratio': 0,
        'mask_w_prob': 0,
        'mask_h_ratio': 0,
        'mask_h_prob': 0,
        'valid_fold_num': args.fold_num,
        'transform_sp': get_transform(crop_size_sp, args.length_process, None),
        'transform_eeg': get_transform(crop_size_eeg, args.length_process, None),
    }
    if not args.gen_all:
        valid_dataset_greater = data_loader.LabelHMSDataset(**dataset_kwargs, is_train=False, filter_vote=True, filter_vote_mode='drop_less')
        valid_dataset_less = data_loader.LabelHMSDataset(**dataset_kwargs, is_train=False, filter_vote=True, filter_vote_mode='drop_greater')
    else:
        valid_dataset_greater = None
        valid_dataset_less = data_loader.LabelHMSDataset(**dataset_kwargs, is_train=False, filter_vote=False)
    train_dataset = data_loader.LabelHMSDataset(**dataset_kwargs, is_train=True)
    # Data loader
    valid_loader = DataLoader(valid_dataset_less, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=0, collate_fn=valid_dataset_less.collate_fn, pin_memory=True)
    # Model
    in_ch = 1
    if args.cat_type == 'ch':
        in_ch = 4
    if args.use_mask:
        in_ch += 1

    if args.network_type == 'cnn':
        width = crop_size_sp[1] + crop_size_eeg[1] * 2
        model = model.EfficientNet(width=width, use_attention=args.use_attention, in_ch=in_ch,
                                   num_classes=train_dataset.num_class, cnn_type=args.cnn_type).to(device)
    elif args.network_type == 'transformer':
        model = model.EfficientTransNet(in_ch=in_ch).to(device)
    else:
        raise NotImplementedError(f'Network type: {args.network_type} not implemented')

    # Initialize model
    trainer = Trainer(
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
        train_loader=valid_loader,
        valid_loader=valid_loader,
        use_ema=True,
        loss_type='CE_only',
        args=args,
    )
    # Resume
    trainer.load(args.path_weights)
    model = trainer.ema.ema_model
    model.eval()

    loader = tqdm(valid_loader)
    predictions = {}
    for batch in loader:
        img, label, soft_label = trainer.get_batch(batch, args.length_process, device, use_mask=args.use_mask)
        with torch.no_grad():
            if args.length_process == 'slide':
                pred = [sliding_pred(im, model, 'mean') for im in img]
                pred = torch.cat(pred, 0)
            else:
                pred = model(img)
        pred_probs = torch.softmax(pred, 1).cpu().numpy()
        prediction = {
            'soft_label': pred_probs,
            'patent_id': batch['patient_id'].cpu().numpy(),
            'spec_id': batch['spectrogram_id'].cpu().numpy()
        }
        for k, v in prediction.items():
            if k not in predictions:
                predictions[k] = v
            else:
                predictions[k] = np.concatenate((predictions[k], v), axis=0)

    valid_dataset_less.set_soft_labels(predictions['spec_id'], predictions['patent_id'], predictions['soft_label'], use_mean=args.use_mean)
    if valid_dataset_greater is not None:
        valid_dataset_less.append_fold_dataframe(valid_dataset_greater.fold_df)
    valid_dataset_less.append_fold_dataframe(train_dataset.fold_df)
    p_csv = os.path.join('pseudo.csv')
    valid_dataset_less.export_csv(p_csv)
