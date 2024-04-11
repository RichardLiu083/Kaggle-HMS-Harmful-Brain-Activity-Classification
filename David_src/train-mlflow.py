import torch
import torch.nn as nn
import model
import data_loader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import os
from utils import check_path, seed_everything, check_mlflow_server, get_transform
import mlflow
from trainer import Trainer
from copy import deepcopy
from lion_pytorch import Lion
from prodigyopt import Prodigy


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for kaggle HMS dataset.')
    # dataset
    parser.add_argument('--path_dataset', type=str, default='G:\dataset\HMS', help='Path to the dataset')
    parser.add_argument('--name_fold_csv', type=str, default='train_spectrogram.csv', help='Path to Richard\'s csv file')
    parser.add_argument('--name_csv', type=str, default='train.csv', help='Path to dataset training csv file')
    parser.add_argument('--length-process', type=str, default='slide', choices=['random_crop', 'first', 'slide', 'resize'],
                        help='The way to process the spectrogram\'s length(width)')
    parser.add_argument('--length-process-train', type=str, default='random_crop',
                        choices=['random_crop', 'resize'], help='The way to process the spectrogram\'s length(width)')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for data loader')
    parser.add_argument('--cat-type', type=str, default='x', choices=['x', 'ch', 'y'],
                        help="The way to concat ['LL', 'RL', 'RP', 'LP'] images, default as x.")
    parser.add_argument('--crop-size_spec', nargs=2, type=int, default=[100, 300], help='Size of the cropped spectrogram')
    parser.add_argument('--crop-size_eeg', nargs=2, type=int, default=[50, 256], help='Size of the cropped eeg spectrogram')
    parser.add_argument('--slide_wd_ratio', type=float, default=0.5, help='The ratio of sliding window')
    parser.add_argument('--mixin_ratio', type=float, default=0.3, help='The ratio of sliding window')
    parser.add_argument('--augs', nargs='+', default=[], choices=['h-flip', 'fill-nan', 'sharp', 'gaussian', 'contrast', 'hist'],
                        help='Augmentations for the dataset')
    parser.add_argument('--aug-h-mask-ratio', type=float, default=0, help='The ratio of using h-mask in the input of network')
    parser.add_argument('--aug-h-mask-prob', type=float, default=0, help='The probability of using h-mask in the input of network')
    parser.add_argument('--aug-w-mask-ratio', type=float, default=0, help='The ratio of using w-mask in the input of network')
    parser.add_argument('--aug-w-mask-prob', type=float, default=0, help='The probability of using w-mask in the input of network')
    parser.add_argument('--fold-num', type=int, default=0, help='The number of i-th fold for validation')
    parser.add_argument('--use-two-stage', action='store_true', help='Use two-stage training')
    parser.add_argument('--use-pl', action='store_true', help='Use pseudo label')
    parser.add_argument('--use-all-pl', action='store_true', help='Use pseudo label')
    # dataset cache
    parser.add_argument('--cache_dir', type=str, default='cache', help='Path to the cache directory')
    parser.add_argument('--no_cache', action='store_true', help='Do not use cache for dataset loading')
    parser.add_argument('--flush_cache', action='store_true', help='Flush the cache for dataset loading')
    # network
    parser.add_argument('--network-type', type=str, default='cnn', choices=['cnn', 'transformer'],
                        help='The type of network')
    parser.add_argument('--use-attention', action='store_true', help='Use attention in the network')
    parser.add_argument('--use-mask', action='store_true', help='Use mask in the input of network')
    parser.add_argument('--cnn-type', type=str, default='b0', choices=['b0', 'v2s', 'mobilenet', 'convnext-tiny'],
                        help='The type of CNN')
    # training
    parser.add_argument('--loss-type', type=str, default='CE_only', choices=['CE_only', 'CE_KL', 'KL_only'],
                        help='The type of loss function')
    parser.add_argument('--optim', type=str, default='SGD', help='The type of optimizer',
                        choices=['SGD', 'Adam', 'AdamW', 'Lion', 'Prodigy'])
    parser.add_argument('--use-amsgrad', action='store_true', help='Use AMSGrad in Adam')
    parser.add_argument('--use-amp', action='store_true', help='Use Automatic Mixed Precision')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--project', type=str, default='./proj', help='Path to the project')
    parser.add_argument('--name', type=str, default='exp', help='Name of the experiment')
    parser.add_argument('--resume', action='store_true', help='Load a checkpoint')
    parser.add_argument('--seed', type=int, default=9527, help='Seed for reproducibility')
    parser.add_argument('--save-cycle', type=int, default=-1, help='Save model for every n epochs')
    parser.add_argument('--ema-type', type=str, default='none', help='Type of EMA',
                        choices=['ema', 'avg', 'none'])
    # MLFlow logging
    parser.add_argument('--mlf-des', type=str, default='', help='Description of the experiment')
    parser.add_argument('--mlf-port', type=int, default=5000, help='Description of the experiment')
    parser.add_argument('--mlf-ip', type=str, default='localhost', help='IP address of the MLflow server')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    crop_size_sp = args.crop_size_spec
    crop_size_eeg = args.crop_size_eeg
    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True)
    # Set the MLflow tracking URI
    mlflow_url = f'http://{args.mlf_ip}:{args.mlf_port}'
    mlflow.set_tracking_uri(mlflow_url)
    check_mlflow_server(mlflow_url)
    # Set up the experiment
    experiment = mlflow.set_experiment('HMS')
    if experiment is None:
        experiment = mlflow.create_experiment('HMS')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dataset
    transform_sp = get_transform(crop_size_sp, args.length_process_train, args.augs)
    transform_eeg = get_transform(crop_size_eeg, args.length_process_train, args.augs)
    if args.use_pl and args.use_two_stage:
        raise ValueError('Cannot use pseudo label and two-stage training at the same time')
    dataset_kwargs = {
        'dataset_path': args.path_dataset,
        'fold_csv': args.name_fold_csv,
        'csv_path': args.name_csv,
        'transform_sp': transform_sp,
        'transform_eeg': transform_eeg,
        'cache_dir': args.cache_dir,
        'cache': not args.no_cache,
        'flush_cache': args.flush_cache,
        'cat_type': args.cat_type,
        'slide_wd_ratio': args.slide_wd_ratio,
        'crop_size_sp': crop_size_sp,
        'crop_size_eeg': crop_size_eeg,
        'mixin_ratio': args.mixin_ratio,
        'mask_w_ratio': args.aug_w_mask_ratio,
        'mask_w_prob': args.aug_w_mask_prob,
        'mask_h_ratio': args.aug_h_mask_ratio,
        'mask_h_prob': args.aug_h_mask_prob,
        'valid_fold_num': args.fold_num,
        'use_pseudo_label': args.use_pl,
        'all_pseudo': args.use_all_pl,
    }
    train_dataset = data_loader.LabelHMSDataset(**dataset_kwargs, is_train=True)
    v_dataset_kwargs = deepcopy(dataset_kwargs)
    v_dataset_kwargs['transform_sp'] = get_transform(crop_size_sp, args.length_process, None)
    v_dataset_kwargs['transform_eeg'] = get_transform(crop_size_eeg, args.length_process, None)
    v_dataset_kwargs['mixin_ratio'] = 0
    v_dataset_kwargs['fold_csv'] = 'train_spectrogram.csv'
    valid_dataset = data_loader.LabelHMSDataset(**v_dataset_kwargs, filter_vote=True, is_train=False)
    # Data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=train_dataset.collate_fn, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=0, collate_fn=valid_dataset.collate_fn, pin_memory=True)
    # Model
    in_ch = 1
    if args.cat_type == 'ch':
        in_ch = 4
    if args.use_mask:
        in_ch += 1

    width = crop_size_sp[1] + crop_size_eeg[1] * 2
    if args.network_type == 'cnn':
        model = model.EfficientNet(width=width, use_attention=args.use_attention, cnn_type=args.cnn_type,
                                   num_classes=train_dataset.num_class, in_ch=in_ch).to(device)
    elif args.network_type == 'transformer':
        model = model.EfficientTransNet(width=width, in_ch=in_ch).to(device)
    else:
        raise NotImplementedError(f'Network type: {args.network_type} not implemented')

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    use_amsgrad = args.use_amsgrad if 'Adam' in args.optim else False
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=use_amsgrad)
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=use_amsgrad)
    elif args.optim == 'Lion':
        optimizer = Lion(model.parameters(), lr=args.lr / 5, weight_decay=args.weight_decay)
    elif args.optim == 'Prodigy':
        optimizer = Prodigy(model.parameters(), lr=1., weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer: {args.optim} not implemented')

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Initialize model
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        use_ema=args.ema_type != 'none',
        use_amp=args.use_amp,
        loss_type=args.loss_type,
        args=args,
    )
    # Resume
    proj_path = check_path(args.project, args.name)
    weight_path = os.path.join(proj_path, 'weights')
    if args.resume:
        if os.path.exists(os.path.join(weight_path, 'last.pt')):
            trainer.load(os.path.join(weight_path, 'last.pt'))
        else:
            raise FileNotFoundError(f"No checkpoint found to resume training: {os.path.join(weight_path, 'last.pt')}")
    else:
        os.makedirs(weight_path, exist_ok=True)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f'{args.name}', description=args.mlf_des):
        train_dataset.log()
        valid_dataset.log()
        dummy_input = None
        mlflow.log_params(vars(args))
        best_kl = 1e9
        # Training
        epochs = args.epochs if not args.use_two_stage else min(int(args.epochs * .2), 3)
        for epoch in range(epochs):
            trainer.train_epoch(epoch, args.epochs)
            # Validation
            trainer.valid_epoch(epoch, args.epochs)
        if args.use_two_stage:
            dataset_kwargs['filter_vote'] = True
            train_dataset = data_loader.LabelHMSDataset(**dataset_kwargs, is_train=True)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                      collate_fn=train_dataset.collate_fn, pin_memory=True)
            trainer.train_loader = train_loader
            v_dataset_kwargs['filter_vote'] = True
            valid_dataset = data_loader.LabelHMSDataset(**v_dataset_kwargs, is_train=False)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=False,
                                      num_workers=0, collate_fn=valid_dataset.collate_fn, pin_memory=True)
            trainer.valid_loader = valid_loader
            for epoch in range(epochs, args.epochs):
                trainer.train_epoch(epoch, args.epochs)
                # Validation
                trainer.valid_epoch(epoch, args.epochs)
        trainer.leave()
