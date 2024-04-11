import os.path
from ema_pytorch import EMA
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import sliding_pred, plot_confusion_matrix, save_model_mlflow
import mlflow
import argparse
from sklearn.metrics import confusion_matrix


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            scheduler,
            device,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            loss_type: str,
            use_ema: bool = False,
            use_amp: bool = False,
            args: argparse.Namespace = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        self.use_ema = use_ema
        self.ema = EMA(model, update_after_step=50, update_every=3, beta=0.99) if use_ema else self.model
        self.loss_type = loss_type
        self.args = args
        self.best_kl = 1e9

    def save_model(self, path: str, model_name: str = 'best', additional: dict = None, save_train_info: bool = False):
        save_dict = {'model': self.model.state_dict(), 'train_info': save_train_info}
        if self.ema:
            if save_train_info:
                save_dict['ema'] = self.ema.state_dict()
            else:
                save_dict['ema'] = self.ema.ema_model.state_dict()
        if additional:
            save_dict.update(additional)
        if save_train_info:
            save_dict.update({
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict() if self.scaler else None,
            })
        torch.save(save_dict, os.path.join(path, f'{model_name}.pt'))

    def leave(self):
        # load best model
        self.load(os.path.join(self.args.project, self.args.name, 'weights', 'best.pt'))
        # save best model with weights only
        self.save_model(
            os.path.join(self.args.project, self.args.name, 'weights'),
            model_name='best',
            save_train_info=False
        )
        mlflow.log_metric('best_KL', self.best_kl, step=0)

    def load(self, path: str):
        checkpoint = torch.load(path)
        load_train_info = checkpoint.get('train_info', False)
        self.model.load_state_dict(checkpoint['model'])
        if 'ema' in checkpoint and self.use_ema:
            try:
                self.ema.load_state_dict(checkpoint['ema'])
            except:
                self.ema.ema_model.load_state_dict(checkpoint['ema'])
        if load_train_info:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            if self.scaler:
                self.scaler.load_state_dict(checkpoint['scaler'])

    @staticmethod
    def train_env(func):
        def wrapper(self, *args, **kwargs):
            self.model.train()
            self.ema.train()
            return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def eval_env(func):
        def wrapper(self, *args, **kwargs):
            self.model.eval()
            self.ema.eval()
            return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def get_batch(batch, get_type: str, device, use_mask: bool = False):
        img, eeg, label, soft_label = [batch[k] for k in 'spec,eeg,label,soft_label'.split(',')]
        if get_type in 'random_crop,first'.split(','):
            img = [i[:1] for i in img] if get_type == 'first' else img
            img = torch.cat(img, 0)
            img = img.to(device)
        elif get_type == 'slide':
            img = [i.to(device) for i in img]
        else:
            raise NotImplementedError(f'get_type: {get_type} not implemented')
        # Concatenate EEG and spec
        if isinstance(img, list):
            imgs = []
            for idx, im in enumerate(img):
                b_size = im.size(0)
                im_w = im.size(3)
                eeg_t = eeg[idx:idx+1] * torch.ones(b_size, 1, 1, 1)
                im_cat = torch.cat([im, eeg_t.to(device)], 3)
                if use_mask:
                    mask = torch.ones_like(im_cat)
                    mask[:, :, :, :im_w] = 0
                    im_cat = torch.cat([im_cat, mask], 1)
                imgs.append(im_cat)
            img = imgs
        else:
            im_w = img.size(3)
            img = torch.cat([img, eeg.to(device)], 3)
            if use_mask:
                mask = torch.ones_like(img)
                mask[:, :, :, :im_w] = 0
                img = torch.cat([img, mask], 1)
        label, soft_label = label.to(device), soft_label.to(device)
        return img, label, soft_label

    def criterion(self, output, label, soft_label):
        ce_loss = torch.nn.functional.cross_entropy(output, soft_label)
        kl_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(output, dim=1), soft_label)
        if self.loss_type == 'CE_only':
            loss = ce_loss * 3
        elif self.loss_type == 'KL_only':
            loss = kl_loss
        elif self.loss_type == 'CE_KL':
            loss = ce_loss * 2 + kl_loss * 2
        else:
            raise NotImplementedError(f'loss_type: {self.loss_type} not implemented')
        if torch.isnan(loss).any():
            raise ValueError('Loss is NaN')
        return loss, {'ce_loss': ce_loss.item(), 'kl_loss': kl_loss.item()}

    @train_env
    def train_epoch(self, epoch: int, total_epochs: int):
        running_ce_loss, running_kl_loss = 0, 0
        train_bar = tqdm(self.train_loader, total=len(self.train_loader), desc=f'Epoch {epoch + 1}/{total_epochs}')
        for i, batch in enumerate(train_bar):
            self.optimizer.zero_grad()
            img, label, soft_label = self.get_batch(batch, self.args.length_process_train, self.device, use_mask=self.args.use_mask)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(img)
                loss, loss_dict = self.criterion(output, label, soft_label)
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            # Update EMA
            if self.use_ema:
                self.ema.update()
            # Update progress bar
            running_ce_loss += loss_dict['ce_loss']
            running_kl_loss += loss_dict['kl_loss']
            train_bar.set_postfix({
                'ce_loss': running_ce_loss / (i + 1),
                'kl_loss': running_kl_loss / (i + 1),
            })
        # Update scheduler
        self.scheduler.step()
        # Log to mlflow
        mlflow.log_metric('train/CrossEntropyLoss', running_ce_loss / len(train_bar), step=epoch)
        mlflow.log_metric('train/KLLoss', running_kl_loss / len(train_bar), step=epoch)

    @eval_env
    def valid_epoch(self, epoch, epochs):
        running_ce_loss, running_kl_loss = 0, 0
        all_preds, all_labels, all_corrects = [], [], []
        valid_bar = tqdm(self.valid_loader, total=len(self.valid_loader), desc=f'Epoch {epoch + 1}/{epochs}')
        img = None
        for i, batch in enumerate(valid_bar):
            img, label, soft_label = self.get_batch(batch, self.args.length_process, self.device, use_mask=self.args.use_mask)
            with torch.no_grad():
                if self.args.length_process == 'slide':
                    pred = [sliding_pred(im, self.ema, 'mean') for im in img]
                    pred = torch.cat(pred, 0)
                else:
                    pred = self.ema(img)
            # Confusion matrix
            _, preds = torch.max(pred, 1)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            # Calculate accuracy
            pred_labels = torch.argmax(F.softmax(pred, dim=1), dim=1)
            correct_predictions = torch.sum(pred_labels == label)
            accuracy = correct_predictions.item()
            all_corrects.append(accuracy)
            # Calculate KL-Divergence
            kl_div_loss = F.kl_div(F.log_softmax(pred, dim=1), soft_label, reduction='batchmean')
            # Cross-Entropy Loss
            loss, loss_dict = self.criterion(pred, label, soft_label)
            running_ce_loss += loss_dict['ce_loss']
            running_kl_loss += kl_div_loss.item()
            valid_bar.set_postfix({
                'ce_loss': running_ce_loss / (i + 1),
                'kl_loss': running_kl_loss / (i + 1),
                'accuracy': accuracy,
            })

        # Log with MLflow
        mlflow.log_metric('valid/CrossEntropyLoss', running_ce_loss / len(self.valid_loader), epoch)
        mlflow.log_metric('valid/accurracy', sum(all_corrects) / len(self.valid_loader), epoch)
        mlflow.log_metric('valid/KLLoss', running_kl_loss / len(self.valid_loader), epoch)
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        class_names = [f'Class {i}' for i in range(self.valid_loader.dataset.num_class)]  # Adjust as per your classes
        fig = plot_confusion_matrix(cm, class_names)
        # Log confusion matrix with MLflow
        mlflow.log_figure(fig, f'confusion_matrix/epoch_{epoch + 1:02d}.png')

        # Save checkpoint
        if self.args.save_cycle > 0 and (epoch + 1) % self.args.save_cycle == 0:
            self.save_model(
                os.path.join(self.args.project, self.args.name, 'weights'),
                model_name=f'epoch_{epoch + 1:02d}',
                save_train_info=False
            )

        # Save best model
        if running_kl_loss < self.best_kl:
            self.best_kl = running_kl_loss
            self.save_model(
                os.path.join(self.args.project, self.args.name, 'weights'),
                model_name='best',
                save_train_info=True
            )
            ema = self.ema.ema_model if self.use_ema else None
            if isinstance(img, list):
                img = img[0]
            save_model_mlflow(self.model, ema, img, name='best')

