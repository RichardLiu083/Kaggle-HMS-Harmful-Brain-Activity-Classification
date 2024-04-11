import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pywt
from librosa.feature import melspectrogram
from librosa import power_to_db


class InferHMSDataset(Dataset):
    def __init__(
            self,
            dataset_path=None,
            csv_path=None,
            s_mean=None,
            s_std=None,
            cat_type='x',
            df=None,
            crop_size_sp=None,
            crop_size_eeg=None,
            slide_wd_ratio=0.5,  # -1 for no sliding window
            is_train=False,
            transform_sp=None,
            transform_eeg=None,
            duration=None,
    ):
        self.num_class = 6
        # Read the main csv file
        if df is None and dataset_path and csv_path:
            self.df = pd.read_csv(os.path.join(dataset_path, csv_path))
        elif df and (dataset_path is None and csv_path is None):
            self.df = df
        else:
            raise ValueError('Either df or dataset_path and csv_path must be provided')

        if s_std is None:
            s_std = [2.3019, 2.2844, 2.2947, 2.3129]
        if s_mean is None:
            s_mean = [-.1119, -.129, -.1395, -.1689]
        self.dataset_path = dataset_path
        self.mean = s_mean
        self.std = s_std
        self.cat_type: str = cat_type
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform_sp = transform_sp if transform_sp else self.transform
        self.transform_eeg = transform_eeg if transform_eeg else self.transform
        if crop_size_sp is None:
            crop_size_sp = [100, 256]
        self.crop_size_sp = crop_size_sp
        if crop_size_eeg is None:
            crop_size_eeg = [100, 256]
        self.crops_size_eeg = crop_size_eeg
        self.slide_wd_ratio = slide_wd_ratio
        self.base_dir = 'train' if is_train else 'test'
        self.is_train = is_train
        self.sensor_names = ['LL', 'RL', 'RP', 'LP']
        self.eeg_feat = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
                         ['Fp1', 'F3', 'C3', 'P3', 'O1'],
                         ['Fp2', 'F8', 'T4', 'T6', 'O2'],
                         ['Fp2', 'F4', 'C4', 'P4', 'O2']]
        self.eeg_sample_duration = duration if duration is not None else [10, 15, 30, 45]
        self.spec_sample_duration = [600]

    def __len__(self):
        return len(self.df)

    def log(self, kwargs: dict):
        import mlflow
        prefix = 'valid' if not self.is_train else 'train'
        transform_sp_list = [t.__class__.__name__ for t in self.transform_sp.transforms]
        transform_eeg_list = [t.__class__.__name__ for t in self.transform_eeg.transforms]
        kwargs = {f'{prefix}-{k}': v for k, v in kwargs.items()}
        mlflow.log_params({
            f'{prefix}-mean': self.mean,
            f'{prefix}-std': self.std,
            f'{prefix}-transform_sp': transform_sp_list,
            f'{prefix}-transform_eeg': transform_eeg_list,
            **kwargs
        })

    def norm_img(self, img: np.ndarray, idx: int) -> np.ndarray:
        """ Input image should be channel-last image """
        # std_img = (img - self.mean[idx]) / self.std[idx]
        eps = 1e-6
        std_img = (img - np.nanmean(img)) / (np.nanstd(img) + eps)
        return std_img

    def read_eeg(self, eeg_id, offset, h=50, w=256) -> list[list[np.ndarray]]:
        parquet_path = f'{self.base_dir}_eegs/{eeg_id}.parquet'
        parquet_path = os.path.join(self.dataset_path, parquet_path)
        # LOAD MIDDLE 50 SECONDS OF EEG SERIES
        eeg_ori = pd.read_parquet(parquet_path)
        # Load multiple eeg from durations
        d_imgs = []
        for duration in self.eeg_sample_duration:
            middle = int(offset + 25)
            eeg = eeg_ori.iloc[(middle - duration // 2) * 200:(middle + duration // 2) * 200]
            # VARIABLE TO HOLD SPECTROGRAM
            img = np.zeros((h, w, 4), dtype='float32')
            signals = []
            for k in range(4):
                COLS = self.eeg_feat[k]
                for kk in range(4):
                    # COMPUTE PAIR DIFFERENCES
                    x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values
                    # FILL NANS
                    m = np.nanmean(x)
                    m = 0 if np.isnan(m) else m
                    if np.isnan(x).mean() < 1:
                        x = np.nan_to_num(x, nan=m)
                    else:
                        x[:] = 0
                    # DENOISE
                    signals.append(x)
                    # RAW SPECTROGRAM
                    mel_spec = melspectrogram(y=x, sr=200, hop_length=len(x) // w, n_fft=1024, n_mels=h, fmin=0,
                                              fmax=20, win_length=128)
                    # LOG TRANSFORM
                    width = (mel_spec.shape[1] // 32) * 32
                    mel_spec_db = power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :width]
                    # STANDARDIZE TO -1 TO 1
                    mel_spec_db = (mel_spec_db + 40) / 40
                    img[:, :, k] += mel_spec_db
                # AVERAGE THE 4 MONTAGE DIFFERENCES
                img[:, :, k] /= 4.0
            d_imgs.append([img[:, :, i] for i in range(img.shape[-1])])
        return d_imgs

    def read_spectrogram(self, spectrogram_id, offset) -> list[np.ndarray]:
        """ Read the given spectrogram and apply normalization """
        spectrogram_path = f'{self.base_dir}_spectrograms/{spectrogram_id}.parquet'
        raw = pd.read_parquet(os.path.join(self.dataset_path, spectrogram_path)).fillna(0)
        sensor_types = self.sensor_names
        raw = raw.loc[(raw.time >= offset) & (raw.time < offset + 600)]
        sensor_data = [list(raw.filter(like=s, axis=1)) for s in sensor_types]
        sensor_data = [np.log1p(raw[s].T.values) for s in sensor_data]
        sensor_data = self.norm_img(np.stack(sensor_data, 0), -1)
        sensor_data = np.split(sensor_data, sensor_data.shape[0], 0)
        sensor_data = [np.nan_to_num(s, nan=0)[0] for s in sensor_data]
        return sensor_data

    def cat_imgs(self, img_list: list[torch.Tensor], data_type: str):
        cat_dim = 'ch,x,y'.split(',').index(self.cat_type)
        if data_type == 'eeg':
            n = len(img_list) // len(self.eeg_sample_duration)
            img_list = [img_list[i:i + n] for i in range(0, len(img_list), n)]
        else:
            img_list = [img_list]

        result = []
        for l in img_list:
            l = torch.cat(l, cat_dim)
            result.append(l)
        if len(result) == 4:
            result = [torch.cat(result[:2], -1), torch.cat(result[2:], -1)]
        return torch.cat(result, 1)

    def length_proces(self, t: torch.Tensor, data_type: str):
        """ Process the length of the given tensor on the last dimension (image width) """
        assert data_type in ['spec', 'eeg'], f'data_type {data_type} not implemented'
        base_length = self.crop_size_sp[1] if data_type == 'spec' else self.crops_size_eeg[1]
        if data_type == 'eeg':
            base_length *= 1 if len(self.eeg_sample_duration) <= 2 else len(self.eeg_sample_duration) // 2
        min_sliding_len = base_length * self.slide_wd_ratio
        # padding short image
        if t.shape[-1] < base_length:
            pad = base_length - t.shape[-1]
            t = torch.nn.functional.pad(t, (0, pad), 'constant', 0)
            return t.unsqueeze(0)
        # sliding window
        if self.slide_wd_ratio > 0 and t.shape[-1] > base_length:
            t_list = []
            while t.shape[-1] > min_sliding_len:
                t_list.append(t[:, :, :base_length])
                t = t[:, :, int(base_length * self.slide_wd_ratio):]
            if t_list[-1].shape[-1] < base_length:
                pad = base_length - t_list[-1].shape[-1]
                t_list[-1] = torch.nn.functional.pad(t_list[-1], (0, pad), 'constant', 0)
            t = torch.stack(t_list, 0)
            return t
        return t.unsqueeze(0)

    def process_spectrogram(self, spectrogram: list[torch.Tensor], data_type: str):
        spectrogram = self.cat_imgs(spectrogram, data_type)
        spectrogram = self.length_proces(spectrogram, data_type)
        return spectrogram

    def get_spectrogram_by_id(self, spectrogram_id, offset: float) -> list[np.ndarray]:
        spectrogram_images = self.read_spectrogram(spectrogram_id, offset)
        return spectrogram_images

    def get_eeg_by_id(self, eeg_id, offset) -> list[np.ndarray]:
        eeg_images = self.read_eeg(eeg_id, offset)
        eeg_image = [eeg for eeg_list in eeg_images for eeg in eeg_list]
        return eeg_image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_id = row['eeg_id']
        spectrogram_id = row['spectrogram_id']
        offset_eeg = row['eeg_label_offset_seconds']
        offset_spec = row['spectrogram_label_offset_seconds']
        eeg, spec = self.get_eeg_by_id(eeg_id, offset_eeg), self.get_spectrogram_by_id(spectrogram_id, offset_spec)
        eeg, spec = [[trans(s) for s in ss] for trans, ss in zip([self.transform_eeg, self.transform_sp], [eeg, spec])]
        return self.process_spectrogram(eeg, 'eeg'), self.process_spectrogram(spec, 'spec'), eeg_id

    @staticmethod
    def collate_fn(batch):
        transposed_batch = list(zip(*batch))
        eeg, spec, eeg_id = transposed_batch
        data = []
        for sidx, s in enumerate(spec):
            bidx = s.size()[0]
            e = eeg[sidx] * torch.ones((bidx, 1, 1, 1))
            data.append(torch.cat([s, e], -1))
        return data, eeg_id


class LabelHMSDataset(InferHMSDataset):
    def __init__(
            self,
            fold_csv,
            cache=True,
            cache_dir='cache',
            flush_cache=False,
            flush_target='both',
            filter_vote: bool = False,
            filter_vote_mode: str = 'drop_less',
            mixin_ratio: float = 0,
            mask_w_prob: float = 0,
            mask_w_ratio: float = 0,
            mask_h_prob: float = 0,
            mask_h_ratio: float = 0,
            valid_fold_num: int = 0,
            use_pseudo_label: bool = False,
            all_pseudo: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.fold_df = pd.read_csv(fold_csv)
        self.data_hq, self.data_lq = create_train_data(os.path.join(self.dataset_path, 'train.csv'))
        self.valid_fold_num = valid_fold_num
        self.use_pseudo_label = use_pseudo_label
        self.all_pseudo = all_pseudo
        if self.is_train:
            self.fold_df = self.fold_df[self.fold_df['fold'] != valid_fold_num]
        else:
            bk = self.fold_df.copy()
            self.fold_df = self.fold_df[self.fold_df['fold'] == valid_fold_num]
            if self.fold_df[self.fold_df['fold'] == valid_fold_num].shape[0] == 0:
                self.fold_df = bk[bk['fold'] == 0]
        if filter_vote:
            for idx, row in self.fold_df.iterrows():
                spec_id = row['spectrogram_id']
                p_data = self.df[self.df['spectrogram_id'] == spec_id]
                votes = 'seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote'.split(',')
                votes = np.asarray([p_data[v] for v in votes]).sum(0)
                p_idx = np.argmax(votes)
                total_votes = votes[p_idx]
                if filter_vote_mode == 'drop_less' and total_votes < 10:
                    self.fold_df.drop(idx, inplace=True)
                elif filter_vote_mode == 'drop_greater' and total_votes >= 10:
                    self.fold_df.drop(idx, inplace=True)
        self.cache = cache
        self.flush_cache = flush_cache
        self.cache_dir = cache_dir
        self.flush_target = flush_target
        self.mixin_ratio = mixin_ratio
        self.mask_w_ratio = mask_w_ratio
        self.mask_w_prob = mask_w_prob
        self.mask_h_ratio = mask_h_ratio
        self.mask_h_prob = mask_h_prob
        if self.cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, 'spec'), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, 'eeg'), exist_ok=True)

        self.flip = transforms.RandomHorizontalFlip(p=1)
        # overwrite the base_dir
        self.base_dir = 'train'

    def __len__(self):
        return len(self.fold_df)

    def log(self, kwargs: dict = None):
        kwargs = kwargs or {}
        kwargs['mixin_ratio'] = self.mixin_ratio
        super().log(kwargs)

    def get_data_by_id(self, data_id, data_type, offset=0, flip_data=False) -> list[torch.Tensor]:
        assert data_type in ['spec', 'eeg'], f'data_type {data_type} not implemented'
        cache_path = os.path.join(self.cache_dir, data_type, f'{data_id}.npy')
        spectrogram = None
        if self.cache:
            if os.path.exists(cache_path):
                if not self.flush_cache or (self.flush_target != data_type and self.flush_target != 'both'):
                    spectrogram = np.load(cache_path)
        # No cache or cache miss
        if spectrogram is None and data_type == 'spec':
            spectrogram = super().get_spectrogram_by_id(data_id, offset)
        elif spectrogram is None and data_type == 'eeg':
            spectrogram = super().get_eeg_by_id(data_id, offset)
        # Cache the data
        if (self.cache and not os.path.exists(cache_path)) or (self.cache and self.flush_cache and self.flush_target == data_type) or (self.cache and self.flush_cache and self.flush_target == 'both'):
            np.save(cache_path, [s for s in spectrogram])
        # Apply transform
        if data_type == 'eeg':
            spectrogram = [self.transform_eeg(s) for s in spectrogram]
        elif data_type == 'spec':
            spectrogram = [self.transform_sp(s) for s in spectrogram]
        n = spectrogram[0].shape[1]
        if flip_data:
            spectrogram = list(self.flip(torch.cat(spectrogram, 1)).split(n, 1))
        # Process data length
        spectrogram = self.process_spectrogram(spectrogram, data_type)
        # Randomly drop width
        if self.mask_w_prob > 0 and np.random.rand() < self.mask_w_prob:
            ratio = np.random.rand() * self.mask_w_ratio
            mask_w = int(spectrogram.shape[-1] * ratio)
            st_w = np.random.randint(0, spectrogram.shape[-1] - mask_w)
            spectrogram[:, :, st_w:st_w + mask_w] = 0
        # Randomly drop height
        if self.mask_h_prob > 0 and np.random.rand() < self.mask_h_prob:
            ratio = np.random.rand() * self.mask_h_ratio
            mask_h = int(spectrogram.shape[-2] * ratio)
            st_h = np.random.randint(0, spectrogram.shape[-2] - mask_h)
            spectrogram[:, st_h:st_h + mask_h, :] = 0
        return spectrogram

    @staticmethod
    def pick_max_vote(dataframe):
        votes = 'seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote'.split(',')
        votes = np.asarray([dataframe[k] for k in votes]).T
        idx = np.argmax(votes.sum(1))
        # re-index the dataframe
        dataframe = dataframe.iloc[idx:]
        votes = votes[idx:]
        eeg_id = dataframe['eeg_id'].iloc[0]
        dataframe = dataframe[dataframe['eeg_id'] == eeg_id]
        # check if there are more than 2 eeg segments, if so, pick middle one
        idx = 0
        if dataframe.shape[0] > 2:
            vt = votes.copy()
            if np.array_equal(vt[0], vt[-1]):
                # remaining votes are the same
                idx = idx + dataframe.shape[0] // 2
            else:
                ed_idx = -1
                for i in range(1, vt.shape[0]):
                    if not np.array_equal(vt[i], vt[0]):
                        ed_idx = i
                        break
                idx = idx + ed_idx // 2
        return votes[idx], idx

    def _get_item(self, idx, flip_data):
        row = self.fold_df.iloc[idx]
        spectrogram_id = row['spectrogram_id']
        # Get eeg id from original csv
        p_data = self.df[self.df['patient_id'] == row['patient_id']]
        p_data = p_data[p_data['spectrogram_id'] == spectrogram_id]
        # p_data = p_data[p_data['eeg_sub_id'] == 0]
        votes, idx = self.pick_max_vote(p_data)
        eeg_id = p_data['eeg_id'].iloc[idx]
        vote_sum = votes.sum()
        offset_spec = p_data['spectrogram_label_offset_seconds'].iloc[idx]
        offset_eeg = p_data['eeg_label_offset_seconds'].iloc[idx]
        # Get the data
        eeg, spec = self.get_data_by_id(eeg_id, 'eeg', offset_eeg, flip_data), self.get_data_by_id(spectrogram_id, 'spec', offset_spec, flip_data)
        # Process other information
        row = {k: row[k] for k in row.keys()}
        if (self.use_pseudo_label and vote_sum < 10) or (self.all_pseudo and self.use_pseudo_label):
            row['soft_label'] = np.asarray(json.loads(row['soft_label']))
        else:
            row['soft_label'] = votes / votes.sum()
        return spec, eeg, row, vote_sum

    def __getitem__(self, idx):
        flip_data = np.random.rand() < .5
        if np.random.rand() < self.mixin_ratio:
            idx_2 = np.random.randint(len(self))
            spec, eeg, row, votes = self._get_item(idx, flip_data)
            spec_2, eeg_2, row_2, votes_2 = self._get_item(idx_2, flip_data)
            spec, soft_label, alpha = self.mixin(spec, spec_2, row['soft_label'], row_2['soft_label'])
            eeg, *_ = self.mixin(eeg, eeg_2, row['soft_label'], row_2['soft_label'], alpha)
            row['soft_label'] = soft_label
            return spec, eeg, row, alpha * votes + (1 - alpha) * votes_2
        else:
            return self._get_item(idx, flip_data)

    def append_fold_dataframe(self, df: pd.DataFrame):
        self.fold_df = pd.merge(self.fold_df, df, how='outer')

    def set_soft_labels(self, spec_ids: list[int], patient_ids: list[int], pseudo_labels: list[np.ndarray], use_mean=False):
        for spec_id, patient_id, pseudo_label in zip(spec_ids, patient_ids, pseudo_labels):
            self.set_soft_label(spec_id, patient_id, pseudo_label, use_mean)

    def set_soft_label(self, spec_id: int, patient_id: int, pseudo_label: np.ndarray, use_mean):
        assert isinstance(pseudo_label, np.ndarray), 'pseudo_label must be a numpy array'
        # query the spec_id and patient_id with &
        idx = (self.fold_df['spectrogram_id'] == spec_id) & (self.fold_df['patient_id'] == patient_id)
        # set the soft label
        if use_mean:
            ori_label = json.loads(''.join(self.fold_df.loc[idx, 'soft_label']))
            pseudo_label = (np.array(ori_label) + pseudo_label) / 2
        self.fold_df.loc[idx, 'soft_label'] = json.dumps(pseudo_label.tolist())

    def export_csv(self, path):
        self.fold_df.to_csv(path, index=False)

    @staticmethod
    def mixin(img_1, img_2, label_1, label_2, alpha=None):
        if alpha is None:
            alpha = np.random.beta(.5, .5)
        return img_1 * alpha + img_2 * (1 - alpha), label_1 * alpha + label_2 * (1 - alpha), alpha

    @staticmethod
    def collate_fn(batch) -> dict:
        transposed_batch = list(zip(*batch))
        eeg_tensor = torch.concat(transposed_batch[1], 0)
        keys_of_interest = ['spectrogram_id', 'image_path', 'expert_consensus', 'patient_id', 'label', 'soft_label',
                            'fold']
        result = {k: [b[k] for b in transposed_batch[2]] for k in keys_of_interest}
        for k in 'label,patient_id,fold,soft_label,spectrogram_id'.split(','):
            result[k] = torch.tensor(np.array(result[k]))
        result['eeg'] = eeg_tensor
        result['spec'] = transposed_batch[0]
        result['votes'] = torch.tensor(transposed_batch[-1])
        return result


def create_train_data(dataset_path):
    classes = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
    # Read the dataset
    df = pd.read_csv(dataset_path)

    # Create a new identifier combining multiple columns
    id_cols = ['eeg_id', 'spectrogram_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote',
               'other_vote']
    df['new_id'] = df[id_cols].astype(str).agg('_'.join, axis=1)

    # Calculate the sum of votes for each class
    df['sum_votes'] = df[classes].sum(axis=1)

    # Group the data by the new identifier and aggregate various features
    agg_functions = {
        'eeg_id': 'first',
        'eeg_label_offset_seconds': ['min', 'max'],
        'spectrogram_label_offset_seconds': ['min', 'max'],
        'spectrogram_id': 'first',
        'patient_id': 'first',
        'expert_consensus': 'first',
        **{col: 'sum' for col in classes},
        'sum_votes': 'mean',
    }
    grouped_df = df.groupby('new_id').agg(agg_functions).reset_index()

    # Flatten the MultiIndex columns and adjust column names
    grouped_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped_df.columns]
    grouped_df.columns = grouped_df.columns.str.replace('_first', '').str.replace('_sum', '').str.replace('_mean', '')

    # Normalize the class columns
    y_data = grouped_df[classes].values
    y_data_normalized = y_data / y_data.sum(axis=1, keepdims=True)
    grouped_df[classes] = y_data_normalized

    # Split the dataset into high and low quality based on the sum of votes
    high_quality_df = grouped_df[grouped_df['sum_votes'] >= 10].reset_index(drop=True)
    low_quality_df = grouped_df[(grouped_df['sum_votes'] < 10) & (grouped_df['sum_votes'] >= 0)].reset_index(drop=True)

    return high_quality_df, low_quality_df


if __name__ == '__main__':
    from model import EfficientNet
    from utils import get_transform, sliding_pred, load_mlflow_model, load_ema_model
    from trainer import Trainer
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    # Set up the experiment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = 'G:\dataset\HMS'
    # dataset_path = '/home/davidjaw/Desktop/datasets/HMS'
    csv_filename = 'train_spectrogram.csv'

    use_mask = True
    in_ch = 1 if not use_mask else 2
    net = EfficientNet(in_ch=in_ch, use_attention=True, width=656, debug_mode=True).to(device)
    # load_mlflow_model('weights/best.pth', net, device)
    # load_ema_model('proj/before-adaP-att1/weights/best.pt', net, device)

    # is_train = False
    # csv_filename = 'train.csv' if is_train else 'test.csv'
    # infer_dataset = InferHMSDataset(dataset_path, csv_filename, is_train=is_train)
    # loader = DataLoader(infer_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=infer_dataset.collate_fn)
    # for data in loader:
    #     with torch.no_grad():
    #         pred = [sliding_pred(x.to(device), net, 'mean') for x in data[0]]
    #     pass

    trans_sp = get_transform([100, 300], 'random_crop', ['h-flip'])
    trans_eeg = get_transform([50, 256], 'random_crop', ['h-flip'])
    dataset = LabelHMSDataset(csv_filename, dataset_path=dataset_path, csv_path='train.csv', is_train=False, crop_size_eeg=[50, 256], crop_size_sp=[100, 300],
                              transform_sp=trans_sp, transform_eeg=trans_eeg, mixin_ratio=.3, flush_cache=True,
                              mask_w_prob=.3, mask_w_ratio=.2)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=dataset.collate_fn)
    data_loader = tqdm(data_loader, desc='Iterating over dataset')
    for data in data_loader:
        img, label, soft_label = Trainer.get_batch(data, 'random_crop', device)
        # img_np = img.cpu().numpy()
        # for i in range(img_np.shape[0]):
        #     npi = img_np[i].transpose(1, 2, 0)
        #     npi_max = npi[:, :256].max()
        #     npi[:, 256:] = npi[:, 256:] / npi[:, 256:].max() * npi_max
        #     plt.imshow(npi)
        #     plt.set_cmap('plasma')
        #     plt.show()
        #     pass
        #     plt.close('all')
        #     plt.clf()

