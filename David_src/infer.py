from data_loader import InferHMSDataset
from model import EfficientNet
from utils import load_ema_model, sliding_pred
import argparse
import torch
from copy import deepcopy
from torch.optim.swa_utils import AveragedModel
import csv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Inference script for kaggle HMS dataset.')
    # dataset
    parser.add_argument('--path_dataset', type=str, default='../input/hms-harmful-brain-activity-classification', help='Path to the dataset')
    parser.add_argument('--name_csv', type=str, default='train.csv', help='Path to Richard\'s csv file')
    parser.add_argument('--weight', type=str, default='weights/best.pt', help='Path to Richard\'s csv file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_dataset = args.path_dataset
    name_csv = args.name_csv
    path_weight = args.weight

    dataset = InferHMSDataset(
        dataset_path=path_dataset,
        csv_path=name_csv,
        cat_type='x',
        crop_size_sp=(100, 400),
        is_train=True,
    )

    model = EfficientNet(in_ch=1, width=656)
    model.to(device)

    load_ema_model(path_weight, model, device)
    model.eval()

    # Open output file
    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow('eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote'.split(','))

        dataset_iter = iter(dataset)
        dataset_iter = tqdm(dataset_iter, total=len(dataset), desc='Infering')
        for data in dataset_iter:
            eeg_tensor, img_tensor, eeg_id = data
            eeg_tensor = eeg_tensor.to(device)
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                output = sliding_pred(img_tensor, model, 'mean')
            probability = torch.nn.functional.softmax(output, dim=1)[0]
            probability = probability.cpu().tolist()
            content = [eeg_id.item()] + probability
            writer.writerow(content)


if __name__ == '__main__':
    main()
