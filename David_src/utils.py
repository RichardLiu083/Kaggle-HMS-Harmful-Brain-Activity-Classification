import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import random
import os
import mlflow
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn as nn


def get_transform(crop_size, length_process_type, augs: list[str] | None) -> transforms.Compose:
    augs = augs or []
    transform = [transforms.ToTensor()]
    either_choice = []
    either_choice_p = 0.5
    for t in augs:
        if t == 'sharp':
            either_choice.append(v2.RandomAdjustSharpness(1))
        elif t == 'gaussian':
            either_choice.append(v2.GaussianBlur((5, 9), sigma=(0.1, 2.0)))
        elif t == 'contrast':
            either_choice.append(v2.RandomAutocontrast())
        elif t == 'hist':
            either_choice.append(v2.RandomEqualize())
        elif t == 'fill-nan':
            raise NotImplementedError('fill-nan not implemented')
    if len(either_choice) > 0:
        transform.append(transforms.RandomApply([transforms.RandomChoice(either_choice)], p=either_choice_p))
    if length_process_type == 'random_crop':
        transform.append(transforms.RandomCrop(crop_size, pad_if_needed=True))
    elif length_process_type == 'first':
        transform.append(transforms.Lambda(lambda x: x[:1]))
    elif length_process_type == 'slide':
        pass
    else:
        raise NotImplementedError(f'length_process_type: {length_process_type} not implemented')
    return transforms.Compose(transform)


def exponential_decay_array(n, k=0.05):
    x = np.arange(n)
    return np.exp(-k * x)


def sliding_pred(imgs: torch.Tensor, model: torch.nn.Module, aggregate_method):
    imgs = imgs.split(1, 0)
    preds = [model(img) for img in imgs]
    if aggregate_method == 'max':
        preds = torch.stack(preds).max(0).values
    elif aggregate_method == 'mean':
        preds = torch.stack(preds).mean(0)
    else:
        raise NotImplementedError(f'aggregate_method: {aggregate_method} not implemented')
    return preds


def process_train_img_batch(img, device, length_process_type):
    if length_process_type == 'random_crop':
        img = torch.cat(img, 0)
        img = img.to(device)
    elif length_process_type == 'first':
        img = [i[:1] for i in img]
        img = torch.cat(img, 0)
        img = img.to(device)
    elif length_process_type == 'slide':
        img = [i.to(device) for i in img]
    else:
        raise NotImplementedError(f'length_process_type: {length_process_type} not implemented')
    return img


def save_model_mlflow(model, ema_model, dummy_input: torch.Tensor, name='best'):
    with torch.no_grad():
        signature = mlflow.models.infer_signature(dummy_input.cpu().numpy(), model(dummy_input).cpu().numpy())
    mlflow.pytorch.log_model(model, f'model/{name}', signature=signature)
    if ema_model is not None:
        mlflow.pytorch.log_model(ema_model, f'model/{name}_ema', signature=signature)


def check_mlflow_server(url="http://localhost:5000"):
    import requests

    response = requests.get(url)
    if response.status_code == 200:
        print("MLflow server is running.")
    else:
        raise ConnectionError(f"MLflow server is not running at {url}")


def check_path(proj: str, exp: str, ok=False) -> str:
    path = os.path.join(proj, exp)
    if os.path.exists(path) and ok:
        return path
    if not os.path.exists(path):
        return path
    # path already exists, append a number to the end
    count = 1
    while os.path.exists(path + str(count)):
        count += 1
    return path + str(count)


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def load_ema_model(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['ema_state_dict'])


def load_mlflow_model(pth_path, model, device):
    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint.state_dict())


def load_checkpoint(checkpoint_path, model, ema_model, optimizer, scheduler, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['epoch']


def save_checkpoint(state, proj, filename="last.pt"):
    torch.save(state, os.path.join(proj, filename))


def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    a = exponential_decay_array(100)
    print(a)

