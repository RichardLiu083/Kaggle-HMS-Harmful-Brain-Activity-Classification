import subprocess
import random
import os


os.chdir('..')

def baseline(kwargs):
    baseline = {
        'augs': 'h-flip contrast',
        'ema-type': 'ema',
        'loss-type': 'CE_KL',
        'name': 'baseline',
        'optim': 'Lion',
        # 'use-attention': '',
        'aug-h-mask-prob': 0.1,
        'aug-h-mask-ratio': 0.1,
        'seed': 10260,
        'network-type': 'transformer',
        # 'network-type': 'cnn',
        'length-process': 'first',
        'num-workers': 4,
        # 'use-two-stage': '',
        'use-pl': '',
        'name_fold_csv': 'pseudo.csv',
    }
    baseline.update(kwargs)
    baseline['epoch'] = 15 if baseline['network-type'] == 'cnn' else 20
    return baseline


# List of dictionaries containing kwargs to update
kwargs_list = [
    # baseline({'name': f'cnn-PL-f{n}', 'fold': n, 'network-type': 'cnn', 'use-amp': ''})
    # for n in range(5)
    # baseline({'name': f'cnn-b0-one', 'fold': -1, 'network-type': 'cnn', 'save-cycle': 2, 'use-amp': ''}),
    # baseline({'name': f'cnn-mb-one', 'fold': -1, 'network-type': 'cnn', 'save-cycle': 2, 'use-amp': ''}),
    baseline({'name': f'cnn-b0-PL2-f{n}', 'fold': n, 'network-type': 'cnn', 'use-amp': '', 'cnn-type': 'b0'}) for n in range(4)
    # baseline({'name': f'cnn-mb3n', 'fold': 0, 'network-type': 'cnn', 'use-amp': '', 'cnn-type': 'mobilenet'}),
    # baseline({'name': f'cnn-v2s', 'fold': 0, 'network-type': 'cnn', 'use-amp': '', 'cnn-type': 'v2s'}),
    # baseline({'name': f'cnn-convnext', 'fold': 0, 'network-type': 'cnn', 'use-amp': '', 'cnn-type': 'convnext-tiny'}),
]
kwargs_list += [
    baseline({'name': f'cnn-mb-PL2-f{n}', 'fold': n, 'network-type': 'cnn', 'use-amp': '', 'cnn-type': 'mobilenet'}) for n in range(4)
]
kwargs_list = list(filter(lambda x: x['fold'] != 1, kwargs_list))
# kwargs_list += [
#     baseline({'name': f'cnn-one', 'fold': -1, 'network-type': 'cnn', 'save-cycle': 3, 'use-amp': ''}),
#     baseline({'name': f'trans-one', 'fold': -1, 'network-type': 'transformer', 'save-cycle': 3}),
# ]
# kwargs_list += [
#     baseline({'name': f'trans-f{n}', 'fold': n, 'use-two-stage': '', 'name_fold_csv': 'train_spectrogram.csv',
#               'network-type': 'transformer'})
#     for n in range(5)
# ]

# Define the base command
base_command = "python train-mlflow.py"

# Iterate through the kwargs list
for kwargs in kwargs_list:
    # Construct the command with updated kwargs
    command = base_command
    for key, value in kwargs.items():
        command += f" --{key} {value}"

    # Run the command
    print(f"Running command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        print("Command executed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}\n")
