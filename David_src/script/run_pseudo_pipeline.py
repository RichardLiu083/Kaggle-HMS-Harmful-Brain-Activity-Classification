import subprocess
import random
import os


os.chdir('..')

def baseline(kwargs):
    baseline = {
        # 'use-attention': '',
        'network-type': 'cnn',
        'gen-all': '',
    }
    baseline.update(kwargs)
    return baseline


# List of dictionaries containing kwargs to update
get_wp = lambda n: os.path.join(n, 'weights/best.pt')
kwargs_list = [
    baseline({'fold-num': 0, 'path-weights': get_wp('proj/cnn-b0-PL-f0'), 'cnn-type': 'b0'}),
    baseline({'fold-num': 0, 'path-weights': get_wp('proj/cnn-mb-PL-f0'), 'cnn-type': 'mobilenet', 'name_fold_csv': 'pseudo.csv', 'use-mean': ''}),
]
for i in range(1, 5):
    kwargs_list += [
        baseline({'fold-num': i, 'path-weights': get_wp(f'proj/cnn-b0-PL-f{i}'), 'cnn-type': 'b0', 'name_fold_csv': 'pseudo.csv'}),
        baseline({'fold-num': i, 'path-weights': get_wp(f'proj/cnn-mb-PL-f{i}'), 'cnn-type': 'mobilenet', 'name_fold_csv': 'pseudo.csv', 'use-mean': ''}),
    ]

# Define the base command
base_command = "python gen_pseudo_label.py"

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
