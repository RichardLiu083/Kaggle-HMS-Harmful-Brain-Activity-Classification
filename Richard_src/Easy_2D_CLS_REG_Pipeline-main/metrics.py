import numpy as np
from sklearn.metrics import recall_score, roc_auc_score
import pandas as pd

__all__= ['Accuracy', 'Mean_Recall', 'AUC', 'kl_divergence']

def Accuracy(all_pred, all_label):
    all_pred= all_pred.argmax(1)
    acc= list((all_label==all_pred)+0).count(1) / len(all_label)
    return acc

def Mean_Recall(all_pred, all_label):
    all_pred= all_pred.argmax(1)
    recall= recall_score(all_label, all_pred, average='macro')
    return recall

def AUC(all_pred, all_label):
    auc= roc_auc_score(all_label, all_pred, multi_class='ovo')
    return auc


def kl_divergence(solution, submission, epsilon=10**-15, micro_average=True, sample_weights=None):
    solution= pd.DataFrame(solution)
    submission= pd.DataFrame(submission)
    
    for col in solution.columns:

        # Clip both the min and max following Kaggle conventions for related metrics like log loss
        # Clipping the max avoids cases where the loss would be infinite or undefined, clipping the min
        # prevents users from playing games with the 20th decimal place of predictions.
        submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

        y_nonzero_indices = solution[col] != 0
        solution[col] = solution[col].astype(float)
        solution.loc[y_nonzero_indices, col] = solution.loc[y_nonzero_indices, col] * np.log(solution.loc[y_nonzero_indices, col] / submission.loc[y_nonzero_indices, col])
        # Set the loss equal to zero where y_true equals zero following the scipy convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html#scipy.special.rel_entr
        solution.loc[~y_nonzero_indices, col] = 0

    if micro_average:
        return np.average(solution.sum(axis=1), weights=sample_weights)
    else:
        return np.average(solution.mean())