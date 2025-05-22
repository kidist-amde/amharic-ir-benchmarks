import os
import datetime
import torch

from ColBERT_AM.utils.utils import print_message, save_checkpoint
from ColBERT_AM.parameters import SAVED_CHECKPOINTS
from ColBERT_AM.infra.run import Run

def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)


def manage_checkpoints(args, colbert, optimizer, batch_idx, savepath=None, consumed_all_triples=False):

    # Use provided `savepath`, otherwise generate a default path
    if savepath:
        checkpoints_path = savepath  # Passed from `train.py`
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoints_path = os.path.join(Run().path_, f'checkpoints-{timestamp}')

    # Ensure the directory exists
    os.makedirs(checkpoints_path, exist_ok=True)
    # print(f"Checkpoints will be saved in: {checkpoints_path}")

    try:
        save = colbert.save
    except:
        save = colbert.module.save

    path_save = None

    if consumed_all_triples or (batch_idx % 2000 == 0):
        path_save = os.path.join(checkpoints_path, "colbert")

    if batch_idx in SAVED_CHECKPOINTS:
        path_save = os.path.join(checkpoints_path, f"colbert-{batch_idx}")

    if path_save:
        print(f"Saving a checkpoint to {path_save} ..")
        save(path_save)

    return path_save

