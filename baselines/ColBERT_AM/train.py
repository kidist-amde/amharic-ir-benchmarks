import json
import torch
import torch.distributed as dist
import os
from ColBERT_AM.infra import Run
from ColBERT_AM.training.training import train
from ColBERT_AM.infra.config import ColBERTConfig
from ColBERT_AM.utils.utils import print_message
import argparse

# Add an argument parser to accept checkpoint directory
parser = argparse.ArgumentParser()
parser.add_argument("--savepath", type=str, default=None, help="Directory to save checkpoints")
args = parser.parse_args()


def setup_distributed():
    """Initializes distributed training."""
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))  
        world_size = int(os.environ.get("WORLD_SIZE", 1))  
        print(f"Initializing process group: rank={rank}, world_size={world_size}")
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        
        # FIX: Ensure valid GPU index (avoid invalid device ordinal)
        torch.cuda.set_device(rank % torch.cuda.device_count())

setup_distributed()

# Load configuration
config_path = "amharic_colbert/infra/config/training_config.json"

try:
    with open(config_path, "r") as f:
        config_dict = json.load(f)
except FileNotFoundError:
    raise ValueError(f"Configuration file not found at {config_path}. Please check the path.")

# Initialize ColBERT configuration
config = ColBERTConfig(**config_dict)

# Initialize Run instance
run = Run()
rank = int(os.environ.get("RANK", 0))

if rank == 0:
    with run.context(config):  # Only rank 0 logs
        try:
            # Train the model, passing all required arguments
            checkpoint_path = train(
                config,
                config.triples,
                queries=config.queries,
                collection=config.collection,
                savepath=args.savepath  
            )
            
            print(f"Model trained and saved at {checkpoint_path}")
        except Exception as e:
            print_message(f"Error during training: {e}")
            raise
else:
    # Other ranks only train, without logging
    try:
        train(config, config.triples, queries=config.queries, collection=config.collection)
    except Exception as e:
        raise RuntimeError(f"Rank {rank} encountered an error: {e}")
