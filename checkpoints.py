import os
import shutil
import glob
import hydra.utils
import wandb
import config as cfg
import torch
import re

def wandb_init():
    os.environ["WANDB_DIR"] = "/tmp/wandb"
    os.environ["WANDB_CACHE_DIR"] = "/tmp/wandb_cache"
    os.environ["WANDB_DATA_DIR"] = "/tmp/wandb_data"
    os.environ["WANDB_CONFIG_DIR"] = "/tmp/wandb_config"
    wandb.init(project=cfg.WANDB_PROJECT)

def wandb_log(step, lr, loss, time, el, rand_el):
    if wandb.run is not None:
        wandb.log(
                    {
                        "train/step": step,
                        "train/lr": lr,
                        "train/loss": loss,
                        "train/time": time,
                        "eval/el": el,
                        "eval/rand_el": rand_el,
                        "eval/el_real": el - rand_el,
                    }
                )

def get_latest_checkpoint():
    '''
    if no checkpoints, returns None
    '''

    orig_cwd = os.getcwd()

    checkpoints = glob.glob(os.path.join(orig_cwd, "checkpoint_epoch_*.pt"))

    if checkpoints:
        latest_cp = max(checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))

        return latest_cp

    return None

def save_checkpoint(checkpoint, upload=False):
    orig_cwd = os.getcwd()

    checkpoint_path = os.path.join(orig_cwd, f"checkpoint_epoch_{checkpoint['epoch']}.pt")
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved for epoch {checkpoint['epoch']} step {checkpoint['step']}")

    if upload and wandb.run is not None:
        artifact = wandb.Artifact(f"model-checkpoint-{checkpoint['epoch']}-{checkpoint['step']}", type="model")
        artifact.add_file(checkpoint_path)
        
        logged_artifact = wandb.log_artifact(artifact)
        logged_artifact.wait()  # Block until the upload finishes

        # Clean up wandb caches (both download cache and staging data cache)
        for cache_env in ["WANDB_CACHE_DIR", "WANDB_DATA_DIR"]:
            cache_dir = os.environ.get(cache_env)
            if cache_dir and os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)

        print(f"Checkpoint uploaded to wandb for epoch {checkpoint['epoch']}")
