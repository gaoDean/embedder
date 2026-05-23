import os
import glob
import hydra.utils
import wandb
import config as cfg

def wandb_init():
    wandb.init(project=cfg.WANDB_PROJECT)

def wandb_log(step, lr, loss, time, el):
    if wandb.run is not None:
        wandb.log(
                    {
                        "train/step": step,
                        "train/lr": lr,
                        "train/loss": loss,
                        "train/time": time,
                        "eval/el": el
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

    checkpoint_path = os.path.join(orig_cwd, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved for epoch {epoch}")

    if wandb.run is not None:

        artifact = wandb.Artifact(f"model-checkpoint-epoch-{epoch}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

        print(f"Checkpoint uploaded to wandb for epoch {epoch}")
