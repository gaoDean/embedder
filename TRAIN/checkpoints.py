import os
import glob
import hydra.utils
import wandb
import config as cfg

def wandb_init():
    wandb.init(project="Inverse Embedder", config=dict(cfg))

def get_lastest_checkpoint():
    '''
    if no checkpoints, returns None
    '''

    orig_cwd = hydra.utils.get_original_cwd()

    checkpoints = glob.glob(os.path.join(orig_cwd, "checkpoint_epoch_*.pt"))

    if checkpoints:
        latest_cp = max(checkpoints, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))

        return latest_cp

    return None

def save_checkpoint(
    epoch,
    model_state,
    optimiser_state,
    scheduler_state,
    scalar_state,
):
    orig_cwd = hydra.utils.get_original_cwd()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimiser_state,
        'scheduler_state_dict': scheduler_state,
        'scaler_state_dict': scaler_state
    }

    checkpoint_path = os.path.join(orig_cwd, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved for epoch {epoch}")

    if wandb.run is not None:

        artifact = wandb.Artifact(f"model-checkpoint-epoch-{epoch}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

        print(f"Checkpoint uploaded to wandb for epoch {epoch}")
