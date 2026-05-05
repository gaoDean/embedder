import wandb
with wandb.init(project="LeJEPA embedder") as run:
    # Indicate the artifact to use. Format is "name:alias"
    artifact = run.use_artifact("model-checkpoint-epoch-5:latest")
    datadir = artifact.download()
