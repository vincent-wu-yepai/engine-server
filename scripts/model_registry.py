import wandb
import os
from dotenv import load_dotenv


load_dotenv()


def create_artifact(s3_model_path: str):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(project="ConditionalUNet", entity="vincent-wu-yepai-yepai")

    # Create an artifact
    artifact = wandb.Artifact("unet", type="model")

    # Add References to the artifact
    artifact.add_reference(os.path.join(s3_model_path, "unet_config.json"))
    artifact.add_reference(os.path.join(s3_model_path, "pytorch_model.bin"))
    
    # Log the artifact to the run
    version = run.log_artifact(artifact)
    
    # Wait for the artifact to be uploaded to the W&B server
    artifact.wait()
    
    # Tag the artifact with "prod" alias
    artifact.aliases.append("prod")
    
    # Save the artifact to the W&B server
    artifact.save()


if __name__ == "__main__":
    version = "your_model_version"  # example: v0
    s3_model_path = f"s3://digital-human-llm/dev/models/musetalk/unet/{version}/"
    create_artifact(s3_model_path)
