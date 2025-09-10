"""
Conversion script for ZKTP (Zero-Knowledge Task Planning) dataset to LeRobot format.

This script converts the custom ZKTP dataset format to LeRobot format for fine-tuning
π₀ and π₀-FAST models. The ZKTP dataset contains robot demonstrations with RGB images,
joint positions/commands, and gripper states/commands.

Usage:
uv run zktp/convert_zktp_data_to_lerobot.py --data_dir datasets/raw_dataset

The resulting dataset will be saved to datasets/zktp_lerobot_dataset.
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

REPO_NAME = "zktp_lerobot_dataset"  # Name of the output dataset


def load_episode_data(episode_dir: Path):
    """Load episode data from the ZKTP format."""
    # Find the JSON file in the episode directory
    json_files = list(episode_dir.glob("*.json"))
    if len(json_files) != 1:
        raise ValueError(f"Expected exactly one JSON file in {episode_dir}, found {len(json_files)}")
    
    json_file = json_files[0]
    with open(json_file, 'r') as f:
        episode_data = json.load(f)
    
    return episode_data


def get_rgb_image_path(motion_step, episode_dir: Path):
    """Get the full path to the RGB image for a motion step."""
    rgb_path = motion_step["rgb_image_path"]
    # Remove the prefix path and get the actual image filename
    image_filename = Path(rgb_path).name
    
    # Find the subfolder (it's the UUID part without the task suffix)
    json_files = list(episode_dir.glob("*.json"))
    json_name = json_files[0].stem
    # Extract the UUID part (everything before "_open" or similar task identifier)
    # The format is: 6aebd020-327a-415f-847d-c621c624ded2_open_the_bottle
    if '_' in json_name:
        uuid_part = json_name.split('_')[0]
    else:
        uuid_part = json_name
    subfolder = episode_dir / uuid_part
    
    full_image_path = subfolder / image_filename
    return full_image_path


def load_and_resize_image(image_path: Path, target_size=(256, 256)):
    """Load and resize an image to the target size."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize to target size
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            return np.array(img_resized)
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def main(data_dir: str):
    """Convert ZKTP dataset to LeRobot format."""
    data_path = Path(data_dir)
    
    # Output to local datasets directory instead of HuggingFace cache
    output_path = Path("datasets") / REPO_NAME
    
    # Clean up any existing dataset in the output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create LeRobot dataset with local path
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=output_path,  # Specify local path
        robot_type="xarm7",  # xArm7 with parallel gripper
        fps=5,  # Based on ~200ms intervals observed in the data
        features={
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),  # 7 joints + 1 gripper
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # 7 joint commands + 1 gripper command
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Process each episode
    episode_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    for episode_dir in episode_dirs:
        print(f"Processing episode: {episode_dir.name}")
        
        try:
            # Load episode data
            episode_data = load_episode_data(episode_dir)
            motion_data = episode_data["motion_data"]
            task_description = episode_data.get("natural_language_task", "")
            
            print(f"  Task: {task_description}")
            print(f"  Number of steps: {len(motion_data)}")
            
            # Process each motion step
            for i, step in enumerate(motion_data):
                # Get RGB image path and load image
                rgb_image_path = get_rgb_image_path(step, episode_dir)
                
                if not rgb_image_path.exists():
                    print(f"  Warning: Image not found: {rgb_image_path}")
                    continue
                
                # Load and process image
                try:
                    wrist_image = load_and_resize_image(rgb_image_path)
                except Exception as e:
                    print(f"  Warning: Failed to load image {rgb_image_path}: {e}")
                    continue
                
                # Prepare state (7 joints + 1 gripper)
                joint_positions = step["joint_positions"]  # Should be 7 values
                gripper_state = step["gripper_state"]  # Single value
                
                if len(joint_positions) != 7:
                    print(f"  Warning: Expected 7 joint positions, got {len(joint_positions)}")
                    continue
                
                state = joint_positions + [gripper_state]  # Combine to get 8 values
                
                # Prepare actions (7 joint commands + 1 gripper command)
                joint_commands = step["joint_commands"]  # Should be 7 values
                gripper_command = step["gripper_command"]  # Single value
                
                if len(joint_commands) != 7:
                    print(f"  Warning: Expected 7 joint commands, got {len(joint_commands)}")
                    continue
                
                actions = joint_commands + [gripper_command]  # Combine to get 8 values
                
                # Add frame to dataset
                dataset.add_frame(
                    {
                        "wrist_image": wrist_image,
                        "state": np.array(state, dtype=np.float32),
                        "actions": np.array(actions, dtype=np.float32),
                        "task": task_description,
                    }
                )
            
            # Save episode
            dataset.save_episode()
            print(f"  Successfully processed episode {episode_dir.name}")
            
        except Exception as e:
            print(f"  Error processing episode {episode_dir.name}: {e}")
            continue
    
    print(f"Dataset conversion complete. Saved to: {output_path}")
    print(f"Total episodes processed: {len([d for d in data_path.iterdir() if d.is_dir()])}")


if __name__ == "__main__":
    tyro.cli(main)
