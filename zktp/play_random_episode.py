"""
Script to play a random episode from the converted ZKTP LeRobot dataset.

This script loads the converted dataset, samples a random episode, and displays
the wrist camera images along with the corresponding robot state and actions.

Usage:
uv run zktp/play_random_episode.py
"""

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import json


def play_random_episode(dataset_path: str = "datasets/zktp_lerobot_dataset"):
    """Load and play a random episode from the ZKTP dataset."""
    
    # Load the dataset from local path
    print(f"Loading dataset from: {dataset_path}")
    # Convert to absolute path to avoid confusion
    dataset_path = Path(dataset_path).resolve()
    print(f"Absolute path: {dataset_path}")
    
    # Load as local dataset by specifying root parameter
    dataset = LeRobotDataset(
        repo_id="zktp_lerobot_dataset",
        root=dataset_path
    )
    
    print(f"Dataset info:")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Total episodes: {dataset.num_episodes}")
    print(f"  Features: {list(dataset.features.keys())}")
    
    # Load episode lengths from meta/episodes.jsonl
    episodes_meta_path = dataset_path / "meta" / "episodes.jsonl"
    episode_lengths = []
    episode_tasks = []
    with open(episodes_meta_path, 'r') as f:
        for line in f:
            episode_data = json.loads(line)
            episode_lengths.append(episode_data['length'])
            episode_tasks.append(episode_data['tasks'][0] if episode_data['tasks'] else "Unknown task")
    
    # Compute cumulative lengths to get start indices
    cumulative_lengths = [0] + list(np.cumsum(episode_lengths))
    
    # Sample a random episode by checking episode indices
    episode_idx = random.randint(0, dataset.num_episodes - 1)
    
    print(f"\nPlaying episode {episode_idx}")
    
    # Get episode start and end indices
    start_idx = cumulative_lengths[episode_idx]
    end_idx = cumulative_lengths[episode_idx + 1]
    episode_length = episode_lengths[episode_idx]
    
    # Load all frames for the episode
    episode_frames = [dataset[i] for i in range(start_idx, end_idx)]
    
    print(f"Episode length: {episode_length} frames")
    
    # Extract task description
    task = episode_tasks[episode_idx]
    print(f"Task: {task}")
    # Setup matplotlib for interactive display
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Episode {episode_idx}: {task}', fontsize=14)
    
    # Axes layout: [image, state plot], [action plot, gripper plot]
    img_ax = axes[0, 0]
    state_ax = axes[0, 1]
    action_ax = axes[1, 0]
    gripper_ax = axes[1, 1]
    
    # Initialize plots
    img_display = None
    state_lines = []
    action_lines = []
    gripper_line = None
    
    # Setup state plot (7 joint positions)
    state_ax.set_title('Joint Positions (State)')
    state_ax.set_xlabel('Time Step')
    state_ax.set_ylabel('Joint Angle (rad)')
    for i in range(7):
        line, = state_ax.plot([], [], label=f'Joint {i+1}')
        state_lines.append(line)
    state_ax.legend()
    state_ax.grid(True)
    
    # Setup action plot (7 joint commands)
    action_ax.set_title('Joint Commands (Actions)')
    action_ax.set_xlabel('Time Step')
    action_ax.set_ylabel('Joint Command (rad)')
    for i in range(7):
        line, = action_ax.plot([], [], label=f'Joint {i+1}')
        action_lines.append(line)
    action_ax.legend()
    action_ax.grid(True)
    
    # Setup gripper plot
    gripper_ax.set_title('Gripper State/Command')
    gripper_ax.set_xlabel('Time Step')
    gripper_ax.set_ylabel('Gripper Value')
    gripper_state_line, = gripper_ax.plot([], [], 'b-', label='State')
    gripper_cmd_line, = gripper_ax.plot([], [], 'r--', label='Command')
    gripper_ax.legend()
    gripper_ax.grid(True)
    
    # Collect data for plotting
    timestamps = list(range(episode_length))
    
    # Extract states and actions from episode frames
    states = np.array([frame['state'] for frame in episode_frames])
    actions = np.array([frame['actions'] for frame in episode_frames])
    
    # Separate joint and gripper data
    joint_states = states[:, :7]  # First 7 are joint positions
    joint_actions = actions[:, :7]  # First 7 are joint commands
    gripper_states = states[:, 7]  # 8th is gripper state
    gripper_commands = actions[:, 7]  # 8th is gripper command
    
    # Set plot limits
    state_ax.set_xlim(0, episode_length)
    state_ax.set_ylim(joint_states.min() - 0.1, joint_states.max() + 0.1)
    
    action_ax.set_xlim(0, episode_length)
    action_ax.set_ylim(joint_actions.min() - 0.1, joint_actions.max() + 0.1)
    
    gripper_ax.set_xlim(0, episode_length)
    gripper_min = min(gripper_states.min(), gripper_commands.min())
    gripper_max = max(gripper_states.max(), gripper_commands.max())
    gripper_ax.set_ylim(gripper_min - 50, gripper_max + 50)
    
    print(f"\nStarting playback... Press Ctrl+C to stop")
    print("Close the matplotlib window to end playback")
    
    try:
        for frame_idx in range(episode_length):
            # Update image
            wrist_image = episode_frames[frame_idx]['wrist_image']
            
            # Convert from CHW to HWC format if needed
            if wrist_image.shape[0] == 3:  # If channels are first
                wrist_image = np.transpose(wrist_image, (1, 2, 0))  # CHW -> HWC
            
            if img_display is None:
                img_display = img_ax.imshow(wrist_image)
                img_ax.set_title(f'Wrist Camera (Frame {frame_idx + 1}/{episode_length})')
                img_ax.axis('off')
            else:
                img_display.set_data(wrist_image)
                img_ax.set_title(f'Wrist Camera (Frame {frame_idx + 1}/{episode_length})')
            
            # Update joint state plots
            current_timestamps = timestamps[:frame_idx + 1]
            for i in range(7):
                current_states = joint_states[:frame_idx + 1, i]
                state_lines[i].set_data(current_timestamps, current_states)
            
            # Update joint action plots  
            for i in range(7):
                current_actions = joint_actions[:frame_idx + 1, i]
                action_lines[i].set_data(current_timestamps, current_actions)
            
            # Update gripper plots
            current_gripper_states = gripper_states[:frame_idx + 1]
            current_gripper_commands = gripper_commands[:frame_idx + 1]
            gripper_state_line.set_data(current_timestamps, current_gripper_states)
            gripper_cmd_line.set_data(current_timestamps, current_gripper_commands)
            
            # Add current frame indicator
            for ax in [state_ax, action_ax, gripper_ax]:
                # Remove previous frame indicator
                for line in ax.lines:
                    if hasattr(line, '_is_frame_indicator'):
                        line.remove()
                
                # Add new frame indicator
                ylim = ax.get_ylim()
                frame_line = ax.axvline(x=frame_idx, color='red', linestyle=':', alpha=0.7)
                frame_line._is_frame_indicator = True
            
            plt.draw()
            plt.pause(1.0 / 5.0)  # Play at 5 FPS (dataset is recorded at 5 FPS)
            
            # Check if window was closed
            if not plt.get_fignums():
                break
                
    except KeyboardInterrupt:
        print("\nPlayback stopped by user")
    except Exception as e:
        print(f"\nError during playback: {e}")
    finally:
        plt.ioff()
        if plt.get_fignums():
            plt.show()


def main():
    """Main function to run the episode player."""
    play_random_episode()


if __name__ == "__main__":
    main()
