import argparse
import functools
import os
import random
import signal
import sys
import time
from distutils.util import strtobool
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import loralib as lora
from torch import int8
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from envpool.vizdoom.registration import register_custom_folder

import torch
import numpy as np
import cv2
import envpool  # Assuming envpool is being used for the environment
import imageio
import pynvml # Monitoring GPU

from transformers import AutoModelForImageClassification
from transformers import SwinForImageClassification
from peft import LoraConfig, get_peft_model, TaskType

import threading
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps
import torch
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

def get_most_free_gpu():
    """Finds the GPU with the most free memory using nvidia-smi."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Run nvidia-smi to get free memory for each GPU
        output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True)
        free_memory = [int(x) for x in output.decode("utf-8").strip().split("\n")]
        
        # Find the index of the GPU with the highest free memory
        best_gpu = free_memory.index(max(free_memory))
        return best_gpu, free_memory[best_gpu]

    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        return None, None

# Funzione per ottenere la memoria della GPU
def get_gpu_memory(device):
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(device)
    return {
        'total': mem_info.total / 1024**2,  # Memoria totale in MB
        'used': mem_info.used / 1024**2,    # Memoria usata in MB
        'free': mem_info.free / 1024**2     # Memoria libera in MB
    }
    
def save_frames_as_gif(frames, filename="episode_recording.gif"):
    """z
    Saves a list of frames as a GIF file.

    Args:
        frames: list of numpy arrays representing frames
        filename: output filename
    """
    if len(frames) > 0:
        processed_frames = []
        for frame in frames:
            # Ensure frame is in HWC format (height, width, channels)
            if frame.shape[0] == 3:  # If in CHW format
                frame = np.transpose(frame, (1, 2, 0))

            # Ensure values are in uint8 range [0, 255]
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            processed_frames.append(frame)
        imageio.mimsave(filename, processed_frames, fps=120)
        print(f"Saved recording to {filename}")
    else:
        print("No frames were recorded")


def log_matrix_with_heatmap(matrix, tasks, name, vmin=0, vmax=1, cmap='YlOrRd'):
    """
    Create and log a heatmap visualization of a matrix to wandb

    Args:
        matrix: numpy array containing the matrix data
        tasks: list of task names for axes labels
        name: name for the wandb log
        vmin: minimum value for color scaling
        vmax: maximum value for color scaling
        cmap: matplotlib colormap name
    """
    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(matrix,
                xticklabels=tasks,
                yticklabels=tasks,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                annot=True,  # Show values in cells
                fmt='.2f',  # Format for cell values
                cbar_kws={'label': 'Score'})

    plt.title(f'{name} Heatmap')
    plt.xlabel('Target Task')
    plt.ylabel('Source Task')
    plt.tight_layout()

    # Log to wandb
    wandb.log({f"{name}_heatmap": wandb.Image(plt)})
    plt.close()


def test(model, test_envs, env_names, global_step, save_gif=False, trackmatrix=False):
    print(f"Testing - Global Steps: {global_step} Time {time.time() - start_time}")
    model.eval()

    with torch.no_grad():
        for i, test_env in enumerate(test_envs):

            next_obs, _ = test_env.reset()
            next_obs = torch.Tensor(next_obs).to(device)
            episode_rewards = np.zeros(10)
            episode_len = np.zeros(10)

            sum_len = 0
            sum_reward = 0
            count_done = 0
            kills = 0

            frames = []
            ep0_end = False

            for _ in range(0, 1250):
                # Action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)

                # Execute game step
                next_obs, reward, terminated, truncated, info = test_env.step(action.cpu().numpy())

                next_done = torch.tensor(terminated | truncated).to(int8).numpy()
                if not ep0_end:
                    frames.append(next_obs[0][-3:])
                    #frames.append(next_obs[0][:3])
                    #frames.append(next_obs[0][3:6])
                    #frames.append(next_obs[0][6:9])
                    #frames.append(next_obs[0][9:])

                episode_rewards += reward
                episode_len += 1

                if next_done[0]:
                    ep0_end = True

                count_done += sum(next_done)
                sum_reward += sum(episode_rewards[next_done == 1])
                kills += sum(info["KILLCOUNT_TOTAL"][next_done == 1])
                sum_len += sum(episode_len[next_done == 1])
                episode_rewards[next_done == 1] = 0
                episode_len[next_done == 1] = 0

                # Process next observation
                next_obs = torch.Tensor(next_obs).to(device)
                if count_done >= 10: break

            mean_return = sum_reward / count_done
            mean_len = sum_len / count_done
            kills = kills / count_done
            success = (kills - 3.5) / 26.5

            print(
                f"{env_names[i]} - global_step={global_step}, mean_episodic_return={mean_return:.2f}, mean_episodic_len={mean_len}, Kills={kills:.2f}, Success={success:.2f}")
            writer.add_scalar(f"{env_names[i]}/episode_len", mean_len, global_step)
            writer.add_scalar(f"{env_names[i]}/reward", mean_return, global_step)
            writer.add_scalar(f"{env_names[i]}/kills", kills, global_step)
            writer.add_scalar(f"{env_names[i]}/success", success, global_step)
            # try:
            if save_gif: 
                os.makedirs("gifs", exist_ok=True)
                save_frames_as_gif(frames=frames, filename=f"gifs/{env_names[i]}_{global_step}.gif")
            # except:
            #     print("Error saving image")

            if trackmatrix:
                results_matrix[current_task][i] = kills
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_vanilla",
                        help="the name of this experiment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-5,
                        help="the learning rate of the optimizer")# 1.0e-3 lr, 2.5e-4 default, 1.0e-4 lrl, 2.5e-5 lrl--
    parser.add_argument("--seed", type=int, default=9,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=99,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--offline", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="VMRdata",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="marcolbilli-universit-di-firenze",
                        help="the entity (team) of wandb's project")

    parser.add_argument("--env-floder", type=str, default=os.getcwd() + '/run_and_gun',
                        help="Folder with custom maps")

    parser.add_argument("--s-p", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this use shrink and perturb")
    parser.add_argument("--ewc", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this use elastic weight consolidation")
    
    # Agent specific arguments
    parser.add_argument("--network-type", type=str, default="cnn", nargs="?", const="cnn",
                        help="the network architecture")
    parser.add_argument("--ac-mlp", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, actor and critic are mlp otherwise linear")
    parser.add_argument("--pretrained-adapt", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, you pass 224x224 images to network")
    parser.add_argument("--forward-type", type=str, default="single_frame", nargs="?", const="single_frame",
                        help="how input frames are passed to the network")#single_frame, multi_frame_patch_concat, conv_adapter
    parser.add_argument("--use-lora", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, you use LoRA in pretrained nets")
    
     # LoRA specific arguments
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="the rank variable of LoRA")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="the alpha variable of LoRA")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=32,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--updates-per-env", type=int, default=500,
                        help="the number of steps to run in each environment")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
                        help="the number of mini-batches") #32 per swin, 4 per resnet, 8 per cnn
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy") #4 per swin, 4 resnet
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

'''
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(12, 32, 8, stride=4),  # Adjusted for RGB input
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.actor = nn.Linear(256, 12)
        self.critic = nn.Linear(256, 1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
'''

class Agent(nn.Module):
    def __init__(self, observation_space_shape, num_actions, network_type="cnn", actor_critic_mlp=False, pretrained_adapt=False, forward_type="single_frame", use_lora=False):
        super().__init__()
        self.network_type = network_type
        self.actor_critic_mlp = actor_critic_mlp
        self.pretrained_adapt = pretrained_adapt
        self.forward_type = forward_type
        self.use_lora = use_lora
        self.args = parse_args()
        
        print(f"Network Type : {self.network_type}")
        print(f"Actor-Critic is MLP : {self.actor_critic_mlp}")
        print(f"Pretrain Adaptation (resize images to 224x224) : {self.pretrained_adapt}")
        print(f"Forward Type : {self.forward_type}")
        print(f"Using LoRA : {self.use_lora}")
        match self.network_type:
            case "cnn":
                if self.forward_type == "conv_adapter":
                    obs_space = observation_space_shape[0]
                elif self.forward_type == "single_frame":
                    obs_space = 3
                self.network = nn.Sequential(
                    nn.Conv2d(obs_space, 32, 8, stride=4),  # Adjusted for RGB input
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.LazyLinear(256),
                    nn.LayerNorm(256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 256),
                    nn.LeakyReLU(),
                )
            case "resnet_s":
                self.network = models.resnet18(weights=None)
                self.conv_adapter = nn.Conv2d(observation_space_shape[0], 3, kernel_size=1)
                self.network.fc = nn.Identity()
            case "resnet_w":
                self.network = models.resnet18(weights="IMAGENET1K_V1")
                self.conv_adapter = nn.Conv2d(observation_space_shape[0], 3, kernel_size=1)
                self.network.fc = nn.Identity()
                '''
                for param in self.network.conv1.parameters():
                    param.requires_grad = False
                for param in self.network.layer1.parameters():
                    param.requires_grad = True
                for param in self.network.layer2.parameters():
                    param.requires_grad = True
                for param in self.network.layer3.parameters():
                    param.requires_grad = True
                for param in self.network.layer4.parameters():
                    param.requires_grad = True
                '''
            case "swin_s":
                self.network = models.swin_transformer.swin_t(weights=None)
                self.conv_adapter = nn.Conv2d(observation_space_shape[0], 3, kernel_size=1)
                self.network.head = nn.Identity()
            case "swin_w":
                self.network = models.swin_transformer.swin_t(weights="IMAGENET1K_V1")
                self.conv_adapter = nn.Conv2d(observation_space_shape[0], 3, kernel_size=1)
                self.network.head = nn.Identity()
                if self.use_lora:
                    self.apply_lora(self.network)
            case "swin_w_hf":
                self.network = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
                self.conv_adapter = nn.Conv2d(observation_space_shape[0], 3, kernel_size=1)
                self.network.classifier = nn.Identity()
                if self.use_lora:
                    self.apply_lora(self.network)
            case default:#TODO
                self.network = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(observation_space_shape[0] * observation_space_shape[1] * observation_space_shape[2], 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU()
                )
        
        #print(self.network)        
        #print(self.adapt_input(torch.randn(1, *observation_space_shape)).shape)
        #print(self.network(self.adapt_input(torch.randn(1, *observation_space_shape))).shape)
        if self.network_type in ["cnn","resnet_s","swin_s","resnet_w","swin_w"]:
            self.output_features = self.forward_backbone(self.adapt_input(torch.randn(1, *observation_space_shape))).shape[1]
        elif self.network_type in ["swin_w_hf"]:
            self.output_features = self.forward_backbone(self.adapt_input(torch.randn(1, *observation_space_shape))).shape[1]
                
        if actor_critic_mlp:
            self.actor = nn.Sequential(
                nn.Linear(self.output_features, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions)
            )
            self.critic = nn.Sequential(
                nn.Linear(self.output_features, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        else:
            self.actor = nn.Linear(self.output_features, num_actions)
            self.critic = nn.Linear(self.output_features, 1)
            
    def adapt_input(self, obs):
        if self.pretrained_adapt:
            obs = nn.functional.interpolate(obs, size=(224, 224), mode="nearest")
        #print(obs.shape)    
        return obs

    def forward_backbone(self, obs):
        match self.forward_type:
            case "single_frame":
                x = obs[:, -3:, :, :]
                #print(x.shape)
                if self.network_type in ["cnn","resnet_s","swin_s","resnet_w","swin_w"]:
                    features = self.network(x / 255.0)
                elif self.network_type in ["swin_w_hf"]:
                    features = self.network(x / 255.0).logits
                #print(features.shape)
                return features
            case "multi_frame_patch_concat":
                bs, c, h, w = obs.shape
                num_frames = 4
                # Splitta l'input in 4 frame separati
                frames = obs.view(bs, num_frames, c // num_frames, h, w)  # (batch_size, num_frames, 3, H, W)
                #print(frames.shape)
                # Processa ogni frame indipendentemente
                all_patches = []
                for i in range(4):  
                    frame = frames[:, i, :, :, :]  # Estrai il frame i-esimo (batch_size, 3, 84, 84)
                    #print(frame.shape)
                    if self.network_type in ["resnet_s","swin_s","resnet_w","swin_w"]:
                        print("Not implemented")
                    elif self.network_type in ["swin_w_hf"]:
                        with torch.no_grad():
                            embeddings = self.network.swin.embeddings(frame / 255.0)[0]
                            all_patches.append(embeddings)
                        #print(self.network.swin.embeddings(frame / 255.0)[0])
                concatenated_patches = torch.cat(all_patches, dim=1)
                #print(f"First-layer patch embeddings shape: {concatenated_patches.shape}")
                features = []
                if self.network_type in ["swin_w_hf"]:
                    outputs = self.network.swin.encoder(concatenated_patches, input_dimensions=(h, int(w/4)))
                    #RICORDA DI GUARDARE IL FORWARD DEL MODELLO ORIGINALE PER CAPIRE COME ESTRARRE I LOGITS
                    outputs = self.network.swin.layernorm(outputs.last_hidden_state)
                    #print(outputs.shape)
                    pooled_output = self.network.swin.pooler(outputs.transpose(1, 2))
                    pooled_output = torch.flatten(pooled_output, 1)
                    #print(pooled_output.shape)
                    features = self.network.classifier(pooled_output)
                #print(f"Final output shape after Swin encoder: {features.shape}")
                return features
            case "multi_frame_avg":
                bs, c, h, w = obs.shape
                num_frames = 4
                # Splitta l'input in 4 frame separati
                frames = obs.view(bs, num_frames, c // num_frames, h, w)  # (batch_size, num_frames, 3, H, W)
                #print(frames.shape)
                # Processa ogni frame indipendentemente
                outputs = []
                for i in range(4):  
                    frame = frames[:, i, :, :, :]  # Estrai il frame i-esimo (batch_size, 3, 84, 84)
                    #print(frame.shape)
                    if self.network_type in ["resnet_s","swin_s","resnet_w","swin_w"]:
                        outputs.append(self.network(frame / 255.0))
                    elif self.network_type in ["swin_w_hf"]:
                        outputs.append(self.network(frame / 255.0).logits)

                # Concatena i risultati e fai la media sui frame
                outputs = torch.stack(outputs, dim=0)  # (num_frames, batch_size, num_classes)
                #print(outputs.shape)
                features = outputs.mean(dim=0)  # (batch_size, num_classes)
                print(features.shape)
                return features
            case "conv_adapter":
                #print(x.shape)
                if self.network_type in ["cnn"]:
                    features = self.network(obs / 255.0)
                elif self.network_type in ["resnet_s","swin_s","resnet_w","swin_w"]:
                    x = self.conv_adapter(obs)
                    features = self.network(x / 255.0)
                elif self.network_type in ["swin_w_hf"]:
                    x = self.conv_adapter(obs)
                    features = self.network(x / 255.0).logits
                return features 
    
    def apply_lora(self, model, rank=32):
        if self.network_type == "swin_w":
            for param in model.parameters():
                param.requires_grad = False
                
            '''
            # Allena solo layer di attenzione e fully connected
            for name, param in model.named_parameters():
                if "attn" in name or "fc" in name:
                    param.requires_grad = True
            '''
            
            modules_copy = list(model.named_modules())  

            for name, module in modules_copy:
                if isinstance(module, nn.Linear) and ("attn" in name or "fc" in name):
                    setattr(model, name, lora.Linear(module.in_features, module.out_features, r=self.args.lora_rank, lora_alpha=self.args.lora_alpha))
                    print(f"Applied LoRA to {name}")
                    
            for name, module in model.named_modules():
                if "attn.qkv" in name or "attn.proj" in name:
                    print(f"{name}: {module.__class__}")  

            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True  
                    
            if hasattr(model, "head"):  
                for param in model.head.parameters():
                    param.requires_grad = True  
                print("Unfrozen Swin head")
        elif self.network_type == "swin_w_hf":
            targets = []
            supported_modules = (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, torch.nn.Conv3d)

            '''
            # Itera sui moduli del modello
            for name, module in model.named_modules():
                if isinstance(module, supported_modules) and ('attention' in name or 'dense' in name):
                        targets.append(name)
            
            print(targets)
            '''
            
            # Definisci la configurazione di LoRA
            lora_config = LoraConfig(
                r=self.args.lora_rank,  # Riduzione della dimensione del rank
                lora_alpha=self.args.lora_alpha,  # Fattore di scalatura
                lora_dropout=0.1,  # Dropout per LoRA
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["query", "key", "value", "dense"],  # Layer target (attenzione e fully connected)
            )
            
            # Applica LoRA al modello usando PEFT
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
                
        #print(model)
            
    def get_value(self, x):
        x = self.adapt_input(x)
        hidden = self.forward_backbone(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        x = self.adapt_input(x)
        hidden = self.forward_backbone(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}"

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    

    tasks = ["Default-Conf-v1"] #["Obstacles-v1", "Green-v1", "Resized-v1", "Monsters-v1", "Default-v1", "Red-v1", "Blue-v1", "Shadows-v1"]
    current_task = 0
    register_custom_folder(args.env_floder)

    infinite_ammo = False
    terminate_on_ammo_depletion = True
    initial_ammo = 100
    max_episode_steps = 1250

    batch_size = args.num_envs

    dict_envs = dict(zip(tasks, [None for _ in range(len(tasks))]))
    dict_test_envs = dict(zip(tasks, [None for _ in range(len(tasks))]))

    print("\n\nCreating main environments...")
    envs_cont = []
    for task in tasks:
        env = envpool.make(
            task,
            env_type="gymnasium",
            num_envs=args.num_envs,
            batch_size=batch_size,
            use_combined_action=True,
            max_episode_steps=max_episode_steps,
            infinite_ammo=infinite_ammo,
            terminate_on_ammo_depletion=terminate_on_ammo_depletion,
            initial_ammo=initial_ammo,
        )
        print("---------------------------------------")
        print(task)
        main_env_observation_space = env.observation_space
        print("Observation Space Shape : " + str(main_env_observation_space.shape))
        main_env_action_space = env.action_space
        print("Action Space Number : " + str(main_env_action_space.n))
        envs_cont.append(env)

    print("---------------------------------------")
    envs = envs_cont[current_task]
    observation_space_shape = envs.observation_space.shape
    action_space_number = envs.action_space.n
    #print(envs_cont)
    print("\n\nCreating test environments...")
    test_envs = []
    for task in tasks:
        env = envpool.make(
            task,
            env_type="gymnasium",
            num_envs=10,
            use_combined_action=True,
            max_episode_steps=max_episode_steps,
            infinite_ammo=infinite_ammo,
            terminate_on_ammo_depletion=terminate_on_ammo_depletion,
            initial_ammo=initial_ammo,
        )
        print("---------------------------------------")
        print(task)
        test_env_observation_space = env.observation_space
        print("Observation Space Shape : " + str(test_env_observation_space.shape))
        test_env_action_space = env.action_space
        print("Action Space Number : " + str(test_env_action_space.n))
        test_envs.append(env)

    print("---------------------------------------")
    # Wandb setup
    if args.track:
        import wandb
        mode = "offline" if args.offline else "online"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            mode=mode,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    print("\n\nCUDA available")
    print("---------------------------------------")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    best_gpu, free_mem = get_most_free_gpu()
    if best_gpu is not None:
        print(f"Using GPU {best_gpu} with {free_mem} MB free.")
        device = torch.device(f"cuda:{best_gpu}")
    else:
        print("No GPU available, using CPU.")
        device = torch.device("cpu")
    pynvml.nvmlInit()
    device_nvml = pynvml.nvmlDeviceGetHandleByIndex(best_gpu)
    memory = get_gpu_memory(device_nvml)
    print(f"Memoria iniziale: Totale = {memory['total']} MB, Usata = {memory['used']} MB, Libera = {memory['free']} MB")
    print("---------------------------------------")
        
    print("\n\nAgent specific arguments")    
    print("---------------------------------------")
    #agent = Agent(envs).to(device)
    agent = Agent(observation_space_shape, 
                  action_space_number, 
                  network_type=args.network_type, 
                  actor_critic_mlp=args.ac_mlp, 
                  pretrained_adapt=args.pretrained_adapt, 
                  forward_type=args.forward_type, 
                  use_lora=args.use_lora).to(device)
    print("---------------------------------------\n\n")
    '''
    for name, param in agent.network.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")
    
    print(f"Memoria allocata per il modello: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    print(f"Memoria riservata per il modello: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
    '''
    memory = get_gpu_memory(device_nvml)
    print(f"Memoria dopo creazione modello: Totale = {memory['total']} MB, Usata = {memory['used']} MB, Libera = {memory['free']} MB")
           
    if args.ewc:
        ewc = EWC(agent, ewc_lambda=250)
        

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    '''
    optimizer = optim.Adam([
        {"params": agent.conv_adapter.parameters(), "lr": args.learning_rate},
        {"params": agent.network.layer1.parameters(), "lr": args.learning_rate * 0.1},  # Layer iniziali meno allenati
        {"params": agent.network.layer2.parameters(), "lr": args.learning_rate * 0.5},  # Un po' più di LR
        {"params": agent.network.layer3.parameters(), "lr": args.learning_rate},        # LR normale
        {"params": agent.network.layer4.parameters(), "lr": args.learning_rate},        # LR normale
        {"params": agent.actor.parameters(), "lr": args.learning_rate},        # LR normale
        {"params": agent.critic.parameters(), "lr": args.learning_rate},   
    ], eps=1e-5)
    '''
    results_matrix = np.zeros([len(test_envs), len(test_envs)])

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device).to(torch.int64)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device).to(torch.bool)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    memory = get_gpu_memory(device_nvml)
    print(f"Memoria dopo storage: Totale = {memory['total']} MB, Usata = {memory['used']} MB, Libera = {memory['free']} MB")
    
    # Take a gif of an observation sample #TRY
    observation_sample = False
    if(observation_sample):
        envs.async_reset()
        next_obs, _, _, _, _ = envs.recv()
        '''
        frames=[]
        np.set_printoptions(threshold=np.inf, linewidth=200)
        print(next_obs[0][:3])
        print(next_obs[0][:3].shape)
        frames.append(next_obs[0][:3])
        frames.append(next_obs[0][3:6])
        frames.append(next_obs[0][6:9])
        frames.append(next_obs[0][9:])
        '''
        if(args.pretrained_adapt):
            next_obs = agent.adapt_input(torch.from_numpy(next_obs))
            
        save_frames_as_gif(frames=[next_obs[0][:3].cpu().numpy()], filename=f"gifs/observation_sample_frame1.gif")
        save_frames_as_gif(frames=[next_obs[0][3:6].cpu().numpy()], filename=f"gifs/observation_sample_frame2.gif")
        save_frames_as_gif(frames=[next_obs[0][6:9].cpu().numpy()], filename=f"gifs/observation_sample_frame3.gif")
        save_frames_as_gif(frames=[next_obs[0][9:].cpu().numpy()], filename=f"gifs/observation_sample_frame4.gif")
        observation_sample = False

    # Initialize environments
    envs.async_reset()
        
    # next_done = torch.zeros(batch_size).to(device)
    # Start training
    global_step = 0
    start_time = time.time()
    num_updates = args.updates_per_env * len(tasks)

    # Initialize reward normalization variables
    running_mean = 0.0
    running_variance = 0.0
    count = 1e-8  # Small initial value to prevent division by zero
    
    print(f"First task! #{current_task + 1}: {tasks[current_task]}")
    
    for update in range(0, num_updates):

        if (update % args.updates_per_env == 0) and update != 0:
            test_time = time.time()
            test(agent, test_envs, tasks, global_step, True, True)
            print(f"Tested! Time elaplesed {time.time() - test_time}")
            print()
            start_time = time.time()

            current_task += 1
            current_task %= len(tasks)

            envs = envs_cont[current_task]
            if args.ewc:
                ewc.update_task_weights(
                    task_id=tasks[current_task],
                    obs=torch.Tensor(next_obs).to(device)
                )

            envs.async_reset()
            # next_done = torch.zeros(batch_size).to(device)
            print(f"Next task! #{current_task + 1}: {tasks[current_task]}")

        if update % 20 == 0 and update % args.updates_per_env != 0:
            test_time = time.time()
            test(agent, test_envs, tasks, global_step)
            print(f"Tested! Time elaplesed {time.time() - test_time}")
            print()
            start_time = time.time()

        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        episode_rewards = np.zeros(args.num_envs)
        episode_lenghts = np.zeros(args.num_envs)

        episode_step = np.zeros(args.num_envs)

        # Rollout
        for step in range(args.num_steps):
            global_step += batch_size

            # Receive state from environments
            next_obs, reward, term, trunc, info = envs.recv()
                
            env_ids = info["env_id"]

            # Store current observation
            obs[step][env_ids] = torch.Tensor(next_obs).to(device)

            # Get actions
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs[step][env_ids])
                values[step][env_ids] = value.flatten()

            actions[step][env_ids] = action
            logprobs[step][env_ids] = logprob

            # Store rewards and dones
            rewards[step][env_ids] = torch.tensor(reward).to(device)
            dones[step][env_ids] = torch.tensor(term | trunc).to(device)

            # Send actions to environments
            envs.send(action.cpu().numpy(), env_ids)
            episode_step[env_ids] += 1

        # Advantage computation
        with torch.no_grad():

            next_value = agent.get_value(obs[-1]).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = ~ dones[-1]
                        nextvalues = next_value
                    else:
                        nextnonterminal = ~ dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = ~ dones[-1]
                        next_return = next_value
                    else:
                        nextnonterminal = ~ dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy and value networks
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                if args.ewc:
                    ewc_loss = ewc.compute_ewc_loss()
                    loss += ewc_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.s_p:
                    shrink_perturb(agent)
                    
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Log training metrics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.ewc:
            writer.add_scalar("losses/ewc", ewc_loss.item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    test(agent, test_envs, tasks, global_step, True, True)
    '''
    # columns = [f"Task {i}" for i in range(results_matrix.shape[0])]
    results_matrix_20 = (results_matrix - 3.5) / 16.5
    results_matrix_20 = np.clip(results_matrix_20, 0, 1)
    results_matrix_tough = (results_matrix - 3.5) / 26.5
    results_matrix_tough = np.clip(results_matrix_tough, 0, 1)

    table = wandb.Table(data=results_matrix.tolist(), rows=tasks, columns=tasks)
    table_paper = wandb.Table(data=results_matrix_20.tolist(), rows=tasks, columns=tasks)
    table_tough = wandb.Table(data=results_matrix_tough.tolist(), rows=tasks, columns=tasks)

    log_matrix_with_heatmap(results_matrix, tasks, "Raw Results",
                            vmin=np.min(results_matrix), vmax=np.max(results_matrix))
    log_matrix_with_heatmap(results_matrix_20, tasks, "Paper Results")
    log_matrix_with_heatmap(results_matrix_tough, tasks, "Tough Results")

    # Calculate metrics
    average_accuracy_20 = np.mean(results_matrix_20[-1])  # Mean accuracy of final row
    average_incremental_accuracy_20 = np.mean([results_matrix_20[i, :i + 1].mean() for i in range(len(tasks))])
    forgetting_20 = np.max(results_matrix_20, axis=0) - results_matrix_20[-1]
    average_forgetting_20 = forgetting_20[:-1].mean()
    forward_transfer_20 = np.mean([results_matrix_20[i - 1, i] for i in range(1, len(tasks))])
    backward_transfer_20 = np.mean([results_matrix_20[-1, i] - results_matrix_20[i, i] for i in range(len(tasks) - 1)])

    average_accuracy_tough = np.mean(results_matrix_tough[-1])  # Mean accuracy of final row
    average_incremental_accuracy_tough = np.mean([results_matrix_tough[i, :i + 1].mean() for i in range(len(tasks))])
    forgetting_tough = np.max(results_matrix_tough, axis=0) - results_matrix_tough[-1]
    average_forgetting_tough = forgetting_tough[:-1].mean()
    forward_transfer_tough = np.mean([results_matrix_tough[i - 1, i] for i in range(1, len(tasks))])
    backward_transfer_tough = np.mean(
        [results_matrix_tough[-1, i] - results_matrix_tough[i, i] for i in range(len(tasks) - 1)])

    wandb.log({"Kills": table})
    wandb.log({"Result Matrix- Paper": table_paper})
    wandb.log({"Result Matrix - tough": table_tough})
    wandb.log({
        "Paper/Average Accuracy": average_accuracy_20,
        "Paper/Average Incremental Accuracy": average_incremental_accuracy_20,
        "Paper/Forgetting": forgetting_20,
        "Paper/Average Forgetting": average_forgetting_20,
        "Paper/Forward Transfer": forward_transfer_20,
        "Paper/Backward Transfer": backward_transfer_20,
        "Tough/Average Accuracy": average_accuracy_tough,
        "Tough/Average Incremental Accuracy": average_incremental_accuracy_tough,
        "Tough/Forgetting": forgetting_tough,
        "Tough/Average Forgetting": average_forgetting_tough,
        "Tough/Forward Transfer": forward_transfer_tough,
        "Tough/Backward Transfer": backward_transfer_tough,
    })
    '''
    envs.close()
    writer.close()
    for test_env in test_envs:
        test_env.close()
        
    os.makedirs("models", exist_ok=True)
    torch.save(agent, f"models/{args.exp_name}.pth")
