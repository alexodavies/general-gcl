# import twitch egos dataset
from datasets import load_dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch

import asyncio
import aiohttp
from tqdm import tqdm

from openai_api_example import call_llm_async
from sklearn.metrics import accuracy_score

import asyncio

from llm_representations import edge_list_to_text_with_multivariate_features
import numpy as np


print("Loading the Twitch Ego dataset from Hugging Face...")
# Load the Twitch Ego dataset from Hugging Face
dataset_hf = load_dataset("graphs-datasets/twitch_egos")
print("Dataset loaded successfully.")

verbose = False

N_shots = 10
batch_size = 2
N_train = N_shots
N_val = batch_size*25


split_names = list(dataset_hf.keys())
if len(split_names) == 1:
    split_name = split_names[0]
else:
    split_name = split_names[0]

print(f"Dataset split: {split_name}")

dataloader = DataLoader(dataset_hf[split_name], batch_size=batch_size, shuffle=True)
for batch in dataloader:
    break
graph_keys = list(batch.keys())
print(f"Graph keys: {graph_keys}")


# Convert it to PyTorch Geometric Data format
dataset_pg_list_train = []
dataset_pg_list_val = []

for idx, graph in enumerate(dataset_hf[split_name]):
    if idx < N_train:
        dataset_pg_list_train.append(Data(edge_index=graph['edge_index'], num_nodes=graph['num_nodes'], y=graph['y']))
    elif idx >= N_train and idx < N_train + N_val:
        dataset_pg_list_val.append(Data(edge_index=graph['edge_index'], num_nodes=graph['num_nodes'], y=graph['y']))
    else:
        break
    
print(f"Number of training graphs: {len(dataset_pg_list_train)}")
print(f"Number of validation graphs: {len(dataset_pg_list_val)}")

dataloader_pyg_train = DataLoader(dataset_pg_list_train, batch_size=N_train)
dataloader_pyg_val = DataLoader(dataset_pg_list_val, batch_size=batch_size)

classes = np.array(dataset_hf[split_name]['y']).flatten()

class_counts = np.bincount(classes)
num_classes = len(class_counts)
print(f"Number of classes: {num_classes}")
print(f"Class counts: {class_counts}")
print(f"Class distribution: {class_counts / len(classes)}")
for batch in dataloader_pyg_train:
    batch_edge_list = batch['edge_index']
    batch_num_nodes = batch['num_nodes']
    batch_y = torch.Tensor(batch['y']).flatten()
    break
graph_keys = list(batch.keys())
print(graph_keys)

task_label_descriptions = '''
    "0": "The graph represents a Twitch user who primarily plays a single game. These users typically have denser, more interconnected social networks.",
    "1": "The graph represents a Twitch user who plays multiple games. These users tend to have less densely connected social networks."
    '''


N_shot_graph_prompt = ''
for batch in dataloader_pyg_train:
    batch_edge_list = batch['edge_index']
    batch_num_nodes = batch['num_nodes']
    batch_y = torch.Tensor(batch['y']).flatten()
    for instance_idx in range(N_shots):
        edge_list = batch_edge_list[instance_idx]
        edge_str = edge_list_to_text_with_multivariate_features(
            edge_list, node_features=[], key_list=[]
        )
        graph_label = batch_y[instance_idx].item()  # Get the true label

        # Prepare the prompts
        N_shot_graph_prompt += f'graph with the following edges: {edge_str}'
        N_shot_graph_prompt += f' The graph belongs to class {int(graph_label)}. \n'
        
        
print(N_shot_graph_prompt)
# Asynchronous function to process a single instance
async def process_instance(instance_idx, batch_edge_list, batch_y, sem, predicted_classes, true_classes, sleeptime = 0.5):
    async with sem:  # Semaphore to limit concurrency
        # Extract the edge list for the current instance
        edge_list = batch_edge_list[instance_idx]
        edge_str = edge_list_to_text_with_multivariate_features(
            edge_list, node_features=[], key_list=[]
        )
        graph_label = batch_y[instance_idx].item()  # Get the true label

        # Prepare the prompts
        graph_prompt = f'graph with the following edges: {edge_str}'
        task_description = (
            f'I am doing graph classification for twitch ego networks. '
            f' Here are some examples: {N_shot_graph_prompt}. '
            f' Here is a graph: {graph_prompt}. '
            f'The classes are: {task_label_descriptions}. '
            f'From this, tell me out of the two classes, which class does this graph belong to?'
        )
        system_prompt = 'You can think but output a || and then either 0 or 1 at the end of your response.'

        if verbose:
            print(f"\nPrompt for instance {instance_idx}:\n{task_description}")

        # Asynchronously call the LLM API
        llm_output = await call_llm_async(task_description, system_prompt=system_prompt, sleeptime = sleeptime)

        # Process the LLM output to extract the predicted class
        try:
            class_predicted = int(llm_output.strip().split('||')[-1].strip())
        except Exception as e:
            # Fallback parsing in case of unexpected LLM output format
            class_predicted = int([char for char in llm_output if char in ('0', '1')][-1])

        if verbose:
            print(f'LLM Output for instance {instance_idx}: {llm_output}')
            print(f'Predicted class: {class_predicted}')
            if class_predicted == int(graph_label):
                print('Correctly predicted')
            else:
                print('Incorrectly predicted')

        # Append the results to the shared lists
        predicted_classes.append(class_predicted)
        true_classes.append(graph_label)


import time
# Main function to process all batches asynchronously
async def main(rate_limit = 10, sleeptime = 0.5):
    predicted_classes = []
    true_classes = []
    sem = asyncio.Semaphore(rate_limit)  # Adjust based on your API's rate limit

    for batch in tqdm(dataloader_pyg_val):
        batch_edge_list = batch['edge_index']  # Assuming this is a list or tensor of edge lists
        batch_y = torch.Tensor(batch['y']).flatten()
        print(len(batch_y))
        tasks = []
        for instance_idx in range(len(batch_edge_list)):
            task = process_instance(
                instance_idx, batch_edge_list, batch_y, sem, predicted_classes, true_classes, sleeptime
            )
            tasks.append(task)

        # Await the completion of all tasks for the current batch
        await asyncio.gather(*tasks)

    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f'Accuracy: {accuracy}, Number of samples: {len(true_classes)}')
if __name__ == '__main__':
    asyncio.run(main(sleeptime = 5))