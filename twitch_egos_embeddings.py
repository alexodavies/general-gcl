# import twitch egos dataset
from datasets import load_dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch

import asyncio
import aiohttp
from tqdm import tqdm

from openai_api_example import call_llm_async, call_embeddings_async
from sklearn.metrics import accuracy_score

import asyncio

from llm_representations import edge_list_to_text_with_multivariate_features
import numpy as np


print("Loading the Twitch Ego dataset from Hugging Face...")
# Load the Twitch Ego dataset from Hugging Face
dataset_hf = load_dataset("graphs-datasets/twitch_egos")
print("Dataset loaded successfully.")

verbose = False

MAX_TOKENS = 8192

N_shots = 0
batch_size = 128
N_train = 16000
N_val = 2000


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
dataset_pg_list_N_shot = []
dataset_pg_list_train = []
dataset_pg_list_val = []

for idx, graph in enumerate(dataset_hf[split_name]):
    if idx < N_shots:
        dataset_pg_list_N_shot.append(Data(edge_index=graph['edge_index'], num_nodes=graph['num_nodes'], y=graph['y']))
    elif idx >= N_shots and idx < N_train:
        dataset_pg_list_train.append(Data(edge_index=graph['edge_index'], num_nodes=graph['num_nodes'], y=graph['y']))
    elif idx >= N_train and idx < N_train + N_val:
        dataset_pg_list_val.append(Data(edge_index=graph['edge_index'], num_nodes=graph['num_nodes'], y=graph['y']))
    else:
        break
    
print(f"Number of training graphs: {len(dataset_pg_list_train)}")
print(f"Number of validation graphs: {len(dataset_pg_list_val)}")

if N_shots > 0:
    dataloader_pyg_N_shot = DataLoader(dataset_pg_list_N_shot, batch_size=N_shots)
dataloader_pyg_train = DataLoader(dataset_pg_list_train, batch_size=batch_size)
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
if N_shots > 0:
    for batch in dataloader_pyg_N_shot:
        batch_edge_list = batch['edge_index']
        batch_num_nodes = batch['num_nodes']
        batch_y = torch.Tensor(batch['y']).flatten()
        for instance_idx in range(N_shots):
            edge_list = batch_edge_list[instance_idx]
            edge_str = edge_list_to_text_with_multivariate_features(
                edge_list, node_features=[], key_list=[]
            )
            graph_label = batch_y[instance_idx].item()  # Get the true label
            print(len(edge_str))
            # Prepare the prompts
            prompt_length = len(N_shot_graph_prompt + f'graph with the following edges: {edge_str}' + f' The graph belongs to class {int(graph_label)}')//2
            print(f'current prompt length {prompt_length} \n')
            if prompt_length > MAX_TOKENS:
                print(f'max tokens reached at {instance_idx}')
                break
            N_shot_graph_prompt += f'graph with the following edges: {edge_str}'
            N_shot_graph_prompt += f' The graph belongs to class {int(graph_label)}. \n'
            
        
print(N_shot_graph_prompt)
# Asynchronous function to process a single instance
async def process_instance(instance_idx, batch_edge_list, batch_y, sem, embeddings, true_classes, sleeptime = 0.5):
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

        if len(task_description)//3 < MAX_TOKENS:
             # Asynchronously call the LLM API
            llm_output = await call_embeddings_async(task_description)

            embeddings.append(llm_output)
            true_classes.append(graph_label)
        else:
            print(f'Prompt too long for instance {instance_idx}')
            
       


import time
# Main function to process all batches asynchronously
async def main(loader, rate_limit = 10, sleeptime = 0.5, loader_string = ''):
    embeddings = []
    true_classes = []
    sem = asyncio.Semaphore(rate_limit)  # Adjust based on your API's rate limit

    for batch in tqdm(loader):
        batch_edge_list = batch['edge_index']  # Assuming this is a list or tensor of edge lists
        batch_y = torch.Tensor(batch['y']).flatten()
        print(len(batch_y))
        tasks = []
        for instance_idx in tqdm(range(len(batch_edge_list))):
            task = process_instance(
                instance_idx, batch_edge_list, batch_y, sem, embeddings, true_classes, sleeptime
            )
            tasks.append(task)

        # Await the completion of all tasks for the current batch
        await asyncio.gather(*tasks)
        sembeddings = np.array(embeddings)
        strue_classes = np.array(true_classes)
        print(sembeddings.shape)
        print(strue_classes.shape)
        np.save(f'embeddings_{loader_string}.npy', sembeddings)
        np.save(f'true_classes_{loader_string}.npy', strue_classes)

    
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
async def process_embeddings(loader, loader_string):
    try:
        embeddings = np.load(f'embeddings_{loader_string}.npy')
        true_classes = np.load(f'true_classes_{loader_string}.npy')
    except:
        # Run your async function to generate embeddings and true classes
        await main(loader, loader_string=loader_string)
        embeddings = np.load(f'embeddings_{loader_string}.npy')
        true_classes = np.load(f'true_classes_{loader_string}.npy')
    
    return embeddings, true_classes

async def main_task():
    # Train data
    loader = dataloader_pyg_train
    loader_string = f'train_{N_train}'
    train_embeddings, train_true_classes = await process_embeddings(loader, loader_string)

    # Train the model
    clf = LogisticRegression(random_state=0)
    # clf = MLPClassifier(random_state=0, max_iter=1000)
    
    clf.fit(train_embeddings, train_true_classes)
    train_pred_classes = clf.predict(train_embeddings)
    train_accuracy = accuracy_score(train_true_classes, train_pred_classes)
    print(f'Training accuracy: {train_accuracy}')

    # Validation data
    loader = dataloader_pyg_val
    loader_string = f'val_{N_val}'
    val_embeddings, val_true_classes = await process_embeddings(loader, loader_string)

    # Evaluate the model
    val_pred_classes = clf.predict(val_embeddings)
    val_accuracy = accuracy_score(val_true_classes, val_pred_classes)
    print(f'Validation accuracy: {val_accuracy}')

if __name__ == '__main__':
    # Run the entire task within a single asyncio event loop
    asyncio.run(main_task())
    