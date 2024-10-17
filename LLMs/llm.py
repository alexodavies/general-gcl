import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from llm_utils import edge_list_to_text
import os


# "meta-llama/Llama-3.2-3B-Instruct"
# "meta-llama/Llama-3.2-1B-Instruct"
# "OpenDFM/ChemDFM-13B-v1.0" # This requires the llama tokenizer (annoying)
# "facebook/galactica-1.3b"

class LLM:
    def __init__(self, 
                 task_prompt="",
                 model_name="meta-llama/Llama-3.2-3B-Instruct",
                 save_dir="/mnt/external_disk/models"):
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # if model_name != "meta-llama/Meta-Llama-3-8B-Instruct":
        # Load the model and tokenizer and save them to the external disk
        # try:
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = save_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = save_dir)
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        # except:
            # self.model = AutoModelForCausalLM.from_pretrained(model_name)
            # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # # Save the model and tokenizer to the specified directory
        # self.model.save_pretrained(save_dir)
        # self.tokenizer.save_pretrained(save_dir)
    
        # Initialize the pipeline with the loaded model and tokenizer
        self.pipe = pipeline("text-generation", model=model_name, tokenizer = self.tokenizer,  device = "cuda")
        # else:
        #     print("Using pipeline")
        #     self.pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", device = "cpu")
        self.task_prompt = task_prompt
        self.extra_prompt_text = "" # Return a number, with no other text or filler, and no linebreaks. Produce no other content."

    def generate_text(self, user_prompt, system_prompt="You are a helpful assistant."):
        print("generating text")
        completion = self.pipe(user_prompt, max_length = 50, return_full_text = False)
        return completion# [0]["generated_text"]

    def forward(self, graph):
        edge_list = graph.edge_index
        prompt = self.produce_prompt(edge_list)
        print(prompt)
        # try:
            # Chat models
        pred = self.generate_text(prompt)
        # except:
        #     # Galactica
        #     prompt = self.tokenizer(prompt + "[START REF]", return_tensors="pt").input_ids
        #     pred = self.model.generate(prompt, max_length=1000)
        #     print(pred)
        #     pred = self.tokenizer.decode(pred[0])
        #     # print(pred)
        return pred

    def produce_prompt(self, edge_list):
        graph_string = edge_list_to_text(edge_list)
        prompt = f"{self.task_prompt} {self.extra_prompt_text} {graph_string}"
        return prompt

if __name__ == "__main__":
    # Example usage
    from torch_geometric.data import Data
    from torch_geometric.utils.random import erdos_renyi_graph
    from torch_geometric.utils import to_undirected

    

    # # Example edge list tensor for graph
    # edge_index = torch.tensor([[0, 1, 1, 2],
    #                            [1, 0, 2, 1]], dtype=torch.long)


    # data = 

    # Initialize the LLM object with task-specific prompt
    llm = LLM("What is the density of this graph?", save_dir=".LLM_Benchmarks")

    # # Create graph data
    edges = erdos_renyi_graph(5, 0.1)
    edges = to_undirected(edges)
    data = Data(edge_index=edges)
    print(data.num_edges)
    pred = llm.forward(data)
    print(pred)


    # # Create graph data
    edges = erdos_renyi_graph(15, 0.1)
    edges = to_undirected(edges)
    data = Data(edge_index=edges)
    print(data.num_edges)
    # Forward pass to predict the answer based on graph input
    pred = llm.forward(data)
    print(pred)
