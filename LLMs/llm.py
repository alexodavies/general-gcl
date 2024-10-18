import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from LLMs.llm_utils import edge_list_to_text
import os


# "meta-llama/Llama-3.2-3B-Instruct"
# "meta-llama/Llama-3.2-1B-Instruct"
# "OpenDFM/ChemDFM-13B-v1.0" # This requires the llama tokenizer (annoying)
# "facebook/galactica-1.3b"

class LLM:
    def __init__(self, 
                 task_prompt="",
                 model_name="OpenDFM/ChemDFM-13B-v1.0",
                 save_dir=".LLM_Benchmarks"):
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model_name = model_name
        # bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        if "Chem" in model_name:
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name, device_map="auto")
            self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto") #, quantization_config=bnb_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = save_dir, device_map="auto") #, quantization_config=bnb_config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = save_dir, device_map="auto")
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

    
        # Initialize the pipeline with the loaded model and tokenizer
        self.pipe = pipeline("text-generation", model=model_name, tokenizer = self.tokenizer, device = "cuda" if "ama" in model_name else "cpu")
        self.task_prompt = task_prompt
        self.extra_prompt_text = "Return a number, with no other text or filler. Do not explain your working. Answer in the format THE ANSWER IS X."

    def generate_text(self, user_prompt, system_prompt="You are a helpful assistant."):
        completion = self.pipe(user_prompt, max_new_tokens = 500, return_full_text = False)
        return completion[0]["generated_text"]

    def forward(self, graph):
        edge_list = graph.edge_index
        prompt = self.produce_prompt(edge_list)
        # try:
            # Chat models
        pred = self.generate_text(prompt)

        return pred

    def produce_prompt(self, edge_list):
        graph_string = edge_list_to_text(edge_list, condense = True)
        prompt = f"question: {self.task_prompt} {self.extra_prompt_text} graph: {graph_string} answer:"
        if "gala" in self.model_name:
            prompt = f"question: {self.task_prompt} {self.extra_prompt_text} graph: {graph_string} answer: [START_REF]"
        return prompt

if __name__ == "__main__":
    # Example usage
    from torch_geometric.data import Data
    from torch_geometric.utils.random import erdos_renyi_graph
    from torch_geometric.utils import to_undirected


    # Initialize the LLM object with task-specific prompt
    llm = LLM("How many edges are in this graph?", save_dir=".LLM_Benchmarks")

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
