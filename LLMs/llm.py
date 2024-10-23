import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from openai import OpenAI
import os
from LLMs.llm_utils import edge_list_to_text


class LLM:
    def __init__(self, 
                 task_prompt="",
                 model_name="OpenDFM/ChemDFM-13B-v1.0",
                 save_dir=".LLM_Benchmarks"):

        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model_name = model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")  # Get the OpenAI API key from environment variable

        # Set up OpenAI if it's the model being used
        if "openai/gpt" in model_name:
            if self.openai_api_key is None:
                raise ValueError("OpenAI API key must be set as an environment variable.")
            self.client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        else:
            # Otherwise, load a local model (e.g., ChemDFM, Llama, Galactica)
            if "Chem" in model_name:
                self.tokenizer = LlamaTokenizer.from_pretrained(model_name, device_map="auto")
                self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_dir, device_map="auto")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_dir, device_map="auto")

            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
            # Initialize the pipeline with the loaded model and tokenizer
            self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device="cuda" if "ama" in model_name else "cpu")

        self.task_prompt = task_prompt
        self.extra_prompt_text = "Return a number, with no other text or filler. Do not explain your working. Answer in the format THE ANSWER IS X."

    def generate_text(self, user_prompt, system_prompt="You are a helpful assistant."):
        # Check if using OpenAI GPT API
        if "openai/gpt" in self.model_name:
            response = self.client.chat.completions.create(# Corrected method
                model="gpt-4o-mini",  # You can change this to other OpenAI GPT models
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500)
            completion = response.choices[0].message.content
            return completion
        else:
            # Use local model
            completion = self.pipe(user_prompt, max_new_tokens=500, return_full_text=False)
            return completion[0]["generated_text"]

    def forward(self, graph):
        edge_list = graph.edge_index
        prompt = self.produce_prompt(edge_list)
        pred = self.generate_text(prompt)
        return pred

    def produce_prompt(self, edge_list):
        graph_string = edge_list_to_text(edge_list, condense=True)
        prompt = f"question: {self.task_prompt} {self.extra_prompt_text} graph: {graph_string} answer:"
        if "gala" in self.model_name:
            prompt = f"question: {self.task_prompt} {self.extra_prompt_text} graph: {graph_string} answer: [START_REF]"
        return prompt


if __name__ == "__main__":
    # Example usage
    from torch_geometric.data import Data
    from torch_geometric.utils.random import erdos_renyi_graph
    from torch_geometric.utils import to_undirected

    # Initialize the LLM object with task-specific prompt and model_name set to OpenAI GPT
    llm = LLM("How many edges are in this graph?", save_dir=".LLM_Benchmarks", model_name="openai/gpt-3.5-turbo")

    # Create graph data
    edges = erdos_renyi_graph(30, 0.1)
    edges = to_undirected(edges)
    data = Data(edge_index=edges)
    print(data.num_edges / 2)
    pred = llm.forward(data)
    print(pred)
