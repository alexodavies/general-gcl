from transformers import pipeline


class LLM:
    def __init__(self, model_name="meta-llama/Llama-2-13b-hf"):
        self.pipe = pipeline("text-generation", model=model_name)

    def generate_text(self, user_prompt, system_prompt="You are a helpful assistant."):
        completion = self.pipe(
            user_prompt,
            max_length=50,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return completion[0]["generated_text"]