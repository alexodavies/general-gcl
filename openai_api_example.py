from openai import OpenAI, AsyncOpenAI
import os
import openai

def call_llm(user_prompt, MODEL="gpt-4o-mini", system_prompt = "You are a helpful assistant."):
    # Set the API key and model name
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    completion = client.chat.completions.create(
    model=MODEL,
    temperature=0.0,
    messages=[
        {"role": "system", "content": system_prompt}, # <-- This is the system message that provides context to the model
        {"role": "user", "content": user_prompt}  # <-- This is the user message for which the model will generate a response
    ]
    )

    llm_text_output = completion.choices[0].message.content
    return llm_text_output
import time
async def call_llm_async(user_prompt, MODEL="gpt-4o-mini", system_prompt="You are a helpful assistant.", sleeptime = 0.5):
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("Calling LLM API")
    completion = await client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},  # System message providing context to the model
            {"role": "user", "content": user_prompt}       # User message prompting the model's response
        ]
    )
    llm_text_output = completion.choices[0].message.content
    print(f"LLM API call complete, sleeping for {sleeptime} seconds")
    time.sleep(sleeptime)
    print("Waking up from sleep")
    
    return llm_text_output

if __name__ == "__main__":
    user_prompt = "What is the capital of France?"
    llm_text_output = call_llm(user_prompt)
    print(llm_text_output)
    # Output: "The capital of France is Paris."