from openai import OpenAI, AsyncOpenAI
import os
import openai
import asyncio

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
    # print(f"LLM API call complete, sleeping for {sleeptime} seconds")
    time.sleep(sleeptime)
    # print("Waking up from sleep")
    
    return llm_text_output


async def call_embeddings_async(user_prompt, MODEL="text-embedding-3-small", sleeptime=0.5):
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # print("Calling LLM API")
    completion = await client.embeddings.create(input=user_prompt, model=MODEL)
    embedding = completion.data[0].embedding
    time.sleep(sleeptime)

    return embedding

async def async_example(api_output, async_func, user_prompt):
    # Run the asynchronous function and append the result to api_output
    result = await async_func(user_prompt)
    api_output.append(result)
    
import time

async def main(user_prompt, n = 10):
    # Asynchronous embeddings call
    tasks = []
    api_output_emb = []
    t0 = time.time()
    

    # Create tasks for asynchronous embeddings calls
    for i in range(n):
        task = asyncio.create_task(async_example(api_output_emb, call_embeddings_async, user_prompt))
        tasks.append(task)

    # Await all tasks
    await asyncio.gather(*tasks)

    print(f"Time taken for {n} async calls: {time.time() - t0}")
    print(len(api_output_emb))  # print the length of the embedding


if __name__ == "__main__":
    user_prompt = "What is the capital of France?"
    
    # Synchronous LLM call 
    llm_text_output = call_llm(user_prompt)
    print(llm_text_output)
    
    # Asynchronous LLM call
    api_output_llm = []
    asyncio.run(async_example(api_output_llm, call_llm_async, user_prompt))
    print(api_output_llm[0])

    # Asynchronous embeddings call
    asyncio.run(main(user_prompt))
    