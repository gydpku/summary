import openai
import anthropic
import tiktoken
import json
import os
import pdb
from openai import OpenAI
#from vllm import LLM, SamplingParams
from datasets import load_dataset
# Replace this with your actual OpenAI API key
#sampling_params = SamplingParams(temperature=0.0,max_tokens=100, top_p=0.95)
#llm = LLM(model='/dccstor/obsidian_llm/yiduo/llama-3-instruct',gpu_memory_utilization=0.8)
encoding = tiktoken.encoding_for_model("gpt-4")


def truncate_text_with_token_count(text, max_tokens):
    num_tokens = len(encoding.encode(text))
    if num_tokens > max_tokens:
        encoded = encoding.encode(text)[:-(num_tokens - max_tokens)]
        truncated_text = encoding.decode(encoded)
        return truncated_text
    return text

def query_azure_openai_chatgpt_chat_2(query, temperature=0):
    truncated_input = truncate_text_with_token_count(query, 2048)
#    sampling_params = SamplingParams(temperature=0.0,max_tokens=100, top_p=0.95)
 #   llm = LLM(model='/dccstor/obsidian_llm/yiduo/llama-3-instruct',gpu_memory_utilization=0.8)
    prompt=truncated_input
    output = llm.generate(prompt, sampling_params)
    output=output[0].outputs[0].text
    return output
def query_azure_openai_chatgpt_chat_2(query, temperature=0):
    truncated_input = truncate_text_with_token_count(query, 30000)

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4000,
        temperature=temperature,
        messages=[{"role": "user", "content": truncated_input}]
    )
    return response.content[0].text
def query_azure_openai_chatgpt_chat(query, temperature=0):
    truncated_input = truncate_text_with_token_count(query, 30000)

    client = OpenAI(api_key=openai.api_key)

    response = client.chat.completions.create(model="gpt-4o", messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": truncated_input}, ], temperature=0.4, max_tokens=4000, )

    for chunk in response:
        if chunk[0]=='choices':
            for piece in chunk[1][0]:
                if piece[0]=='message':
                    for sub in piece[1]:
                        if sub[0]=='content':
                            output=sub[1]
        if chunk[0]=='usage':
            tokens=chunk[1]
    try:
        return output
    except:
        return ' '
def query_azure_openai_chatgpt_chat_3(query, temperature=0):
    query = truncate_text_with_token_count(query, 30000)
    '''    
    sampling_params = SamplingParams(temperature=0.0,max_tokens=100, top_p=0.95)
    llm = LLM(model='/dccstor/obsidian_llm/yiduo/llama-3-instruct',gpu_memory_utilization=0.8)
    prompt=truncated_input
    output = llm.generate(prompt, sampling_params)
    output=output[0].outputs[0].text
    '''
    client = OpenAI(api_key=openai.api_key)
    #response=#ChatCompletion(id='chatcmpl-9W1Wpkj9IAWgjsWmZ4BxZw3QtjI9V', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='```python\n{\n    "Domain": "online continual learning, catastrophic forgetting, class incremental learning",\n    "Problem": "The paper addresses the challenge of catastrophic forgetting in online continual learning, particularly in the class incremental learning setting, where a model learns from a sequence of tasks incrementally without forgetting previously acquired knowledge.",\n    "Assumption and Claim": {\n        "Holistic Representations": "Learning holistic representations of input data reduces catastrophic forgetting by preserving features that may be useful for future tasks.",\n        "Mutual Information Maximization": "Maximizing mutual information between input data and their representations helps in learning comprehensive features and preserving knowledge across tasks.",\n        "Feature Bias Reduction": "Reducing feature bias by learning non-discriminative features alongside discriminative ones helps in generalizing across different tasks and reduces forgetting."\n    },\n    "Method": {\n        "OCM Technique": "Introduces Online Continual Learning through Mutual Information Maximization (OCM) which incorporates mutual information maximization into the learning process to address catastrophic forgetting.",\n        "Feature Learning": "Employs a feature extractor and a classifier where mutual information between the input and its representation is maximized to ensure comprehensive feature learning.",\n        "Loss Functions": "Uses cross-entropy loss for classification and an InfoNCE loss as a proxy for mutual information maximization.",\n        "Data Augmentation": "Applies data augmentation techniques like local rotation to increase the diversity of training samples, aiding in better generalization and robustness.",\n        "Replay Mechanism": "Utilizes a replay buffer to store previous data, which is used along with new data for training to prevent forgetting."\n    }\n}\n```', role='assistant', function_call=None, tool_calls=None))], created=1717419399, model='gpt-4-turbo-2024-04-09', object='chat.completion', system_fingerprint='fp_31a6c0accf', usage=CompletionUsage(completion_tokens=336, prompt_tokens=25421, total_tokens=25757))

    response = client.chat.completions.create(model="gpt-4", messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": query}, ], temperature=temperature, max_tokens=4000, )
    #print(temperature)
    for chunk in response:
        if chunk[0]=='choices':
            for piece in chunk[1][0]:
                if piece[0]=='message':
                    for sub in piece[1]:
                        if sub[0]=='content':
                            output=sub[1]
    
    #pdb.set_trace()
    return output
