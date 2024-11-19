import time
import numpy as np
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# Constants
SERVER_URL = "http://localhost:8000/v1/chat/completions"  # Update to your LLM server endpoint
CHECKPOINT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

def get_theoretical_max_throughput(max_tokens=512, time_per_token=0.01):
    """Calculate the theoretical maximum throughput based on model capabilities."""
    tokens_per_second = max_tokens / time_per_token
    return tokens_per_second

def benchmark_tokens_per_sec(num_requests=100):
    """Benchmark the LLM API for tokens per second."""
    total_tokens_generated = 0
    start_time = time.time()

    for _ in range(num_requests):
        prompt = "What is the capital of Australia?"  # Example prompt
        response = requests.post(SERVER_URL, json={"messages": [{"role": "user", "content": prompt}]})

        if response.status_code == 200:
            try:
                output = response.json()
                # Adjust the parsing logic based on the actual response format
                if 'choices' in output and output['choices']:
                    generated_text = output['choices'][0]['message']['content']
                    total_tokens_generated += len(generated_text.split())  # Count tokens
                else:
                    print(f"Unexpected response format: {output}")
            except (KeyError, IndexError, ValueError) as e:
                print(f"Error parsing response: {e}")
                print(f"Response JSON: {response.json()}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response Text: {response.text}")

    end_time = time.time()
    total_time = end_time - start_time

    tokens_per_sec = total_tokens_generated / total_time if total_time > 0 else 0
    theoretical_max = get_theoretical_max_throughput()

    return tokens_per_sec, theoretical_max

def run_benchmarks():
    """Run the benchmark and print results."""
    tokens_per_sec, theoretical_max = benchmark_tokens_per_sec(num_requests=100)

    print(f"Tokens per second: {tokens_per_sec:.2f}")
    print(f"Theoretical maximum tokens per second: {theoretical_max:.2f}")
    print(f"Efficiency: {tokens_per_sec / theoretical_max * 100:.2f}%")

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.bar(['Actual Throughput', 'Theoretical Max'], [tokens_per_sec, theoretical_max], color=['blue', 'orange'])
    plt.ylabel('Tokens per second')
    plt.title('Tokens per Second Benchmarking')
    plt.ylim(0, max(theoretical_max, tokens_per_sec) * 1.1)  # Set y-limit to 10% above the max value
    plt.grid(axis='y')
    plt.savefig('llm_benchmark_results.png')
    plt.show()

if __name__ == "__main__":
    run_benchmarks()
