
import os
from time import perf_counter
import numpy as np
from transformers import pipeline, set_seed


def measure_pipeline_latency(generator, prompt, max_length, num_return_sequences):
    latencies = []
    # warm up
    for _ in range(2):
        output = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    # Timed run
    for _ in range(10):
        start_time = perf_counter()
        output = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    return f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};", time_p95_ms


model_id = "philschmid/gpt-j-6B-fp16-sharded"
local_rank = int(os.getenv('LOCAL_RANK', '0'))
generator = pipeline('text-generation', model=model_id, device=local_rank)
set_seed(42)
output = generator("Hello, I'm a language model,", max_length=50, num_return_sequences=4)
print(output)

vanilla_results = measure_pipeline_latency(generator, "Hello, I'm a language model,", 50, 4)
print(f"Vanilla model: {vanilla_results[0]}")