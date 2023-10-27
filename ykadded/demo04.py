
import torch
import deepspeed
world_size = int(os.getenv('WORLD_SIZE', '4'))
generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_with_kernel_inject=True)

string = generator("DeepSpeed is", do_sample=True, max_length=50, num_return_sequences=4)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)

ds_results = measure_pipeline_latency(generator, "Hello, I'm a language model,", 50, 4)
print(f"DS model: {ds_results[0]}")