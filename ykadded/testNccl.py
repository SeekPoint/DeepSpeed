# https://zhuanlan.zhihu.com/p/675360966
# DeepSpeed ZeRO理论与VLM大模型训练实践
import os
import torch.distributed as dist
import argparse
import torch

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
device = torch.device("cuda", int(os.environ['LOCAL_RANK']))

dist.init_process_group("nccl")
dist.all_reduce(torch.ones(1).to(device), op=dist.ReduceOp.SUM)