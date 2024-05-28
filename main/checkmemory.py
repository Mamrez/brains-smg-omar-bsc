import torch
import gc
import os


print(torch.cuda.current_device())

gc.collect()
torch.cuda.empty_cache()

print(torch.cuda.memory_summary(device=None, abbreviated=False))
# Get total, reserved, and allocated memory info
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r - a  # Free memory inside reserved
print(f"Total GPU memory: {t / (1024 ** 2):.2f} MB")
print(f"Free GPU memory: {r / (1024 ** 2):.2f} MB")
