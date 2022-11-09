"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""
from torch import distributed as dist


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def synchronize():
    """Helper function to synchronize (barrier)
        among all processes when using distributed training"""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    current_world_size = dist.get_world_size()
    if current_world_size == 1:
        return
    dist.barrier()


def all_gather_object(obj):
    world_size = get_world_size()
    if world_size < 2:
        return [obj]
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, obj)
    return output


def is_available() -> bool:
    return dist.is_available()
