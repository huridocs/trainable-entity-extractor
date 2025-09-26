import math

import torch


def get_batch_size(samples_count: int):
    batch_size_by_samples = math.ceil(samples_count / 100)
    memory_available = torch.cuda.get_device_properties(0).total_memory / 1000000000
    limit_batch = max(int(memory_available / 5), 1)
    return min(limit_batch, batch_size_by_samples)


def get_max_steps(samples_count):
    if samples_count * 25 < 2000:
        return math.ceil(25 * samples_count / 200) * 200

    min_epochs = 3

    if samples_count < (2000 / min_epochs):
        return 2000

    return min(math.ceil(min_epochs * samples_count / 200) * 200, 6000)
