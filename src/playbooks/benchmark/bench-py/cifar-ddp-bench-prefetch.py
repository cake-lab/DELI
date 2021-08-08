#!/usr/bin/env python3

import torch, torch.nn, torch.utils.data, torch.optim
import torchvision, torchvision.datasets, torchvision.transforms, torch.distributed, torchvision.models.resnet
import io
import time
import dataclasses
from google.cloud import storage
from typing import List, Optional, Tuple, Callable
from PIL import Image
import contextlib
import click
import os
import pymongo
import mongoset
import uuid
import random

import loader as gcp_loader
import mongoset
import prefetch_sampler
import functools

DATA_PATH = "./data"
NUM_EPOCHS = 1
LEARNING_RATE = 0.18
MOMENTUM = 0.92


@dataclasses.dataclass
class EpochMeasurement:
    run_time: float
    data_loading_time: float
    num_items_observed: int
    accesses: int
    misses: int
    disk_time: float
    bucket_time: float
    cache_time: float
    prefetch_time: float


@dataclasses.dataclass
class StatsCollector:
    def __init__(self, mongo_cache: Optional[mongoset.MongoCache] = None) -> None:
        self.mongo_cache = mongo_cache
        self._bucket_time_taken = 0
        self._cache_time_taken = 0
        self._disk_time_taken = 0
        self._prefetch_time_taken = 0

    def get_miss_stats(self) -> Tuple[int, int]:
        if not self.mongo_cache:
            return (0, 0)

        return self.mongo_cache.get_miss_stats()

    def get_bucket_time(self) -> float:
        return self._bucket_time_taken

    def add_bucket_time(self, time: float) -> None:
        self._bucket_time_taken += time

    def get_cache_time(self) -> float:
        return self._cache_time_taken

    def add_cache_time(self, time: float) -> None:
        self._cache_time_taken += time

    def add_disk_time(self, time: float) -> None:
        self._disk_time_taken += time

    def get_disk_time(self) -> float:
        return self._disk_time_taken

    def add_prefetch_time(self, time: float) -> None:
        self._prefetch_time_taken += time

    def get_prefetch_time(self) -> float:
        return self._prefetch_time_taken

    def reset_stats(self) -> None:
        self._bucket_time_taken = 0
        self._cache_time_taken = 0
        self._disk_time_taken = 0
        self._prefetch_time_taken = 0
        if self.mongo_cache:
            self.mongo_cache.reset_miss_stats()


def get_transforms():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.486, 0.406], std=[0.299, 0.224, 0.225]
            ),
            # there's probably a better way to do this, but this puts all tensors on CUDA.
            lambda tensor: tensor.cuda(),
        ]
    )




def instrument_training(
    net,
    training_set,
    epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    momentum: float = MOMENTUM,
    set_epoch: Callable[[int], None] = lambda n: None,
    stats_collector: Optional[StatsCollector] = None,
) -> List[EpochMeasurement]:
    """
    Train the given network with the given training set.
    """

    def time_iteration_fetch(iterable):
        iterator = iter(iterable)
        while True:
            start_time = time.time()
            try:
                result = next(iterator)
            except StopIteration:
                return

            time_taken = time.time() - start_time
            yield result, time_taken

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
    epoch_measurements = []
    for i in range(epochs):
        set_epoch(i)
        total_loss = 0
        print(f"Epoch #{i + 1}")
        start_time = time.time()
        data_loading_time = 0
        num_items = 0
        for (inputs, labels), time_taken in time_iteration_fetch(training_set):
            print(inputs.shape)
            data_loading_time += time_taken
            num_items += 1
            if num_items % 1 == 0:
                print(stats_collector.get_miss_stats())
                print(num_items, f"Last measured time={time_taken*1000}ms")

            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            print("Loss:", loss)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        lr_scheduler.step(total_loss)

        epoch_time = time.time() - start_time
        misses, accesses = (0, 0)
        gcp_time, cache_time, disk_time, prefetch_time = (0, 0, 0, 0)
        if stats_collector:
            misses, accesses = stats_collector.get_miss_stats()
            gcp_time = stats_collector.get_bucket_time()
            cache_time = stats_collector.get_cache_time()
            disk_time = stats_collector.get_disk_time()
            prefetch_time = stats_collector.get_prefetch_time()
            stats_collector.reset_stats()

        measurement = EpochMeasurement(
            epoch_time, data_loading_time, num_items, accesses, misses, disk_time, gcp_time, cache_time, prefetch_time
        )
        epoch_measurements.append(measurement)
        print("CYCLE:", epoch_time, data_loading_time)

    return epoch_measurements


def get_gcp_dataset(bucket_name: str, stats_collector: StatsCollector):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    return gcp_loader.Bucket(
        bucket,
        "train",
        transform=torchvision.transforms.Compose(
            [lambda data: Image.open(io.BytesIO(data)), get_transforms()],
        ),
        track_time_taken=stats_collector.add_bucket_time,
    )


def test(net, loader: torch.utils.data.DataLoader) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


@contextlib.contextmanager
def process_group(rank, world_size):
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    yield
    torch.distributed.destroy_process_group()


def profiling_image_loader(track_time_taken: Callable[[float], None], path: str) -> Image.Image:
    def profile_read(old_read, *args, **kwargs) -> bytes:
        start_time = time.time()
        res = old_read(*args, **kwargs)
        track_time_taken(time.time() - start_time)

        return res

    with open(path, 'rb') as f:
        old_read = f.read
        f.read = functools.partial(profile_read, old_read)
        img = Image.open(f)
        return img.convert('RGB')

@click.command()
@click.option("--world_size", required=True, type=click.INT)
@click.option("--rank", required=True, type=click.INT)
@click.option("--addr", default="localhost")
@click.option("--port", default="29500")
@click.option("--seed", required=True, type=click.INT)
@click.option("--batch_size", default=512, type=click.INT)
@click.option("--fetch_size", default=1024, type=click.INT)
@click.option("--min-queue-size", default=0, type=click.INT)
def main(*, world_size, rank, addr, port, seed, batch_size, fetch_size, min_queue_size):
    print(f"I am rank {rank} talking to master {addr}... batch_size={batch_size} fetch_size={fetch_size} min_queue_size={min_queue_size}")
    torch.backends.cudnn.enabled = False
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    print("Opening folder...")
    stats_collector = StatsCollector()
    mongo_client = pymongo.MongoClient("localhost")
    cloud_data = get_gcp_dataset("modeling-distributed-trainig-cifar10", stats_collector)
    print("connecting to mongo")
    print("Making sampler")
    uncached_training_set, testing_set = torch.utils.data.random_split(
        cloud_data,
        [int(len(cloud_data) * 0.7), int(len(cloud_data) * 0.3)],
        generator=torch.Generator().manual_seed(seed),
    )
    session_id = uuid.uuid4()
    training_set = mongoset.MongoCache(
        uncached_training_set,
        mongo_client,
        "cache",
        session_id,
        cache_on_miss=False,
        track_time_taken=stats_collector.add_cache_time
    )
    stats_collector.mongo_cache = training_set

    with process_group(rank=rank, world_size=world_size):
        base_training_sampler, base_testing_sampler = (
            torch.utils.data.DistributedSampler(
                training_set, num_replicas=world_size, rank=rank, seed=seed,
            ),
            torch.utils.data.RandomSampler(
                testing_set
            )
        )
        sample_server_sampler = prefetch_sampler.PrefetchSampler(
            base_training_sampler,
            num_to_fetch=fetch_size,
            server_address="%2Ftmp%2Fprefetch-server.sock",
            is_unix_socket=True,
            session_id=session_id,
            min_queue_size=min_queue_size,
            track_time_taken=stats_collector.add_prefetch_time
        )
        print("Starting...")

        train_loader, _ = (
            torch.utils.data.DataLoader(
                training_set, sampler=sample_server_sampler, batch_size=batch_size
            ),
            torch.utils.data.DataLoader(
                testing_set, sampler=base_testing_sampler, batch_size=batch_size
            ),
        )
        print("Training...")
        network = torchvision.models.resnet.resnet50().cuda()
        ddp_net = torch.nn.parallel.DistributedDataParallel(network, device_ids=[0])
        training_times = instrument_training(
            ddp_net,
            train_loader,
            set_epoch=base_training_sampler.set_epoch,
            stats_collector=stats_collector
        )
        print("TRAINING TIMES")
        print("--------------")
        for training_time in training_times:
            print(f"Epoch run time: {training_time.run_time} s")
            print(f"Total data loading time: {training_time.data_loading_time} s")
            print(
                f"Time per batch: {(training_time.data_loading_time/training_time.num_items_observed)*1000} ms"
            )
            print(f"Time spent getting data from GCP: {training_time.bucket_time} s")
            print(f"Time spent talking to the cache: {training_time.cache_time} s")
            print(f"Time spent reading from disk: {training_time.disk_time} s")
            print(f"Time spent communicating with the prefetch server: {training_time.prefetch_time} s")
            print(
                f"Num Misses/Num Accesses/Miss Rate: {training_time.misses}/{training_time.accesses}/{training_time.misses/training_time.accesses if training_time.accesses > 0 else 'inf'}"
            )
            print("")
        print("Not testing accuracy")


if __name__ == "__main__":
    main()
