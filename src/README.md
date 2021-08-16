# Source Code

This contains the source code of the DELI prototype. Each directory includes individual components and documentation on their dependencies. The `playbooks` directory contains several benchmarking scripts, and the scripts used for benchmarking the experiments.


## Example of DELI's Basic Usage

What follows is a basic example illustrating how DELI can be used in an existing PyTorch workflow. Please see the paper for details on the meanings of terminology.

```py
from torch.utils.data import DataLoader, RandomSampler
from pymongo import MongoClient

mongo_client = MongoClient("localhost")
# Get a dataset that fetches from bucket
gcp_dataset = get_gcp_dataset("our_bucket")
# Create the cache dataset
data = MongoCache(
    cloud_data,
	mongo_client=mongo_client,
    database_name="cache",
	session_id=uuid.uuid4(),
	fetch_on_miss=False # Do not fetch on miss; allow the pre-fetcher to do that.
)

# Can be any user-provided sampler
sub_sampler = RandomSampler(mongo_dataset)
# Wrap sub_sampler to pre-fetch samples
sample_server_sampler = PrefetchSampler(
    sub_sampler,
	num_to_fetch=512
)

# The cache dataset can now be used with PyTorch,
# just as any other dataset would
loader = DataLoader(
    data,
	batch_size=1024,
    sampler=sample_server_sampler
)
```

