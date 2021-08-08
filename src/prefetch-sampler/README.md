# Prefetch Sampler

This sampler is designed to go in tandem with the [prefetch server](../prefetch-server/README.md). When samples are requested, it tells the prefetch server about the next few samples, which will then begin to populate the cache.

### Dependencies:
	- requests
	- requests_unixsocket
	- torch

### Usage
See the docblock for more details, but in essence, this is used in tandom with a PyTorch sampler, which it will pull samples from. This sampler does not do any sampling itself, and will ONLY get samples from the underlying sampler.
