# Prefetch Server

This server is designed to go in tandem with the [prefetch sampler](../prefetch-sampler/README.md). When the prefetch server recieves a request from the prefetch sampler, it attempts to cache the indices specified in the request. These samples are loaded from a GCP bucket and modified into a serializable form, then stored in the MongoDB cache.

## API
Prefetching samples into the cache:
```js
// POST /fetch
{
	"requested_samples": <list_of_indices>,
	"session_id": <uuid_session_id>
}
```

Clearing the cache (mostly for debugging):
```js
// GET /clear
```
