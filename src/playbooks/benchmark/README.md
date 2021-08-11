# Benchmark Playbooks

The `bench` directory should hold any and all files needed to run the benchmarks. To use these playbooks, you should copy in any dependencies necessary from the rest of the repository into that directory. It is worth noting that these scripts are **not** intended to fully set up the environment; they were originally written to aid execution of experiments in an already set-up environment. However, the setup needed is documented below; the files are preserved as they were during our experiment runs.

The scripts within the `bench` directory will assume that the pre-fetch server is present in the `prefetch-server` directory; this is a required dependency for the benchmarks. `bench.py` is where the actual benchmark should live. Some options for `bench.py` are included in the `bench-py` directory (these make use of our personal bucket IDs and may thus require minor modifications. They both depend on `prefetch-sampler/prefetch_sampler.py`, `mongo-loader/mongoset.py`, and `gcp-bucket-dataset/loader.py`). Use of our benchmarking scripts requires setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable, described [here](https://cloud.google.com/docs/authentication/getting-started). These are all driven using Ansible playbooks.


`copyrun.yml` assumes that there is a Docker container that exists with [a MongoDB setup](https://hub.docker.com/_/mongo) named `ddl-mongo` (see `mongo-loader/README.md`), but the playbook will attempt to setup the database.


The playbooks are controlled with several Ansible variables that are passed to the benchmark scripts. These are as follows

- `batch_size` -  The batch size to use in the training loop
- `fetch_size` - The number of samples to pre-fetch
- `cache_max_items` - The number of samples that can be held in the cache
- `min_queue_size` - The minimum queue size

Example invocation
```
$ ansible-playbook -i inventory.ini copyrun.yml -e "batch_size=512 min_queue_size=2048"
```

To run more than one benchmark at a time, you can use the `batch.sh` script, and either pass in a file (or pipe in input through stdin) in the format of Ansible's `--extra-args` (`-e`), e.g.

```
batch_size=512 fetch_size=1024 cache_max_items=2560
batch_size=512 fetch_size=1024
batch_size=512 fetch_size=2048
```


## Dependencies

On top of the dependencies normally required, these playbooks make use of `uwsgi`.
