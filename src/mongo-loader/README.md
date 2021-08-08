# Mongo Loader

This is the component that will allow us to cache data directly from Mongo. It's not meant to be run on its own, but rather os part of a larger training program (as a `Dataset`).

## Dependencies

The dependencies for this module are as follows:

- pytorch
- pymongo
- pytest (only for tests)


## Setting up Mongo for testing

```bash
docker run
	--name ddl-mongo
	-v $STORAGE_PATH:/etc/mongo # If you wish to store on a folder on disk, the first must be an absolute path. Otherwise, can be volume name.
	-p 127.0.0.1:27017:27017 # Default Mongo Port
	-d # Run in background
	mongo
```
