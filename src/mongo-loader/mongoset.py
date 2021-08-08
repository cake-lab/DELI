import torch.utils.data
import pymongo, pymongo.client_session
import uuid
import io


class MongoCache(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        client: pymongo.MongoClient,
        database_name: str,
        session_id: uuid.UUID,
        cache_on_miss: bool = True,
    ):
        """
        Create a new MongoCache with the given dataset

        Args:
            base_dataset: The dataset to pull from.
                          NOTE: This dataset _MUST_ support __getitem__.
            client: The mongo client that will be used to connect to the cache
            database_name: The database to use within the mongo database
            session_id: The session_id to use in Mongo
            cache_on_miss: Whether or not the dataset should store in the cache upon a cache miss
        """
        self.base_dataset = base_dataset
        self.mongo_client = client
        self.session_id = session_id
        self.mongo_database = client.get_database(database_name)
        self.cache_on_miss = cache_on_miss

    def __len__(self) -> int:
        # Since we are caching the base_dataset, our length is the same as our base
        return len(self.base_dataset)

    def __getitem__(self, index):
        """
        Get an item from the underlying dataset if it is not in the mongo cache; otherwise, get it from the mongo cache
        """
        # causal_consistency: https://docs.mongodb.com/manual/core/read-isolation-consistency-recency/#causal-consistency
        with self.mongo_client.start_session(causal_consistency=True):
            return self._get_or_cache(index)

    def _get_or_cache(self, index):
        """
        Get an item from the cache using the given Mongo session
        """
        cached_item = self.mongo_database.cache.find_one(
            {"session_id": self.session_id, "sample_identifier": index}
        )

        if cached_item:
            cached_data = io.BytesIO(cached_item["sample"])
            return torch.load(cached_data)

        raw_item = self.base_dataset[index]
        if self.cache_on_miss:
            self._cache_item(index, raw_item)

        return raw_item

    def _cache_item(self, index, data):
        # Unfortunately, we can't put tensors directly in mongo so we must
        # pickle them...
        data_to_store = io.BytesIO()
        torch.save(data, data_to_store)
        data_to_store.seek(0)

        self.mongo_database.cache.insert_one(
            {
                "session_id": self.session_id,
                "sample_identifier": index,
                "sample": data_to_store.read(),
            }
        )
