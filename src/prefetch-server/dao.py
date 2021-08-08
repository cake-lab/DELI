import torch.utils.data
import pymongo, pymongo.client_session
import uuid
import io
import numpy as np
from typing import Iterable, Set


class MongoDAO:
    """
    A Database Access Object to abstract away some mongo nonsense
    """

    def __init__(
        self,
        client: pymongo.MongoClient,
        database_name: str,
        session_id: uuid.UUID,
    ):
        """
        Create a new MongoCache with the given dataset
        Args:
            base_dataset: The dataset to pull from.
                          NOTE: This dataset _MUST_ support __getitem__.
            client: The mongo client that will be used to connect to the cache
            database_name: The database to use within the mongo database
            session_id: The session_id to use in Mongo
        """
        self.mongo_client = client
        self.session_id = session_id
        self.mongo_database = client.get_database(database_name)

    def get_cached_indices(self, start=None, end=None):
        """
        Fetches the indices of all data stored in the cache
        Args:
            start: The number of indices to exclude from the beginning of the
                list of indices
            end: The number of indices from which to subtract start to get the
                total number of indices fetched

        """
        params = {}
        indices = [
            y["sample_identifier"]
            for y in self.mongo_database.cache.find(
                params, {"_id": 0, "sample_identifier": 1}
            )[start:end]
        ]
        return np.unique(indices).tolist()

    def check_for_indices(self, indices: Iterable[int]) -> Set[int]:
        """
        Returns the subset of the given indices that correspond to samples that
        are currently cached
        Args:
            indices: The indices of samples to find in the cache
        """
        cursor = self.mongo_database.cache.find(
            {"session_id": self.session_id, "sample_identifier": {"$in": indices}},
            {"_id": 0, "sample_identifier": 1},
        )
        return {i["sample_identifier"] for i in cursor}

    def cache_item(self, index: int, data):
        """
        Cache the given data into Mongo
        Args:
            index: The sample index of the data to cache
            data: The data to cache
        """
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

    def cache_items(self, dict: dict = None, batch: bool = True):
        """
        Cache items from the given dictionary into mongo
        Args:
            dict: A dictionary where the keys are indices and values are the
                data to cache
            batch: A flag that indicates whether or not to consolidate the data
                insertion into one request
        """
        if not batch:
            for index, data in dict.items():
                self.cache_item(index, data)

        for index, data in dict.items():
            # Unfortunately, we can't put tensors directly in mongo so we must
            # pickle them...
            data_to_store = io.BytesIO()
            torch.save(data, data_to_store)
            data_to_store.seek(0)
            dict[index] = data_to_store

        self.mongo_database.cache.insert_many(
            [
                {
                    "session_id": self.session_id,
                    "sample_identifier": index,
                    "sample": data.read(),
                }
                for index, data in dict.items()
            ]
        )

    def clear_cache(self):
        """
        Empties the entire cache.
        """
        self.mongo_database.cache.delete_many({})

    def clear_session(self):
        """
        Empties the cache of all entries matching the given session id.
        """
        self.mongo_database.cache.delete_many({"session_id": self.session_id})
