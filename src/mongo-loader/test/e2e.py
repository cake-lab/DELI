import unittest.mock
import uuid

import pytest
import torch.utils.data
import pymongo, pymongo.mongo_client, pymongo.database

import mongoset

# This should perhaps be more configurable than just a constant...
MONGO_ADDRESS = "localhost"


@pytest.fixture(scope="session")
def mongo_instance():
    return pymongo.MongoClient(MONGO_ADDRESS)


@pytest.fixture(scope="session")
def mongo_database(mongo_instance):
    return mongo_instance.database("cache_test")


@pytest.fixture
def session_id():
    return uuid.uuid4()


@pytest.fixture
def mock_dataset():
    def getitem(index):
        if not (0 <= index < 1000):
            raise KeyError

        return 2 ** index

    mock = unittest.mock.MagicMock()
    mock.__len__ = unittest.mock.MagicMock(return_value=1000)
    mock.__getitem__ = unittest.mock.MagicMock(side_effect=getitem)

    return mock


def test_gets_from_underlying_dataset_when_not_cached(
    mongo_instance: pymongo.mongo_client.MongoClient,
    mock_dataset: torch.utils.data.Dataset,
    session_id: uuid.UUID,
):
    mongo_cache = mongoset.MongoCache(
        mock_dataset, mongo_instance, "cache_test", session_id
    )

    mongo_cache[10]
    mock_dataset.__getitem__.assert_called_once()


def test_does_not_access_dataset_when_cached(
    mongo_instance: pymongo.mongo_client.MongoClient,
    mock_dataset: torch.utils.data.Dataset,
    session_id: uuid.UUID,
):
    mongo_cache = mongoset.MongoCache(
        mock_dataset, mongo_instance, "cache_test", session_id
    )

    mongo_cache[10]
    mongo_cache[10]
    # should _ONLY_ be called once, not twice
    mock_dataset.__getitem__.assert_called_once()
