import http

import click
import flask
import filelock
import itertools
import torch
import io
import uuid
import os

from multiprocessing import Process, active_children

from bao import BAO
from dao import MongoDAO
from pymongo import MongoClient

app = flask.Flask(__name__)

@app.route("/fetch", methods=["POST"])
def fetch():
    """
    Responds to the fetch endpoint and spins a subprocess to cache the given
    indices
    """
    request_json = flask.request.get_json()
    sample_ids = request_json.get("requested_samples")
    session_id = request_json.get("session_id")
    if sample_ids is None or session_id is None:
        return "", http.HTTPStatus.BAD_REQUEST

    p = Process(target=_prepopulate_samples, args=(sample_ids, uuid.UUID(session_id)))
    p.start()

    # multiprocessing.active_children() is not called here because it returns a
    # list of active children, but rather for its side effect of joining any
    # finished processes.
    # https://docs.python.org/2/library/multiprocessing.html#miscellaneous
    active_children()
    return "", http.HTTPStatus.OK

@app.route("/clear", methods=["POST"])
def clear():
    """
    Clears the cache. If given a session ID, this will clear all entries for
    that session. If given no session ID, this will clear the entire cache.
    """
    request_json = flask.request.get_json()
    session_id = uuid.uuid4()
    clear_all = True
    if request_json is not None:
        id_that_might_exist = request_json.get("session_id")
        if id_that_might_exist is not None:
            session_id = uuid.UUID(id_that_might_exist)
            clear_all = False
    dao = MongoDAO(MongoClient(), 'cache', session_id)
    if clear_all:
        dao.clear_cache()
    else:
        dao.clear_session()

    return "", http.HTTPStatus.OK


def _prepopulate_samples(sample_ids, session_id):
    """
    Loads the samples of the given indices into the cache from GCP bucket
    Args:
        sample_ids: A list of indices of samples to cache
    Flags:
        BATCH_BUCKET: If true, samples requested from the bucket will be fetched
            in a batched request rather than one-by-one
        BATCH_MONGO: If true, samples will be inserted into the cache in a
            batched request rather than one-by-one
        BATCH_PARALLEL: If true, samples will be fetched and cached one-by-one,
            removing the bottleneck of waiting for all samples to be fetched
            before any are cached. If this flag is set, the other two don't do
            anything.
    """
    BATCH_BUCKET = True
    BATCH_MONGO = True
    BATCH_PARALLEL = False

    dao = MongoDAO(MongoClient(), 'cache', session_id)

    # Trim samples based on what is cached

    cached_samples = dao.check_for_indices(sample_ids)
    sample_ids = [id for id in sample_ids if id not in cached_samples]

    # Cache from bucket
    if not len(sample_ids):
        return  # If everything is already cached, we're done

    bao = BAO('modeling-distributed-trainig-mnist')

    if BATCH_PARALLEL:
        for id in sample_ids:
            dao.cache_item(id, bao.get_item(id))

    else:
        samples = bao.get_items(sample_ids, batch=BATCH_BUCKET)

        dict = {}
        for i in range(len(samples)):
            dict[sample_ids[i]] = samples[i]

        dao.cache_items(dict, batch=BATCH_MONGO)

@click.command()
def main():
    app.run()

if __name__ == "__main__":
    main()
