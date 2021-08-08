import torch.utils.data
import collections
import itertools
import requests
import requests_unixsocket
import uuid
from typing import Sequence, Any

from torch.utils.data.sampler import Sampler


class _NoSamplesNeeded(Exception):
    pass


class _SamplerExhausted(Exception):
    pass


class PrefetchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        base_sampler: torch.utils.data.Sampler,
        num_to_fetch: int,
        session_id: uuid.UUID,
        server_address: str = "127.0.0.1",
        port: int = 5000,
        # This such a silly way to do this, but parsing the address is overkill for this purpose.
        is_unix_socket: bool = False,
        min_queue_size: int = 0,
    ):
        """
        Make a new PrefetchSampler
        Args:
            base_sampler: The sampler to request samples from,
            num_to_fetch: The number of samples to fetch per iteration. This may be equal to the batch size, but does not necessarily have to be.
            server_address: the address of the prefetch server
            port: The port of the prefetch server
            is_unix_socket: Whether or not to access the server over a unix socket. If specified, the port is not used.
            min_queue_size: The number of samples below which the sampler will initiate another prefetch. The default min_queue_size is half of the fetch size.
        """
        self.base_sampler = base_sampler
        self.num_to_fetch = num_to_fetch
        self._server_address = server_address
        self._port = port
        self._is_unix_socket = is_unix_socket
        self._session_id = session_id
        self._queue = collections.deque([])
        self._init_sampler_iterator()
        self._min_queue_size = min_queue_size

    def _init_sampler_iterator(self):
        self._base_sampler_iterator = iter(self.base_sampler)

    def _get_samples_to_queue(self) -> Sequence[Any]:
        """
        Get the next samples from the base sampler
        """
        # islice will handle overrequesting samples
        samples = list(itertools.islice(self._base_sampler_iterator, self.num_to_fetch))
        # If there are no samples left, we're done.
        if not samples:
            raise _NoSamplesNeeded

        return samples

    def _queue_samples(self):
        """
        Queue the next samples from the base sampler, requesting the needed samples from the prefetch server as needed.
        """
        try:
            samples = self._get_samples_to_queue()
        except _NoSamplesNeeded:
            if self._queue:
                # If there's items in the queue, but no samples, then we should exhaust those first
                return
            else:
                # If there's nothing in the queue, and we don't need any samples, then we're done
                raise _SamplerExhausted

        self._queue.extend(samples)
        self._request_prefetch(samples)

    def _request_prefetch(self, samples_to_request):
        requests_obj = requests
        base_uri = f"http://{self._server_address}:{self._port}"
        if self._is_unix_socket:
            requests_obj = requests_unixsocket.Session()
            base_uri = f"http+unix://{self._server_address}"

        res = requests_obj.post(
            f"{base_uri}/fetch",
            json={
                "requested_samples": samples_to_request,
                "session_id": str(self._session_id),
            },
        )
        res.raise_for_status()

    def _queue_below_min(self) -> bool:
        """
        Returns True if the number of samples queued is below the minimum queue
        """
        return len(self._queue) <= self._min_queue_size

    def __iter__(self):
        self._init_sampler_iterator()
        while True:
            if self._queue_below_min():
                try:
                    self._queue_samples()
                except _SamplerExhausted:
                    return

            # Pop whatever's in the queue
            yield self._queue.popleft()
