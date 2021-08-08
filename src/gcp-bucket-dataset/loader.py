import torchvision
from google.cloud import storage
import functools
import threading
import queue

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    TypeVar,
    Optional,
    Union,
    Iterable,
)

Transformed = TypeVar("Transformed")
TargetTransformed = TypeVar("TargetTransformed")
FetchedResult = Tuple[Any, Union[int, TargetTransformed]]


class Bucket(torchvision.datasets.VisionDataset):
    """
    Similar to torchvision.datasets.DatasetFolder but from a GCP Bucket.

    Unfortunately, DatasetFolder is tightly coupled to the filesystem, so we must do this by hand
    """

    def __init__(
        self,
        bucket: storage.Bucket,
        path_prefix: str = "",
        transform: Optional[Callable[[Any], Transformed]] = None,
        target_transform: Optional[Callable[[Any], TargetTransformed]] = None,
    ):
        """
        Args:
            bucket: The bucket to pull data from
            path_prefix: The prefix to use on any path operations. The first remaining component of the path _MUST_
                         be the class of the file, e.g. a/b/c means that file c has class a.
            transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        """
        super().__init__(bucket, transform=transform, target_transform=target_transform)
        self.bucket = bucket
        self.path_prefix = path_prefix
        self._blob_classes = self._classify_blobs()

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[storage.Blob, int]]:
        """
        Holds all of the blobs and their classes
        """
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self._blob_classes)
            for blob in self._blob_classes[blob_class]
        ]

    def _classify_blobs(self) -> Dict[str, List[storage.Blob]]:
        """
        Get the class for each blob in the bucket.

        Returns:
            A dict of the classes to the blobs
        """

        blob_classes: Dict[str, List[storage.Blob]] = {}

        for blob in self.bucket.list_blobs():
            blob_name = blob.name
            stripped_path = _remove_prefix(blob_name, self.path_prefix).lstrip("/")
            # Indicates that it did not match the starting prefix
            if stripped_path == blob_name:
                continue

            blob_class = stripped_path.split("/")[0]

            blobs_with_class = blob_classes.get(blob_class, [])
            blobs_with_class.append(blob)
            blob_classes[blob_class] = blobs_with_class

        return blob_classes

    def __getitem__(
        self, index: Union[int, Iterable[int]]
    ) -> Union[FetchedResult, Iterable[FetchedResult],]:
        if _is_iterable(index):
            return self._get_items_in_parallel(index)
        else:
            return self._get_item(index)

    def _get_item(self, index: int):
        """
        Get a single item from the GCP bucket
        """
        blob, target_class = self._classed_items[index]
        sample_data = blob.download_as_bytes()
        if self.transform:
            sample_data = self.transform(sample_data)
        if self.target_transform:
            target_class = self.target_transform(target_class)

        return sample_data, target_class

    def _get_items_in_parallel(self, indexes: Iterable[int]) -> Iterable[FetchedResult]:
        """
        Get several items in parallel from the bucket
        """
        in_queue = queue.Queue()

        def worker():
            index = in_queue.get()
            res = self._get_item(index)
            out_queue.put((index, res))
            in_queue.task_done()

        out_queue = queue.Queue()
        # Queue all of the items that we will process
        index_list = list(indexes)
        for item in index_list:
            in_queue.put(item)

        # Start one thread for every item in the queue
        for _ in index_list:
            threading.Thread(target=worker).start()

        # Wait for all items to be processed
        in_queue.join()

        res = {}
        while not out_queue.empty():
            index, data = out_queue.get_nowait()
            res[index] = data

        for index in indexes:
            yield res[index]

    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self._blob_classes.values())


def _remove_prefix(s: str, prefix: str) -> str:
    if not s.startswith(prefix):
        return s

    return s[len(prefix) :]


def _is_iterable(obj) -> bool:
    try:
        iter(obj)
        return True
    except TypeError:
        return False
