from bucketset import Bucket
from google.cloud import storage
import torchvision
from PIL import Image
import io
from typing import Iterable


class BAO:
    """
    Bucket Access Object: The goal of this class is to abstract away some
    details of accessing GCP buckets to help standardize code across classes
    """

    def __init__(self, bucket_name: str):
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
        self._bucketset = self.get_gcp_dataset(bucket_name)

    def get_gcp_dataset(self, bucket_name: str):
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        return Bucket(
            bucket,
            "training",
            transform=torchvision.transforms.Compose(
                [
                    lambda data: Image.open(io.BytesIO(data)),
                    self.get_mnist_transforms(),
                ],
            ),
        )

    def get_mnist_transforms(self):
        """
        Wrapper function that returns a composition of PyTorch transforms to
            help maintain consistency for requesting samples from the GCP Bucket
        """
        # normalization_parameters = get_normalization_parameters()
        # This method takes a while to run and isn't relevant to the snippet of what we're experimenting with here
        normalization_parameters = (0.13066047770230949, 0.30524499715157616)

        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(*normalization_parameters),
                # there's probably a better way to do this, but this puts all tensors on CUDA.
                lambda tensor: tensor.cuda(),
            ]
        )

    def get_item(self, index: int):
        """
        Returns the sample of the given index from the GCP Bucket
        """
        return self._bucketset[index]

    def get_items(self, indices: Iterable[int], batch: bool = True):
        """
        Returns samples of the given indices from the GCP bucket. If batch is
        set to true, this will use the GCP dataset's method for returning
        multiple samples.

        Args:
            indices: the indices of the samples to return
            batch: true if samples should be collected in parallel, false
                otherwise
        """
        if not batch:
            return [self.get_item(index) for index in indices]
        return list(self._bucketset[indices])
