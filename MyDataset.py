
import tensorflow_datasets.public_api as tfds


class MyDataset(tfds.core.GeneratorBasedBuilder):
  """Short description of my dataset."""


VERSION = tfds.core.Version('0.1.0')


def _info(self):
    return tfds.core.DatasetInfo(
        # Specifies the tfds.core.DatasetInfo object
        builder=self,
        description=("This is the ML Dataset for predicting the variance "
                      "field of a Mixed, Turbulent Jet Engine Simulation"),
        features=tfds.features.FeaturesDict({
            "Smoothed field": tfds.features.Image(),
            "Variance field": tfds.features.Image(),
        }),
        supervised_keys=("Smoothed Field", "Variance Field"),
        homepage="https://github.com/bkowalski99/DNS_ML"
    )


def _split_generators(self, dl_manager):
    # Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    pass  # TODO


def _generate_examples(self):
    # Yields examples from the dataset
    yield 'key', {}

