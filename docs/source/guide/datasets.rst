.. _datasets:

Datasets
==========

Meddlr provides a standardized API for accessing datasets using `dataset_dicts`,
a sequence of dictionaries that contain the information necessary to generate examples
as well as corresponding metadata.

Meddlr also supports adding datasets in the standard PyTorch format.
Users can extend `torch.utils.data.Dataset` and return individual training
examples from the `__getitem__` method.


Basic Usage
------------
To use built in datasets, please set your dataset path in Meddlr (see :ref:`path`)

.. code-block:: python

    from meddlr.data import DatasetCatalog
    from meddlr.data import SliceDataset

    # List available datasets.
    DatasetCatalog().list()

    # Fetch a dataset. For example, the training split of the mridata 3D FSE Knee dataset.
    dataset_dicts = DatasetCatalog().get("mridata_knee_2019_train")

    # Add the dataset dicts into one of the pre-defined datasets offered by Meddlr.
    # If you want to add special functionality to your dataset, you can do so by
    # creating your own dataset that inherits from torch.utils.data.Dataset.
    dataset = SliceDataset(dataset_dicts)


Formatting Datasets
---------------------
Given the large size of certain datasets, some datasets need to be downloaded
and formatted manually. We provide utilities for doing this in the `datasets/`
folder of the Meddlr repository.

For example, to download and format the mridata 3D FSE Knee dataset, you can
run the following commands:

.. code-block:: bash

    cd datasets
    python format_mridata_org.py mridata_org_knee_dataset 



Adding New Datasets
--------------------
The most stable way to add datasets involves writing a function that returns
dataset dictionaries for your appropriate dataset. This function
will then have to be registered with the `DatasetCatalog`. Names 
for each dataset must be unique.

.. code-block:: python

    from meddlr.data import DatasetCatalog

    def get_dummy_dataset_dicts():
        return [{"id": 0}, {"id": 1}, {"id": 2}]
    
    DatasetCatalog.register("dummy_dataset", get_dummy_dataset_dicts)

As mentioned earlier, a simpler way to get started with your dataset
is to follow the PyTorch guide for `building your own dataset <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>_`.