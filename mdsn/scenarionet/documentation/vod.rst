#############################
View-of-Delft (VoD) 
#############################

| Website: https://intelligent-vehicles.org/datasets/view-of-delft/ 
| Download: https://intelligent-vehicles.org/datasets/view-of-delft/ (Registration required)
| Papers: 
    Detection dataset: https://ieeexplore.ieee.org/document/9699098
    Prediction dataset: https://ieeexplore.ieee.org/document/10493110

The View-of-Delft (VoD) dataset is a novel automotive dataset recorded in Delft,
the Netherlands. It contains 8600+ frames of synchronized and calibrated
64-layer LiDAR-, (stereo) camera-, and 3+1D  (range, azimuth, elevation, +
Doppler) radar-data acquired in complex, urban traffic. It consists of 123100+
3D bounding box annotations of both moving and static objects, including 26500+
pedestrian, 10800 cyclist and 26900+ car labels. It additionally contains
semantic map annotations and accurate ego-vehicle localization data.

Benchmarks for detection and prediction tasks are released for the dataset. See
the sections below for details on these benchmarks.

**Detection**: 
    An object detection benchmark is available for researchers to develop and
    evaluate their models on the VoD dataset. At the time of publication, this
    benchmark was the largest automotive multi-class object detection dataset
    containing 3+1D radar data, and the only dataset containing high-end (64-layer)
    LiDAR and (any kind of) radar data at the same time.

**Prediction**: 
    A trajectory prediction benchmark is publicly available to enable research
    on urban multi-class trajectory prediction. This benchmark contains challenging
    prediction cases in the historic city center of Delft with a high proportion of
    Vulnerable Road Users (VRUs), such as pedestrians and cyclists. Semantic map
    annotations for road elements such as lanes, sidewalks, and crosswalks are
    provided as context for prediction models.

1. Install VoD Prediction Toolkit
=================================

We will use the VoD Prediction toolkit to convert the data.
First of all, we have to install the ``vod-devkit``.

.. code-block:: bash

    # install from github (Recommend)
    git clone git@github.com:tudelft-iv/view-of-delft-prediction-devkit.git 
    cd vod-devkit
    pip install -e .

    # or install from PyPI
    pip install vod-devkit

By installing from github, you can access examples and source code the toolkit.
The examples are useful to verify whether the installation and dataset setup is correct or not.


2. Download VoD Data
==============================

The official instruction is available at https://intelligent-vehicles.org/datasets/view-of-delft/.
Here we provide a simplified installation procedure.

First of all, please fill in the access form on vod website: https://intelligent-vehicles.org/datasets/view-of-delft/.
The maintainers will send the data link to your email. Download and unzip the file named ``view_of_delft_prediction_PUBLIC.zip``.

Secondly, all files should be organized to the following structure::

    /vod/data/path/
    ├── maps/
    |   └──expansion/
    ├── v1.0-trainval/
    |   ├──attribute.json
    |   ├──calibrated_sensor.json
    |   ├──map.json
    |   ├──log.json
    |   ├──ego_pose.json
    |   └──...
    └── v1.0-test/

**Note**: The sensor data is currently not available in the Prediction dataset, but will be released in the near future.  

The ``/vod/data/path`` should be ``/data/sets/vod`` by default according to the official instructions,
allowing the ``vod-devkit`` to find it.
But you can still place it to any other places and:

- build a soft link connect your data folder and ``/data/sets/vod``
- or specify the ``dataroot`` when calling vod APIs and our convertors.


After this step, the examples in ``vod-devkit`` is supposed to work well.
Please try ``view-of-delft-prediction-devkit/tutorials/vod_tutorial.ipynb`` and see if the demo can successfully run.

3. Build VoD Database
===========================

After setup the raw data, convertors in ScenarioNet can read the raw data, convert scenario format and build the database.
Here we take converting raw data in ``v1.0-trainval`` as an example::

    python -m scenarionet.convert_vod -d /path/to/your/database --split v1.0-trainval --dataroot /vod/data/path

The ``split`` is to determine which split to convert. ``dataroot`` is set to ``/data/sets/vod`` by default,
but you need to specify it if your data is stored in any other directory.
Now all converted scenarios will be placed at ``/path/to/your/database`` and are ready to be used in your work.


Known Issues: VoD 
=======================

N/A
