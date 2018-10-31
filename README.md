## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[final]: writeup_files/test2_Step08_final.jpg
![final]

An implementation of advanced lane finding from the Udacity Self Driving Car Engineering Nano Degree.

## Key files in this repo

* `writeup.md`
    * Project reflection including a brief description of each stage in the pipeline, shortcomings, and possible improvements.
* `pipeline.py`
    * Entry point to the project, call with python 3.6 or greater, no arguments
    * `$ python3 pipeline.py`
* `helpers.py`
    * Imported by `pipeline.py` no need to run directly
    * Contains `PipelineDebug` class used to easily save images at different stages in the pipeline
* `lanes.py`
    * Imported by `pipeline.py` no need to run directly
    * Contains the `LaneFinder` class which holds all the state necessary for each lane as it is searched for in the image.
* `calibrate_camera.py`
    * Generates camera calibration constants from calibration targets
    * Saves them in `camera.cal` as a python pickled dictionary object
