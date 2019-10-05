# IndoorPlants

[![Build Status](https://travis-ci.org/lermana/indoorplants.svg?branch=master)](https://travis-ci.org/lermana/indoorplants)

### Background

This library is named for the wonderful coffee shop _Tommy_ in Montreal where, as a good friend said: "the plants really tie the room together." It is my hope that this library will tie together `pandas`, `sklearn`, and `matplotlib` in ways that are useful for the practicing data scientist.

`indoorplants` provides data analysis and model validation tools, and the top-level namespace is broken out as such. Most of the code assumes that data is stored in `DataFrame` instances, and models are assumed to provide the `sklearn` interface. This code is built for *Python* 3.6+.

### High-Level Description of Selected Functionality

On the _analysis_ side of things:
- `analysis.exploratory` aids with exploratory data analysis
- `analysis.wrangle` is for data wrangling, including the detection of leaky features in classification problems
- `analysis.features` at the moment allows for calculating feature distances

On the _validation_ side of things:
- `validation.crossvalidate` provides the core cross-validation functions
- `validation.curves` offers curves for ML model evaluation, including: validation, learning, and calibration

Example usage can be found in the `census_data_analysis` notebook.

### Installation

You can run `pip install indoorplants` to get this repo.
