# IndoorPlants

(named for the wonderful coffe shop _Tommy_ in Montreal)

#### Note

This repo (including its documentation) is still under construction, but currently available functionality is detailed briefly below. Some example usage can be found in the *census_data_analysis* notebook.

#### Current functionality 

This *Python* repo provides some data analysis and model evaluation tools to be used with *Pandas* data and *scikit-learn* models. Built using *Python* 3.6, *Pandas* .2, *sklearn* .18 and *Numpy* 1.12

`analysis/exploratory.py` provides visualization capabilities for exploratory data analysis purposes.

`validation/crossvalidate.py` provides cross-validation functionality.

`validation/curves.py` provides curves to be used for ML model evaluation - validation, calibration, and precision-recall.

`validation/boundaries.py` provides tools for cross validating over different decision boundaries. 

#### Installation

After cloning the repo, enter *indoorplants/indoorplants/* and run `pip install -e .`
