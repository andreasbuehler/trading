# Trading Machine Learning Project

## Installation
Make sure python version 3.11 is installed.

Open a terminal. Navgate to the project root.

**Create a virtual environment**
> python3 -m venv trading-env

**Activate the virtual environment**
> source trading-env/bin/activate

**Install requirements and package**
> python3 -m pip install -e .

## Code Structure

> data

Contains all data: interim (intermediate data, indicators) and processed (conditions).
Will be created automatically when running build_features.py

> src

Contains all the source code.

> src/data

Loading and dumping data.

> src/features

Creation of features from data.

> src/models

Actual ML models with training and prediction code.

> src/visualization

All visualizations/plots.

## How to run

How to actually run the code.

### Generate input features from raw data
Navigate to root directory.

> python3 src/features/build_features.py path/to/rawData.csv

This will take some time. See the terminal output for progress.
This should write interim and processed data to the data folder.

### Train models

> python3 src/models/train.py