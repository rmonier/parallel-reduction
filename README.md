# Parallel Reduction

## Prerequisites

- Install [Python 3.10](https://www.python.org/downloads/release/python-3109/)
- Install [Pipenv](https://pipenv.pypa.io/): `pip install pipenv`

## Installation

Install the dependencies by doing:
```sh
pipenv install --dev
```

Fill the Cloud configuration file with your credentials and parameters to be able to use the Cloud services:
```sh
cp .lithops_config.template .lithops_config
```

## Usage

Run the reduction script by doing:
```sh
pipenv run reduct
```

This will run the analysis of the execution time of the parallel reduction algorithm depending on the number of workers. It will save the results in the `metrics` folder and show a plot of the results for each dataset.

## Credits

Romain Monier [ [GitHub](https://github.com/rmonier) ] – Developer

## Contact

Project Link: https://github.com/rmonier/parallel-reduction
