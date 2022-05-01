# KPM (Kinetic Predictor Model Package)

A Python package for training and testing neural networks for the prediction of reaction activation energies. Implemented with a full command line interface (CLI) for ease of use.

## Installation

*KPM* requires a few packages to be installed:

* NumPy (ver. 1.20 or greater)
* Matplotlib (ver. 3.5 or greater)
* Pandas
* SciPy
* Seaborn
* Scikit-learn (ver. 0.21 or greater)
* OpenBabel (ver. 3.1 or greater)
* RDKit

While installation of these packages through `pip` is possible, it is not recommended since OpenBabel installation via this method is not simple and is prone to issues. Instead, the preferred method of installation is through Anaconda.

To prepare a new `conda` environment for running KPM, run the following command from the repo's root directory:

```bash
conda create --name KPM python=3.7 --file requirements.txt
```

This will create a new `conda` environment named *KPM* with all of the prerequisites installed. Using Python 3.7 is optional, but is known to work correctly. To install *KPM*, run the following:

```bash
cd ..
pip install --no-deps ./KineticPredictorModel
```

This will install *KPM* in the new conda environment's `site_packages` but ignore the pip-based dependencies in `setup.py`, instead using the conda-based dependencies just installed in the environment. 

To use *KPM*'s CLI, it is neccessary to copy the script at `KineticPredictorModel/bin/KPM` to a place on your system's `PATH`, or to add this directory to your `PATH`. The latter can be done by adding

```bash
export PATH=$PATH:/path/to/KineticPredictorModel/bin
```

to your `.bashrc` file (or equivalent).

### Installation through `pip` (not recommended)

If you wish to avoid using `conda`, installation through `pip` is still possible, but is unsupported. Directly installing through `setup.py` by running 

```bash
cd ..
pip install ./KineticPredictorModel
```

requires a separate installation of OpenBabel 3, and is likely to fail due to improper linkage of OpenBabel's Python package (installed as a `pip` dependency) to the already installed backend. It is therefore advisable to separately install the OpenBabel Python package through `pip` before attempting to run the above command to install *KPM*.

## Usage

*KPM* has a fully functional CLI for using all of its functions. This can be invoked by running

```bash
KPM -h
```

to display the help text for the main program. 

*KPM* has three sub-commands:

* train (Trains a new model for $E_{act}$ prediction and tests prediction quality)
* test (Tests a previously trained model on a specified dataset)
* predict (Predicts $E_{act}$ for a given reaction/set of reactions, using a previously trained model. Ideal for integration with other programs)

These can be invoked with

```bash
KPM {sub-command} [args]
```

with the arguments for each sub-command documented in that sub-command's help text, accessible through

```bash
KPM {sub-command} -h
```

Examples of running these sub-commands for their respective purposes are provided in the `examples` folder.
