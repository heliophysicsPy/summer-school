---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# SpacePy Tutorial -- SWMF Dynamics

### Background
The Space Weather Modeling Framework (SWMF) is a powerful coupled-model approach for investigating the system dynamics of the magnteophere, ionosphere, ring current, and other regions of geospace. It couples several models together to create self-consistent simulations of geospace. Most commonly, this includes BATS-R-US, the Ridley Ionosphere Model (RIM), and one of several ring current models. The output from these simulations can be complicated to work with as they have different formats, different values, and different approaches to analysis. This tutorial demonstrates how to use the Spacepy Pybats module to explore and analyze SWMF output.

Specifically, we're going to explore the result set from [Welling et al., 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020SW002489), Figures 2 and 3. The SWMF was used to explore the geospace response to a hypothetical worst-case-scenario storm sudden commencement. A subset of outputs are provided in the tutorial data directory representing the two simulations explored in the paper: an extreme CME arrival with *northward* IMF, and an extreme CME arrival with *southward* IMF. 

For our purposes, let's assume that the dataset has just been downloaded from the supercomputing environment. We do not know if the results are valid and reasonable yet. We need to first confirm that the results are reasonable. Then we want to illustrate the magnetosphere's response to the CMEs, including the compression of the dayside magnetopause, the formation of Region 1 Field-Aligned Currents (R1 FACs), and the time dynamics of the magnetopause. This will require opening files from BATS-R-US and RIM, creating 2D figures, and performing field-line tracing to see the open-closed boundary.

Concepts explored in this tutorial include,

  - Opening SWMF output files using [spacepy.pybats](https://spacepy.github.io/pybats.html).
  - Using the features of [spacepy.datamodel](https://spacepy.github.io/datamodel.html) to explore unfamiliar data sets.
  - Various features of [spacepy.plot](https://spacepy.github.io/plot.html), including the `target` keyword argument syntax.
  - Classes and inheritance in Python.



### Setup

As is the case for the other Spacepy tutorials, we use a single directory containing all the data for this tutorial and also the `.spacepy` directory (normally in a user's home directory). We use an environment variable to [point SpacePy at this directory](https://spacepy.github.io/configuration.html) before importing SpacePy; although we set the variable in Python, it can also be set outside your Python environment. Most users need never worry about this.

```python
tutorial_data = 'spacepy_tutorial'  # All data for this summer school, will be used throughout
import os
os.environ['SPACEPY'] = tutorial_data  # Use .spacepy directory inside this directory
```

Next, import the pertinent modules:

```python
import matplotlib.pyplot as plt
import spacepy.plot as splot
from spacepy.pybats import bats, rim

# for convenient notebook display and pretty out-of-the-box plots...
%matplotlib inline
splot.style('default')
```

### Reviewing BATS-R-US Log Files


```python
log = bats.BatsLog(tutorial_data+'/log_north_e20150321-054500.log')
print(log)
```

```python

```
