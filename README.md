# morphVQ
An R, PyTorch, and MATLAB implementation of "Automated morphological phenotyping using learned shape descriptors and functional maps: A novel approach to geometric morphometrics"

Each folder named after the coding language contains related codes for data processing, modeling, post-processing, or performance evaluation.


## Contents
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Running Python Codes](#implementation)


## Dependencies
This implementation requires python >= 3.7 and requires the following dependencies before the full installation. The version numbers have been tested and shown to work, other versions are likely, but not guaranteed, to work.

- [Progressbar2](https://pypi.org/project/progressbar2/)
- [vectorheadt](https://github.com/rubenwiersma/hsn)
- [Suitesparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) (used for the Vector Heat Method)

Here we refer to [HSN](https://github.com/rubenwiersma/hsn) for the instruction on installing the above three dependencies


Clone this repository and its submodules
```
$ git clone --recurse-submodules https://github.com/rubenwiersma/hsn.git
```

Install the `vectorheat` python module, explained in the following subsection.

### The vectorheat extension

**[Update April 2021]** Nick Sharp has created his own bindings for the Geometry Central library, called [Potpourri3d](https://github.com/nmwsharp/potpourri3d). This library includes computations of the vector heat method on meshes **and point clouds**. You can install it with pip:
```
$ pip install potpourri3d
```
The transforms used in this repository make use of our own binding, which can be installed as follows:

To perform computations on the mesh - e.g. Vector Heat Method, surface area computation, etc. - we use [Geometry Central](https://geometry-central.net). We have created a small Python binding into a C++ function that computes these quantities globally for each vertex in a mesh. Run the following commands in your shell to install the extension:
```
$ pip install ./vectorheat
```

**Having trouble building?**
First make sure that you have the latest version of CMake installed (> 3.10). Next, check that every dependency is present in this repo (pybind11, geometrycentral). If not, you might not have cloned submodules. To fix this:
```
$ git submodule update --recursive
```

**Suitesparse?** When processing shapes with the Vector Heat Method, you might run into a solver error. This is fixed when you build the `vectorheat` extension with suitesparse. Installation in Linux with:

```
$ apt-get install libsuitesparse-dev

```


## Installation
After install the dependencies, use pip or conda to install dependencies based on the package list: "python/requirements.txt":

For pip installation:
```bash
pip3 install -r python/requirements.txt
```

For conda installation:
```bash
conda install --yes --file python/requirements.txt
```

## Implementation

Once the dependecies and the packages are installed, users can refer to "python/Cuboid Descriptor Learning.ipynb" for training and evaluating the morphVQ model.

