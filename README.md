![Reconstruction](https://github.com/hm-group/reconstruction/actions/workflows/main.yaml/badge.svg)

# Reconstruction
3D reconstruction from RGBD data

## Usage
First clone the project as follows,
```
git clone <url> <newprojname>
cd <newprojname>
```
Then build the project by using the following command, (assuming build is already installed in your virtual environment, if not then activate your virtual environment and use `conda install build`)
```
make build
```
Next, install the build wheel file as follows,
```
pip install <path to wheel file>
```

## Requirements
I used Anaconda with python3,

```
make install
conda activate cv3d-env
```