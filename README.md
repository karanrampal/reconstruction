![Reconstruction](https://github.com/hm-group/reconstruction/actions/workflows/main.yml/badge.svg)

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

## Download data
To download a capture use the download data script, it requires the directory to download from gcp as follows,
```
gcloud auth login
./src/download_data.py -d <capture directory path relativeto bucket>
```

## Requirements
I used Anaconda with python3,

```
make install
conda activate cv3d-env
```