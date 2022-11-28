# Code for the submission "Prediction of Molecular Field Points using SE(3)-Transformer Model"


## Download the data:

```
wget https://drive.switch.ch/index.php/s/HKjnWZqU25mBc1k/download -O ./data/moleucle_data.zip
unzip ./data/moleucle_data.zip -d ./data/
```
## Installing and starting docker environment

```
docker build -t se3-transformer .
docker run --gpus all --privileged -it --runtime=nvidia --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --rm -v ${PWD}/ se3-transformer:latest
```

## Create conda environment and install missing dependencies:

```
conda create -n se3_transformer
conda activate se3_transformer
conda install zarr
```

## increase open file limit:
```
ulimit -Sn 30000
```

## Start learning
```
python -m runtime.training
```
