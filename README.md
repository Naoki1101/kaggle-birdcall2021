# kaggle-birdcall2021

44th place solution to [BirdCLEF 2021 - Birdcall Identification](https://www.kaggle.com/c/birdclef-2021)

Public LB: 15th (0.77)
Private LB: 44th (0.63) 🥈


## Docker
1. Building a new image
```
$ git clone https://github.com/Kaggle/docker-python.git
$ cd docker-python

sudo ./build --gpu
```

2. Running the image
```
$ docker run -itd --gpus all -p 8888:8888  --shm-size=32gb  -v $PWD:/working -w /working --rm --name kaggle-gpu  -h host  kaggle/python-gpu-build /bin/bash
```

3. Running container
```
$ docker exec -it kaggle-gpu /bin/bash
```

## Setting
1. Overwrite credentials
```
export KAGGLE_USERNAME={USERNAME}
export KAGGLE_KEY={KEY}
```

2. Run
```
/bin/bash setup.sh
```

## Model Training
1. Preprocess
```
cd bins
/bin/bash preprocess.sh
```

2. Training
```
/bin/bash exp.sh
```

