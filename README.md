# kaggle-birdcall2021

42th place solution to [BirdCLEF 2021 - Birdcall Identification](https://www.kaggle.com/c/birdclef-2021)

- Public LB: 15th (0.77)
- Private LB: 42th (0.63) 🥈

<br>
<br>

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

<br>
<br>

## Setting
1. Overwrite credentials
```
export KAGGLE_USERNAME={USERNAME}
export KAGGLE_KEY={KEY}
```

2. Setting configs/notify.yml and account.yml
- account.yml
    - https://neptune.ai/
```
neptune:
    token: {YOUR TOKEN}
    project: {PROJECT NAME}
```
- notify.yml
```
line:
    token: {YOUR TOKEN}
notion:
    token: {YOUR TOKEN}
    url: {YOUR URL}
```

3. Run
```
/bin/bash setup.sh
```

<br>
<br>

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

