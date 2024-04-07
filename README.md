# therapanacea-project


## System dependencies

`Python 3.10.5` with `CUDA 11.6` related to `Pytorch==1.13.1+cu116` and `Torchvision==0.14.1+cu116`.
```bash
# check nvidia divers by running
nvidia-smi
```

## Pyenv
Some ubuntu installs needed which should be installed before installing `python` with `pyenv`:
```
sudo apt-get install libffi-dev
sudo apt-get install libsqlite3-dev
```

 We use pyenv to manage python, currently version `3.10.5`:
```bash
# download pyenv
curl https://pyenv.run | bash

# install python
pyenv install 3.10.5

# select python
pyenv global 3.10.5
```

**Note**: Beware to set your correct `pyenv` version before installing `poetry` to prevent [this issue](https://github.com/python-poetry/poetry/issues/5252#issuecomment-1055697424) from happening. If you have an old version of pyenv, just remove existing pyenv with `sudo rm -r /home/ubuntu/.pyenv`.

## Poetry

Install `Poetry 1.3.1` on your root system following [this documentation](https://www.notion.so/allisone-ai/ML-Dev-environment-on-your-remote-1d5891a6ba6a4ebdb8f0e32e704e71c3#705b1b1887664a02b291a12ba9bdaf55).
Do not use `pip` to install poetry to avoid conflicts between various poetry installs.

```bash
# install all module dependencies inside pyproject.toml
poetry install

# activate virtual environment
poetry shell
```

**Note** : If you activate your environment within your shell with `poetry shell`, you can execute all your commands directly without specifying `poetry run` first.

Select venv in VSCode located at `/home/ubuntu/.cache/pypoetry/virtualenvs`


## Evaluation metrics 

The image classifier has to minimize the Half-Total Error Rate (HTER). HTER is derived from FAR and FRR to provide a balanced evaluation of biometric system performance

### Half-Total Error Rate (HTER)

The Half-Total Error Rate (HTER) is the average of False Acceptance Rate (FAR) and False Rejection Rate (FRR).
HTER provides a comprehensive measure of the overall performance of the classifier, taking into account both false acceptances and false rejections.
$HTER = \frac{FAR + FRR}{2}$

### False Acceptance Rate (FAR)

The False Acceptance Rate (FAR) measures the rate at which unauthorized users are incorrectly accepted by the classifier.
FAR is calculated as the ratio of the number of false acceptances to the total number of identification attempts.
$FAR = \frac{Number ~of ~False ~Acceptances}{Total ~Number ~of ~Genuine ~Identification ~Attempts}$

### False Rejection Rate (FRR)
FRR measures the rate at which authorized users are incorrectly rejected by the classifier.
FRR is calculated as the ratio of the number of false rejections (instances where an authorized user is incorrectly rejected) to the total number of identification attempts for authorized users.
$FRR = \frac{Number ~of ~False ~Rejections}{Total ~Number ~of ~Impostor ~Identification ~Attempts}$

A custom implementation of FRR and HTER has been implemented. 

# Training 

Training has been implemented with different split stratefies: 
- stratified split 
- cross validation 

The user can specify the split strategy within the configuration file 

## Random split training

Execute the stratified split training script by running the Python file. You can use the command line to specify the path to the configuration file using the --config argument. Here's an example command:

```bash
poetry run python therapanacea_project/train/training.py --config therapanacea_project/configs/training/stratified_split/training_resnet18.yaml
```

## Cross-validation training

Run the Training Script: Execute the cross-validation training script by running the Python file. You can use the command line to specify the path to the configuration file using the --config argument. Here's an example command:

```bash
poetry run python therapanacea_project/train/training.py --config therapanacea_project/configs/training/cross_validation/training_resnet18.yaml
```