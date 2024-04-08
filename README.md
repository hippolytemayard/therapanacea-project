# therapanacea-project

- [Therapanacea Project](#therapanacea-project)
  - [Installation](#installation)
    - [Library requirements](#library-requirements)
    - [Install environment](#install-environment)
    - [Pre-commit setup](#pre-commit-setup)


## Installation

### Library requirements

- Python 3.10.5
- Poetry 1.8.2


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

## Poetry

Install `Poetry 1.8.2` on your root system
```bash
curl -sSL https://install.python-poetry.org | python3 - --version 1.8.2
```

Install all module dependencies inside `pyproject.toml`.

```bash
poetry install

# activate virtual environment
poetry shell
```

**Note** : If you activate your environment within your shell with `poetry shell`, you can execute all your commands directly without specifying `poetry run` first.

Select venv in VSCode located at `/home/ubuntu/.cache/pypoetry/virtualenvs`


### Pre-commit setup

Pre-commit modify your code file with some tools (black for example) before each commit:

```bash
poetry run pre-commit install
```

You can run pre-commit manually with

```bash
poetry run pre-commit run --all-files
```


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