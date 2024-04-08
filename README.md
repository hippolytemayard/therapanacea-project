# therapanacea-project

- [Therapanacea Project](#therapanacea-project)
  - [Installation](#installation)
    - [Library requirements](#library-requirements)
    - [Install environment](#install-environment)
    - [Pre-commit setup](#pre-commit-setup)

  - [Data](#data)
  - [Code organisation](#code-organisation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)




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

## Data

The data has to be downloaded and moved in the `/data` with the following arborescence:

```text
data
  |___ train_img/           # Contains training images
  |___ val_img/             # Contains validation images (for inference)
  |___ label_train.txt      # Files containing training labels
  ```


## Code organisation


Source code is available within `therapanacea_project` package, with the following arborescence:

```text
therapanacea_project
  |___ configs
          |____ inference         # Gathers configs for inference
          |____ training     # Gathers configs for training

  |___ dataset
  |___ evaluate
  |___ inference
  |___ losses
  |___ metrics
  |___ models
  |___ optimizer
  |___ train
  |___ utils


data                            # Where to store needed data
tests                           # Unit test files
```

## Training


Training has been implemented based on experiments config files located in `configs/training`.
You can check `therapanacea_project/configs/training/cross_validation/training_resnet18.yaml` for an example of a generic config file.

Training has been implemented with different split stratefies:
- stratified split
- cross validation

The user can specify the split strategy within the configuration file

### Run training

#### Random split

Execute the stratified split training script by running the Python file. You can use the command line to specify the path to the configuration file using the --config argument. Here's an example command:

```bash
poetry run python therapanacea_project/train/training.py --config therapanacea_project/configs/training/stratified_split/training_resnet18.yaml
```

#### Cross-validation

Run the Training Script: Execute the cross-validation training script by running the Python file. You can use the command line to specify the path to the configuration file using the --config argument. Here's an example command:

```bash
poetry run python therapanacea_project/train/training.py --config therapanacea_project/configs/training/cross_validation/training_resnet18.yaml
```

## Evaluation

A training model is directly evaluated from config file.

#### Random split evaluation

Here's an example command with the stratified split strategy configuration file :

```bash
poetry run python therapanacea_project/evaluate/evaluate_stratified_split.py --config therapanacea_project/configs/training/stratified_split/training_resnet18.yaml
```

#### Cross-validation evaluation

Here's an example command with the cross-validation strategy configuration file :

```bash
poetry run python therapanacea_project/evaluate/evaluate_cross_validation.py --config therapanacea_project/configs/training/cross_validation/training_resnet18.yaml
```

#### Evaluation metrics

The image classifier has to minimize the Half-Total Error Rate (HTER). HTER is derived from FAR and FRR to provide a balanced evaluation of biometric system performance

##### Half-Total Error Rate (HTER)

The Half-Total Error Rate (HTER) is the average of False Acceptance Rate (FAR) and False Rejection Rate (FRR).
HTER provides a comprehensive measure of the overall performance of the classifier, taking into account both false acceptances and false rejections.
$HTER = \frac{FAR + FRR}{2}$

##### False Acceptance Rate (FAR)

The False Acceptance Rate (FAR) measures the rate at which unauthorized users are incorrectly accepted by the classifier.
FAR is calculated as the ratio of the number of false acceptances to the total number of identification attempts.
$FAR = \frac{Number ~of ~False ~Acceptances}{Total ~Number ~of ~Genuine ~Identification ~Attempts}$

##### False Rejection Rate (FRR)
FRR measures the rate at which authorized users are incorrectly rejected by the classifier.
FRR is calculated as the ratio of the number of false rejections (instances where an authorized user is incorrectly rejected) to the total number of identification attempts for authorized users.
$FRR = \frac{Number ~of ~False ~Rejections}{Total ~Number ~of ~Impostor ~Identification ~Attempts}$

A custom implementation of FRR and HTER has been implemented.

## Inference

The inference on validation images is done using inference config files located within the `config/inference` directory.

#### Random split evaluation

Here's an example command with the stratified split strategy configuration file :

```bash
poetry run python therapanacea_project/inference/predict.py --config therapanacea_project/configs/inference/stratified_split/inference_resnet18.yaml
```

#### Cross-validation evaluation

Here's an example command with the cross-validation strategy configuration file :

```bash
poetry run python therapanacea_project/inference/cross_val_predict.py --config therapanacea_project/configs/inference/cross_validation/inference_resnet18.yaml
```

## Test

### Unit tests

A unit test folder has been initiate. To run unit test you can

```bash
poetry run pytest --cov=.
```

## Results

The Table above displays the results obtained training a resnet18 and a resnet34 architecture using a 5-Fold cross validation.
The config files can be found in `config/training/cross_validation/resnet18.yaml` (`resnet34.yaml` respectively).

The selected model for submitting prediction is the resnet18 architecture (in bold).

|         | FAR      | FRR      | HTER     |
|---------|----------|----------|----------|
| **resnet18**| **0.083 ± 0.012**  | **0.113 ±  0.013** | **0.096  ± 0.002**|
| resnet34| 0.082 ± 0.015| 0.113 ± 0.014 | 0.097 ± 0.001 |
| vit_b_16| na| na | na |

The results of the 5-Folds are given in `notebooks/models_evaluation.ipynb`

The experiment using a ViT architecture (`vit_b_16`) are not displayed in the table as the training time was too long using a Tesla T4 GPU.