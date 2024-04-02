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