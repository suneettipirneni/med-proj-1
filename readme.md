# Medical Imaging Assignment 1

## Usage

1. Firstly install the required dependencies

```bash
pip install -r requirements.txt
```

2. Download the dataset from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. Place the downloaded data in the `data/` directory.

3. Run the main script

```bash
python main.py <task> <model>
```

where `task` is can be `train` or `test`, and `model` can be `1` or `2`.

> Model 1 corresponds to the model trained from scratch. Model 2 corresponds to the model trained using transfer learning.
