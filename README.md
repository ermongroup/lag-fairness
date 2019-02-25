# Learning Controllable Fair Representations

TensorFlow implementation for the paper [Learning Controllable Fair Representations](https://arxiv.org/abs/1812.04218), AISTATS 2019.

## Overview

## Running the experiments

Requirements: 

- Tensorflow
- pandas
- numpy
- tf_utils

### How to install tf_utils
```
git clone git@github.com:jiamings/tf_utils.git
cd tf_utils
pip install -e .
```

### Running MIFR (Adult)
```
python -m exmaples.adult
```

### Running L-MIFR (Adult)
```
python -m examples.adult --lag
```

### Options
If MIFR then the e hyperparameter values corresponds to individual `lambda` parameters, if L-MIFR then they correspond to `epsilon` contraints in the paper.
 - `e1`: Upper bound for MI
 - `e2`: Adversarial approximation to Demographic parity
 - `e4`: Adversarial approximation to Equalized odds
 - `e5`: Adversarial approximation to Equalized opportunity
 - `disc`: discriminator iterations

## References

If you find the idea or code useful for your research, please consider citing our paper:
```
@article{song2019learning,
  title={Learning Controllable Fair Representations},
  author={Song, Jiaming and and Grover, Aditya and Zhao, Shengjia and Ermon, Stefano},
  journal={arXiv preprint arXiv:1812.04218},
  year={2018}
}
```

## Acknowledgements

`utils/logger.py` is based on an implementation in [OpenAI Baselines](https://github.com/openai/baselines).

## Contact

`tsong [at] cs [dot] stanford [dot] edu`




