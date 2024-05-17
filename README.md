# CS6910-Assignment3
Deep Learning Assignment 3 - Use recurrent neural networks to build a transliteration system [Link](https://wandb.ai/cs6910-dl-assignments/assignment%203/reports/Assignment-3--Vmlldzo3NTUwNzY4?accessToken=cb5ahfcp8eisq1oe6ixumae10ttzpp16rtdbtsfm30le7l9zgdqko388iasvrh93)

`train.py` can used to train networks without attention. `run.sh`is wrapper file for `train.py`. Arguments can be changed inside `run.sh`. After changing arguments, you can simply execute it. Keep `use_wandb=false` if you don't want to sweep for hyperparameters.
```
./run.sh
```
If using train.py directly, you will have to pass all the necessary arguments. \
To see usage:
```
python train.py --help
```
To run network with attention execute `EncoderDecoderWithAttn.py`. Executing this file will create a folder `prediction_attentions` with test set predictions. It will also save a attention heat map.

Link to my wandb report: [Link](https://wandb.ai/arjungangwar/CS6910-Assignment3/reports/Assignment-3--Vmlldzo3OTE4ODE1)
