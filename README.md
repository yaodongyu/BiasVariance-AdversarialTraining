# BiasVariance-AdversarialTraining

This is the code for the paper "Understanding Generalization in Adversarial Training via the Bias-Variance Decomposition".


<p align="center">
    <img src="images/main_fig.pdf" width="600"\>
</p>
<p align="center">
Risk, bias, and variance for ResNet34 on CIFAR10 dataset (25,000 training samples).
</p>


## Prerequisites
* Python
* Pytorch (1.3.1)
* CUDA
* numpy

There are 4 folders, ```cifar10```, ```cifar100```, ```2d```, and ```logistic_regression```. 
First ```cd``` into the directory. 

### CIFAR10
To run the L-infinity adversarial training with ```eps=8.0``` on the CIFAR10 dataset saved to folder ```model_Linf_eps8```, run
```text
python train_adv.py --norm l_inf --fname model_Linf_eps8 --epsilon 8 --width-factor 10
```
To evaluate the standard (squared loss) bias-variance on the above model ```model_Linf_eps8``` on epoch ```200```, run
```text
python eval_adv_bv_mse.py  --fname model_Linf_eps8 --resume 200 --attack none
```
To evaluate the adversarial (squared loss) bias-variance on the above model ```model_Linf_eps8``` on epoch ```200``` 
with perturbation size ```eps=6```, run
```text
python eval_adv_bv_mse.py  --fname model_Linf_eps8 --resume 200 --attack pgd --epsilon 6 
```
To evaluate the standard (cross-entropy) bias-variance on the above model ```model_Linf_eps8``` on epoch ```200```, run
```text
python eval_adv_bv_kl.py  --fname model_Linf_eps8 --resume 200 --attack none
```

### CIFAR100
To run the L-infinity adversarial training with ```eps=8.0``` on the CIFAR100 dataset saved to folder ```model_Linf_eps8```, run
```text
python train_adv.py --norm l_inf --fname model_Linf_eps8 --epsilon 8
```
To evaluate the standard (squared loss) bias-variance on the above model ```model_Linf_eps8``` on epoch ```200```, run
```text
python eval_adv_bv_mse.py  --fname model_Linf_eps8 --resume 200 --attack none
```

### 2D box example
To reproduce the 2D box example results, run ```2d_bv.ipynb```.

### Linear Logistic Regression
To reproduce the logistic_regression results, run ```logistic_regression.ipynb```.
