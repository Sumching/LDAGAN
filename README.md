# Latent Dirichlet Allocation in Generative Adversarial Network
Code for the image generation experiments in [Latent Dirichlet Allocation in Generative Adversarial Network](https://arxiv.org/abs/1812.06571).


# Usage
We experimented on 5 different datasets:

***CIFAR10, CIFAR100, ImageNet(size 32x32), CelebA and CelebAHQ.***

To train a model, use 

```
python train.py --yaml ./config/dataset_name.yml
```

To generate samples for evaluation, use
```
python test.py --yaml ./config/dataset_name.yml --checkpoint ./output/checkpoint/500000_G.pth --output_name outputname
```

It will return a ".npy" file contains 50,000 samples by default.

# Experiments

|     dataset     |  IS  | FID  |
| :-------------: | :--: | :--: |
|     CIFAR10     | 8.77 | 10.4 |
|    CIFAR100     | 8.81 | 15.2 |
| ImageNet(32x32) | 9.70 | 18.5 |

**./utils/InceptionScore_and_FID.py**

This file contains the implementation of functions to calculate the Inception Score and the FID. It compares the ".npy" file mentioned above with pre-calculated statistic:

```
python ./utils/InceptionScore_and_FID.py --input npy_filename --stats pre_calculated_stats
```



Precalculated statistics for datasets can be found [here](https://drive.google.com/drive/folders/13bTFYdPLHv3QbkVUiq32gumnAqnI1MaH?usp=sharing).
