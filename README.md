# AUVAE-Net
Code for Adversarial U-shaped Variational Autoencoder Network (AUVAE-Net) —— A Deep Learning Method for Hyperspectral Image Unmixing in Remote Sensing
# Abstract
<p>
Hyperspectral unmixing is an important technique in RS image interpretation. Its goal is to decompose the pixel spectra into a set of extracted endmembers and their associated abundance maps, so as to achieve precise identification and classification of ground objects. Although traditional unmixing methods have advantages in physical interpretability, they face significant limitations in handling complex nonlinear mixing, noise interference, and large-scale data. While in recent years, deep learning has demonstrated remarkable advantages in this area due to its powerful feature extraction and modeling capabilities.
</p>
<p>
Based on the AAE-Net, this study proposes an improved deep learning-based hyperspectral unmixing model—Adversarial U-shaped Variational Autoencoder Network (AUVAE-Net). This model aims to combine U-Net and VAE within the unmixing network to fully leverage the strengths of both: U-Net structure facilitates better preservation of local detail features of ground objects in the image, while VAE module enhances robustness and generalization by representing abundance distributions and perturbation sampling through the mean and variance of latent variables. Furthermore, the model incorporates the adversarial training strategy of AAE-Net to constrain the prior characteristics of abundance distribution, effectively improving the physical consistency of unmixing results.
</p>
<p>
Experimental results show that AUVAE-Net, by integrating U-Net's multi-scale features with VAE's probabilistic modeling mechanism, achieves certain advantages in unmixing accuracy over traditional methods and existing deep learning models. It provides a novel approach for deep-learning-based hyperspectral remote sensing image unmixing.
</p>
<b>Keywords</b>: remote sensing; hyperspectral unmixing; deep learning; U-Net; VAE
# Environment
* Windows
* python 3.10.16
* pytorch 1.12.1
* cudatoolkit 11.2.2 + cudnn 8.1.0.77
* keras 2.10.0
* numpy 1.23.5
* tensorflow-gpu 2.10.0
# Reference
[1] [Adversarial Autoencoder Network for Hyperspectral Unmixing](https://github.com/qiwenjjin/AAENet)<br>
[2] [Hyperspectral Unmixing Models](https://github.com/UPCGIT/Hyperspectral-Unmixing-Models)
