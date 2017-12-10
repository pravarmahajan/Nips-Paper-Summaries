# README

This repository contains the summaries of projects in this organisations.



# How to add your own Summary
1. Fork this repo
2. Add a folder with your Paper Name
3. Create a Readme.md file in that folder
4. Write your summary there. You can include additional files (E.g. reference images) under your folder
5. Update the Readme.md at the root of this project (This File), to include a short summary of your paper
5. Submit a pull request on github



### This is done to prevent merge conflicts

___

# NIPS - 2017 Paper Summaries

### Towards Accurate Binary Convolutional Neural Network
**Link**: https://papers.nips.cc/paper/6638-towards-accurate-binary-convolutional-neural-network.pdf

**Summary** : This paper talks about using only -1 or +1 as weights and activations while building CNNs. Paper claims that this tweak to the traditional CNNs uses significantly lesser amount of memory, faster inference in runtime on test data and comparable performance on IMAGENET.

**To-do** to implement the paper:
  - Constrain the weights to {-1, +1} - This ensures convolutions are only additions or subtractions
  - Use five binary activations instead of two.
  - ...(Work in progress)

**Implementations and Results**:
  - Detailed Summary and Results: Link-to-Your-Folder-Name-In-This-Repo. Code: Your-Repo-Here.
  
  ### Selective Classification for Deep Neural Networks
  **Link**: https://papers.nips.cc/paper/7073-selective-classification-for-deep-neural-networks.pdf
  
  **Summary** : This paper proposes a new method to construct a selective classifier for a given trained DNN. At test time the classifier rejects instances as needed to grant the desired risk. Performance is measured on ImageNet, CIFAR-10, CIFAR-100 datasets.
  
  **To-do** to implement the paper:
   - Implement SR and MC-dropout confidence-rate functions, κf , and the induced rejection function, gθ(x|κf ) on trained models.
