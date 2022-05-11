# Code Generation with Semi-Supervision and Small Models

There are two main paradigms for doing code generation with language models: 
1. Few-shot prompting, which requires few labeled examples but large, computationally expensive models. An example of an application where we face computational constraints is deployment of code-completion models in IDEs. 
2. Supervised fine-tuning, which is performant with smaller models but requires collecting fully-labeled data, which can be very time-consuming. 

The aim of this project is to move towards a third paradigm where we get the best of both worlds: a paradigm where we require minimal labeled data to train lean models. In this project, we explore two methods for doing this. One is **knowledge transfer**, where a small model trains on the knowledge discovered by a large model. The other is **expert iteration**, where a model teaches itself by training on problems it previously solved. 

## Knowledge Transfer
Knowledge transfer works as follows: we do zero-shot learning on the entire training set with a very large language model such as Codex (by "large" we mean infeasible to fine-tune given user's computational budget). Then, we do supervised fine-tuning with a small model on the Codex solutions that led to the correct answer. 


## Expert Iteration
We seek to find out whether our small model can "climb the hill", that is, solve hard problems by learning to solve easier ones. The way we do this is using expert iteration, first introduced in [Silver et al. (2017)](https://arxiv.org/abs/1712.01815), where a model trains on previously successful trajectories. 


Our experiment is as follows: we sort the MBPP dataset of basic python programs [(Austin et al, 2021)](https://arxiv.org/abs/2108.07732) by length of the solution. We give the model the shortest half of the dataset as fully-supervised training examples, and see if the model is able to solve the harder half of the dataset using expert iteration. 
