# Code Generation with Semi-Supervision and Small Models

There are two main paradigms for doing code generation with language models: 
1. Few-shot prompting, which requires few labeled examples but large, computationally expensive models. An example of an application where we face computational constraints is deployment of code-completion models in IDEs. 
2. Supervised fine-tuning, which is performant with smaller models but requires collecting fully-labeled data, which can be very time-consuming. 

The aim of this project is to move towards a third paradigm where we get the best of both worlds: a paradigm where we require minimal labeled data to train lean models. In this project, we explore two methods for doing this. One is **knowledge transfer**, where a small model trains on the knowledge discovered by a large model. The other is **expert iteration**, where a model teaches itself by training on problems it previously solved. 

## Knowledge Transfer
We do our knowledge transfer experiments on what I'm going to call the *math-small* dataset, a restriction of [Deepmind's Math dataset](https://github.com/deepmind/mathematics_dataset), which contains 4000 training examples randomly sampled from Deepmind's full 2 million. Importantly, this dataset contains only questions and answers, but not labeled solutions. 

Knowledge transfer works as follows: we do zero-shot learning on the entire training set with a very large language model such as Codex (by "large" we mean infeasible to fine-tune given user's computational budget). Then, we do supervised fine-tuning with a small model on the Codex solutions that led to the correct answer. 

|Model| Teacher| Model Training/Prompting Method| Accuracy|
|-----|--------|--------------------------------|---------|
|CodeNet 350M| None | Zero-shot | ? | 
|CodeNet 350M| Codex | Knowledge Transfer | ? | 

## Expert Iteration
We seek to find out whether our small model can "climb the hill", that is, solve hard problems by learning to solve easier ones. 
