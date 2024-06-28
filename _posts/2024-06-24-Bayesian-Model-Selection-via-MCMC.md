---
layout: distill
title: Bayesian Model Selection via MCMC
description: Bayesian model selection
tags: Bayesian model/variable selection
giscus_comments: false
date: 2024-06-24
featured: true

bibliography: 2024-06-24-Bayesian-Model-Selection-via-MCMC.bib

toc:
  - name: Method
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Implementaton

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Method <d-cite key="carlin1995bayesian"></d-cite>

Choose between $$K$$ models with corresponding parameter vector $$\boldsymbol{\theta}_j$$, $$j=1,...K$$. Let $$M$$ be an integer-valued parameter that indexes the model, for model $$j$$, we have a likelihood $$f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,M=j)$$ and a prior $$p(\boldsymbol{\theta}_j\mid M=j)$$. Given $$M=j$$, $$\boldsymbol.{y}$$ is independent of $$\{\boldsymbol{i\neq j}\}$$. Assume that given the indicator $$M$$, $$\boldsymbol{\theta}_j$$ are independent of each other, we can complete the Bayesian model specification by choosing proper `pseudopriors` $$p(\boldsymbol{\theta}_j\mid M\neq j)$$, which is a conveniently chosen linking density. Reason shows below,
$$
p(\boldsymbol{y} \mid M=j)=\int f(\boldsymbol{y}\mid \boldsymbol{\theta},M=j)p(\boldsymbol{\theta}\mid M=j)d\boldsymbol{\theta}=\int f(\boldsymbol{y}\mid \boldsymbol{\theta}_{j},M=j)p(\boldsymbol{\theta}_{j}\mid M=j)d\boldsymbol{\theta}
$$
Given prior model probabilities $$\pi_{j}\equiv P(M=j)$$ such that $$\sum_{j=1}^{K}\pi_{j}=1$$, let $$\boldsymbol{\theta}=\{\boldsymbol{\theta}_1,\ldots,\boldsymbol{\theta}_K\}$$, when $$M=j$$, the joint distribution of $$\boldsymbol{y}$$ and $$\boldsymbol{\theta}$$ is 

$$
\begin{aligned} 
p(\boldsymbol{y},\boldsymbol{\theta},M=j) & = f(\boldsymbol{y}\mid \boldsymbol{\theta},m=j)p(\boldsymbol{\theta},M=j)p(M=j) \\
 & = f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,m=j)p(\boldsymbol{\theta},M=j)p(M=j) \\
 & = f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,m=j) \prod_{i=1}^{K} p(\boldsymbol{\theta}_i,M=j) p(M=j)\\
 & = f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,m=j) \prod_{i=1}^{K} p(\boldsymbol{\theta}_i,M=j) \pi_{j}
\end{aligned}
$$

To implement Gibbs sampler the full conditional distributions of each $$\boldsymbol{\theta}_j$$ and $$M$$.
For $$\boldsymbol{\theta}_j$$, when $$M=j$$, we generate from the usual model $$j$$ full conditional; when $$M\neq j$$, we generate from the linking function (`pseudoprior`).

$$
p(\boldsymbol{\theta}_j \mid \boldsymbol{\theta}_({i\neq j},M,\boldsymbol{y}) = \begin{cases}
   f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,M=j)p(\boldsymbol{\theta}_j\mid M = j) & M=j, \\
   p(\boldsymbol{\theta}_j\mid M\neq j) & M\neq j,
\end{cases}
$$

For discrete finite parameter $$M$$:

$$
  p(M=j\mid \boldsymbol{\theta},\boldsymbol{y})=\frac{f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,M=j)\prod_{i=1}^{K} p(\boldsymbol{\theta}_i\mid M=j) \pi_j}{\sum_{k=1}^{K} \left(f(\boldsymbol{y}\mid \boldsymbol{\theta}_k,M=k)\prod_{i=1}^{K} p(\boldsymbol{\theta}_i\mid M=k) \pi_k \right)}
$$

The algorithm will produce samples from the correct joint posterior distribution. The ratio

$$
\hat{p}(M=j\mid \boldsymbol{y})=\frac{\text{number of }M^{(g)}=j}{\text{total number of }M^{(g)}},\quad j=1,\ldots,K.
$$

provides estimates that be used to compute the Bayes factor (ratio of the observed marginal densities for the two models)

$$
B_{ij}=\frac{p(\boldsymbol{y}\mid M=i)}{p(\boldsymbol{y}\mid M=j)}
$$

between any two of the models.

--- 
## Implementaton <d-cite key="carlin1995bayesian"></d-cite> <d-cite key="jauch2021mixture"></d-cite>

Poor choices of the linking density (`pseudopriors`) $$p(\boldsymbol{\theta}_j\mid M\neq j)$$ will make jumps between models extremely unlikely, so that the convergence of the Gibbs sampling may trapped to one model, which might not be the true one in fact. Good choices will produce $$boldsymbol{\theta}_j^{(g)}$$-values that are consistent with the data, so that $$p(M=j\mid \boldsymbol{\theta},\boldsymbol{y})$$ will still be reasonably large at the next $$M$$ update step. 

Key point: Use the data to help to select the `pseudopriors` but `not` the prior, match the `pseudopriors` as nearly as possible to the true model-specific posteriors. 



---
