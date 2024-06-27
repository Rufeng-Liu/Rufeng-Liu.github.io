---
layout: distill
title: Bayesian Model Selection via MCMC
description: Bayesian model selection
tags: Bayesian model/variable selection
giscus_comments: false
date: 2024-06-24
featured: true

bibliography: 1995-Carlin.bib

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

## Method

Choose between $$K$$ models with corresponding parameter vector $$\boldsymbol{\theta}_j$$, $$j=1,...K$$. Let $$M$$ be an integer-valued parameter that indexes the model, for model $$j$$, we have a likelihood $$f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,M=j)$$ and a prior $$p(\boldsymbol{\theta}_j\mid M=j)$$. Given $$M=j$$, $$\boldsymbol.{y}$$ is independent of $$\{\boldsymbol{i\neq j}\}$$. Assume that given the indicator $$M$$, $$\boldsymbol{\theta}_j$$ are independent of each other, we can complete the Bayesian model specification by choosing proper `pseudopriors` $$p(\boldsymbol{\theta}_j\mid M\neq j)$$, which is a conveniently chosen linking density. Reason shows below,
$$
p(\boldsymbol{y} \mid M=j)=\int f(\boldsymbol{y}\mid \boldsymbol{\theta},M=j)p(\boldsymbol{\theta}\mid M=j)d\boldsymbol{\theta}=\int f(\boldsymbol{y}\mid \boldsymbol{\theta}_{j},M=j)p(\boldsymbol{\theta}_{j}\mid M=j)d\boldsymbol{\theta}
$$
Given prior model probabilities $$\pi\equiv P(M=j)$$ such that $$\sum_{j=1}^{K}\pi_{j}=1$$, let $$\boldsymbol{\theta}=\{\boldsymbol{\theta}_1,\ldots,\boldsymbol{\theta}_K\}$$, when $$M=j$$, the joint distribution of $$\boldsymbol{y}$$ and $$\boldsymbol{\theta}$$ is 

\begin{equation} 
\begin{split}
p(\boldsymbol{y},\boldsymbol{\theta},M=j) & = f(\boldsymbol{y}\mid \boldsymbol{\theta},m=j)p(\boldsymbol{\theta},M=j)p(M=j) \\
 & = f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,m=j)p(\boldsymbol{\theta},M=j)p(M=j) \\
 & = f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,m=j) \prod_{i=1}^{K} p(\boldsymbol{\theta}_j,M=j) p(M=j)
\end{split}
\end{equation}

\begin{eqnarray*}
p(\boldsymbol{y},\boldsymbol{\theta},M=j) = f(\boldsymbol{y}\mid \boldsymbol{\theta},m=j)p(\boldsymbol{\theta},M=j)p(M=j) = f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,m=j)p(\boldsymbol{\theta},M=j)p(M=j) = f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,m=j)\prod_{i=1}^{K} p(\boldsymbol{\theta}_j,M=j)p(M=j)
\end{eqnarray*}

--- 
## Implementaton

---
