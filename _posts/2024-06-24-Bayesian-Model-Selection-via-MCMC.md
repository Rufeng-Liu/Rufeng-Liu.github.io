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

Choose between $$K$$ models with corresponding parameter vector $$\boldsymbol{\theta}_j$$, $$j=1,...K$$.

$$
p(y \mid M=j)=\int f(y\mid \boldsymbol{\theta},M=j)p(\boldsymbol{\theta}\mid M=j)d\boldsymbol{\theta}=\int f(y\mid \boldsymbol{\theta}_{j},M=j)p(\boldsymbol{\theta}_{j}\mid M=j)d\boldsymbol{\theta}
$$
--- 
## Implementaton

---
