---
layout: distill
title: Variational Inference
description: Variational Bayesian
tags: Optimization, Variational
giscus_comments: false
date: 2024-06-28
featured: true

bibliography: 2024-06-28-Variational-Inference.bib

toc:
  - name: Preliminary
  - name: Inference
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Implementation

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
Use optimization rather than use sampling. 
First posit a family of densities, then to find a member of that family which is close to the target density. Use exponential family as an example.

## Preliminary
[Exponential families](https://en.wikipedia.org/wiki/Exponential_family) include [normal](https://en.wikipedia.org/wiki/Normal_distribution), [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution), [exponential](https://en.wikipedia.org/wiki/Exponential_distribution), [inverse Gaussian](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution), [gamma](https://en.wikipedia.org/wiki/Gamma_distribution), [chi-squared](https://en.wikipedia.org/wiki/Chi-squared_distribution), [beta](https://en.wikipedia.org/wiki/Beta_distribution), [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution), [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution), [categorical](https://en.wikipedia.org/wiki/Categorical_distribution), [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution), [Wishart](https://en.wikipedia.org/wiki/Wishart_distribution), [inverse Wishart](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution), [geometric](https://en.wikipedia.org/wiki/Geometric_distribution), [binomial](https://en.wikipedia.org/wiki/Binomial_distribution)(with fixed number of failures), [multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution)(with fixed number of failures), [negative binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution)(with fixed number of failures), [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution)(with fixed shape parameter)...

For variable $$\boldsymbol{x}=(x_1,\ldots,x_k)^{T}$$, a family of distributions with paramter $$\boldsymbol{\theta}\equiv (\theta_1,\ldots,\theta_s)^{T}$$ is said to belong to an exponential family if the p.d.f (or p.m.f) can be written as

$$
f_X(\boldsymbol{X}\mid\boldsymbol{\theta})=h(\boldsymbol{x})exp\left(\sum_{i=1}^{s} \eta_{i}(\boldsymbol{\theta})T_i(\boldsymbol{x})-A(\boldsymbol{\theta})\right)
$$

or campactly 

$$
f_X(\boldsymbol{X}\mid\boldsymbol{\theta})=h(\boldsymbol{x})exp\left(\boldsymbol{\eta}(\boldsymbol{\theta})\cdot T(\boldsymbol{x})-A(\boldsymbol{\theta})\right)
$$

The dimensions $$k$$ of the random variable need not match the dimension $$d$$ of the parameter vector, nor (in the case of a curved exponential function) the dimension $$s$$ of the natural parameter 
$$\boldsymbol{\eta}$$ and sufficient statistic $$T(\boldsymbol{x})$$ .

## Inference <d-cite key="blei2017variational"></d-cite>

## Implementation <d-cite key="blei2017variational"></d-cite> <d-cite key="blei2006variational"></d-cite>

---