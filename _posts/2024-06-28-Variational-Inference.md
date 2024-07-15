---
layout: distill
title: Variational Inference
description: Variational Bayesian
tags: Optimization Variational Dirichlet-process
giscus_comments: false
date: 2024-06-28
featured: true

bibliography: 2024-06-28-Variational-Inference.bib

toc:
  - name: Exponential families
  - name: Dirichlet process and Dirichlet process mixture 
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

## Exponential families
[Exponential families](https://en.wikipedia.org/wiki/Exponential_family) include [normal](https://en.wikipedia.org/wiki/Normal_distribution), [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution), [exponential](https://en.wikipedia.org/wiki/Exponential_distribution), [inverse Gaussian](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution), [gamma](https://en.wikipedia.org/wiki/Gamma_distribution), [chi-squared](https://en.wikipedia.org/wiki/Chi-squared_distribution), [beta](https://en.wikipedia.org/wiki/Beta_distribution), [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution), [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution), [categorical](https://en.wikipedia.org/wiki/Categorical_distribution), [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution), [Wishart](https://en.wikipedia.org/wiki/Wishart_distribution), [inverse Wishart](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution), [geometric](https://en.wikipedia.org/wiki/Geometric_distribution), [binomial](https://en.wikipedia.org/wiki/Binomial_distribution)(with fixed number of failures), [multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution)(with fixed number of failures), [negative binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution)(with fixed number of failures), [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution)(with fixed shape parameter)...

For variable $$\boldsymbol{x}=(x_1,\ldots,x_k)^{T}$$, a family of distributions with paramter $$\boldsymbol{\theta}\equiv (\theta_1,\ldots,\theta_s)^{T}$$ is said to belong to an exponential family if the p.d.f (or p.m.f) can be written as

$$
f_X(\boldsymbol{X}\mid\boldsymbol{\theta})=h(\boldsymbol{x})\text{exp}\left(\sum_{i=1}^{s} \eta_{i}(\boldsymbol{\theta})T_i(\boldsymbol{x})-A(\boldsymbol{\theta})\right)
$$

or campactly 

$$
f_X(\boldsymbol{X}\mid\boldsymbol{\theta})=h(\boldsymbol{x})\text{exp}\left(\boldsymbol{\eta}(\boldsymbol{\theta})\cdot T(\boldsymbol{x})-A(\boldsymbol{\theta})\right)
$$

The dimensions $$k$$ of the random variable need not match the dimension $$d$$ of the parameter vector, nor (in the case of a curved exponential function) the dimension $$s$$ of the natural parameter 
$$\boldsymbol{\eta}$$ and sufficient statistic $$T(\boldsymbol{x})$$ .

## Dirichlet process and Dirichlet process mixture 

Citation <d-cite key="ferguson1973bayesian"></d-cite>

A Dirichlet process $$G$$ is parameterized by a centering measure $$G_0$$ and a positive presicion/scaling parameter $$\alpha$$, if for all natural numbers $$k$$ and $$k$$-partitions $$\{B_1,\ldots,B_k\}$$:

$$
\left(G(B_1),\ldots,G(B_k)\right)\sim \text{Dir}\left(\alpha G_0(B_1),\ldots,\alpha G_0(B_k)\right).
$$

Suppose we independently draw $$N$$ random variables $$\eta_n$$ from $$G$$:

$$
\begin{aligned} 
G\mid G_0,\alpha &\sim \text{DP}(G_0,\alpha)\\
\eta_n &\sim G, \quad n\in\{1,\ldots,N\}.
\end{aligned}
$$

Conditioning on $$n − 1$$ draws, the $$n$$th value is, with positive probability, exactly equal to one of those draws:

$$
p(\cdot\mid \eta_1,\ldots,\eta_{n-1})\propto \alpha G_0(\cdot)+\sum_{i=1}^{n-1} \delta_{\eta_i}(\cdot).
$$

Thus, the variables $$\{\eta_1,\ldots,\eta_{n−1}\}$$ are randomly partitioned according to which variables are equal to the same value, with the distribution of the partition obtained from a Polya urn scheme.

Let $$\{\eta^{*}_{1},\ldots,\eta^{*}_{\lvert \boldsymbol{c} \rvert }\}$$ denote the distinct values of $$\{\eta_1,\ldots,\eta _{n-1}\}$$, let $$\boldsymbol{c} = \{c_1,...,c_ {n−1}\}$$ be assignment variables such that $$\eta_i = \eta^*_ {c_i}$$, and let $$\lvert\boldsymbol{c}\rvert$$ denote the number of cells in the partition. The distribution of $$\eta_n$$ follows the urn distribution:

$$
\eta_n =  \begin{cases}
               \eta^*_i & \text{with prob.} \frac{|\lbrace j:c_j=i \rbrace|}{n-1+\alpha} \\
               \eta, \eta\sim G_0  & \text{with prob.} \frac{\alpha}{n-1+\alpha},
          \end{cases}
$$

where 

$$|\{j : c_ {j}=i\}|$$

is the number of times the value $$\eta^{*}_{i}$$ occurs in $$\{\eta_{1},\ldots,\eta_{n−1}\}$$.

Given Dirichlet process $$G$$, a DP mixtures are densities $$p(x)=\int p(x, \eta)d\eta$$, or we can have non-i.i.d observations $$x_n\overset{ind}{\sim}p_{n,G}(x)=\int p(x;\eta)dG(\eta)$$, in terms of $$N$$ latent variables $$\eta_1,\ldots,\eta_N$$, the model can be written as 

$$
x_n\mid\eta_n\overset{ind}{\sim}p_n(x_n;\eta_n), \quad \eta_n\mid G\overset{i.i.d}{\sim}G, \quad G\mid G_0,\alpha \sim \text{DP}(G_0,\alpha)
$$

Given a sample $$\{x_1,\ldots,x_N\}$$ from a DP mixture, the predictive density is

$$
p(x\mid x_1,\ldots,x_N,\alpha,G_0)=\int p(x\mid \eta)p(\eta\mid x_1,\ldots,x_N,\alpha,G_0)d\eta
$$

which we can use MCMC to achieve posterior draws, together with posterior distribution $$p(\eta\mid x_1,\ldots,x_N,\alpha,G_0)$$.

The stick-breaking representation <d-cite key="sethuraman1994constructive"></d-cite> is widely used. Consider two infinite collections of independent random variables, $$V_i\sim\text{Beta}(1,\alpha)$$ and $$\eta^*_i\sim G_0$$, for $$i=\{1,2,\ldots\}$$. The stick-breaking     representation of $$G$$ is as follows:

$$
G=\sum_{i=1}^{\infty} \pi_i(\boldsymbol{v})\delta_{\eta^*_i}, \quad \pi_i(\boldsymbol{v})=v_{i} \prod_{j=1}^{i-1}(1-v_j)
$$

In the DP mixture, the vector $$\pi(\boldsymbol{v})$$ comprises the infinite vector of mixing proportions and $$\{\eta^*_1,\eta^*_2,\ldots\}$$ are the atoms representing the mixture components. Let $$Z_n$$ be an assignment variable of the mixture component with which the data point $$x_n$$ is associated. The data can be described as arising from the following process:

1. Draw $$V_i\sim \text{Beta}(1,\alpha), \quad i=\{1,2,\ldots\}$$
2. Draw $$\eta^*_i\mid G_0\sim G_0, \quad \quad i=\{1,2,\ldots\}$$
3. For the $$n$$th data point:
   
   (a) Draw $$Z_n\mid \{v_1,v_2,\ldots\}\sim \text{Mult}(\pi(\boldsymbol{v}))$$
   (b) Draw $$X_n\mid z_n\sim p(x_n\mid \eta^*_{z_n})$$

Restrict the DP mixtures that the observable data are drawn from an exponential family distribution, and where the base distribution for the DP is the corresponding conjugate prior.

The distribution of $$X_n$$ conditional on $$Z_n$$ and $${\eta^*_1,\eta^*_2,\ldots}$$ is:

$$
p(x_n\mid z_n,\eta^*_1,\eta^*_2,\ldots)=\prod_{i=1}^{\infty} \left(h(x_n) \text{exp}\left\{{\eta^*_i}^T x_n-a(\eta^*_i)\right\} \right)^{\mathbf{1}\lbrack z_n=i\rbrack}
$$

where $$a(\eta^*_i)$$ is the appropriate cumulant function and we assume for simplicity that $$x$$ is the sufficient statistic for the natural parameter $$\eta$$.

Thevector of sufficient statistics of the corresponding conjugate family is $$({\eta^*_i}^T, -a(\eta^*_i))^T$$. The base distribution is:

$$
p(\eta^*\mid \lambda) = h(\eta^*) \text{exp}\left\{\lambda_1^T \eta^* + \lambda_2 (-a(\eta^*))-a(\eta^*)\right\}
$$

where we decompose the hyperparameter $$\lambda$$ such that $$\lambda_1$$ contains the first $$\dim(\eta^*)$$ components and $$\lambda_2$$ is a scalar.

## Inference 
Citation <d-cite key="blei2017variational"></d-cite>

## Implementation 
Citation <d-cite key="blei2017variational"></d-cite> <d-cite key="blei2006variational"></d-cite>

---
