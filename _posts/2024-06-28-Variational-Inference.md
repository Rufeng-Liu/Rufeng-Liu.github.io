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
    subsections:
       - name: Gibbs sampling
         subsubsections:
            - name: Collapsed Gibbs sampling
            - name: Blocked Gibbs sampling
       - name: Variational inference
         subsubsections:
            - name: Mean field variational inference in exponential families
            - name: Coordinate ascent algorithm for DP mixtures    
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

A `Dirichlet process` $$G$$ is parameterized by a centering measure $$G_0$$ and a positive presicion/scaling parameter $$\alpha$$, if for all natural numbers $$k$$ and $$k$$-partitions $$\{B_1,\ldots,B_k\}$$:

$$
\left(G(B_1),\ldots,G(B_k)\right)\sim \text{Dir}\left(\alpha G_0(B_1),\ldots,\alpha G_0(B_k)\right).
$$

Suppose $$N$$ random variables $$\eta_n$$ are independently drawn from $$G$$:

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

Given Dirichlet process $$G$$, a DP mixtures are densities $$p(x)=\int p(x, \eta)d\eta$$, or there can be non-i.i.d observations $$x_n\overset{ind}{\sim}p_{n,G}(x)=\int p(x;\eta)dG(\eta)$$, in terms of $$N$$ latent variables $$\eta_1,\ldots,\eta_N$$, the model can be written as 

$$
x_n\mid\eta_n\overset{ind}{\sim}p_n(x_n;\eta_n), \quad \eta_n\mid G\overset{i.i.d}{\sim}G, \quad G\mid G_0,\alpha \sim \text{DP}(G_0,\alpha)
$$

Given a sample $$\{x_1,\ldots,x_N\}$$ from a DP mixture, the predictive density is

$$
p(x\mid x_1,\ldots,x_N,\alpha,G_0)=\int p(x\mid \eta)p(\eta\mid x_1,\ldots,x_N,\alpha,G_0)d\eta
$$

which one can use MCMC to achieve posterior draws, together with posterior distribution $$p(\eta\mid x_1,\ldots,x_N,\alpha,G_0)$$.

The `stick-breaking` representation <d-cite key="sethuraman1994constructive"></d-cite> is widely used. Consider two infinite collections of independent random variables, $$V_i\sim\text{Beta}(1,\alpha)$$ and $$\eta^*_i\sim G_0$$, for $$i=\{1,2,\ldots\}$$. The stick-breaking representation of $$G$$ is as follows:

$$
G=\sum_{i=1}^{\infty} \pi_i(\boldsymbol{v})\delta_{\eta^*_i}, \quad \pi_i(\boldsymbol{v})=v_{i} \prod_{j=1}^{i-1}(1-v_j)
$$

In the DP mixture, the vector $$\pi(\boldsymbol{v})$$ comprises the infinite vector of mixing proportions and $$\{\eta^*_1,\eta^*_2,\ldots\}$$ are the atoms representing the mixture components. Let $$Z_n$$ be an assignment variable of the mixture component with which the data point $$x_n$$ is associated. The data can be described as arising from the following process:

1. Draw $$V_i\sim \text{Beta}(1,\alpha), \quad i=\{1,2,\ldots\}$$
2. Draw $$\eta^*_i\mid G_0\sim G_0, \quad \quad i=\{1,2,\ldots\}$$
3. For the $$n$$th data point:
   
   (a) Draw $$Z_n\mid \{v_1,v_2,\ldots\}\sim \text{Mult}(\pi(\boldsymbol{v}))$$;
   (b) Draw $$X_n\mid z_n\sim p(x_n\mid \eta^*_{z_n})$$

Restrict the DP mixtures that the observable data are drawn from an exponential family distribution, and where the base distribution for the DP is the corresponding conjugate prior.

The distribution of $$X_n$$ conditional on $$Z_n$$ and $${\eta^*_1,\eta^*_2,\ldots}$$ is:

$$
p(x_n\mid z_n,\eta^*_1,\eta^*_2,\ldots)=\prod_{i=1}^{\infty} \left(h(x_n) \text{exp} \left\{ {\eta^* _i}^T x_n-a(\eta^*_i) \right\} \right)^{\mathbf{1}\lbrack z_n=i \rbrack}
$$

where $$a(\eta^*_i)$$ is the appropriate cumulant function and it is assumed for simplicity that $$x$$ is the sufficient statistic for the natural parameter $$\eta$$.

The vector of sufficient statistics of the corresponding conjugate family is $$({\eta^* _i}^T, -a(\eta^*_i) )^T$$. The base distribution is:

$$
p(\eta^*\mid \lambda) = h(\eta^*) \text{exp}\left\{\lambda_1^T \eta^* + \lambda_2 (-a(\eta^*))-a(\lambda)\right\}
$$

where the hyperparameter $$\lambda$$ is decomposed, such that $$\lambda_1$$ contains the first $$\dim(\eta^*)$$ components and $$\lambda_2$$ is a scalar.

## Inference 

Citation <d-cite key="blei2017variational"></d-cite>

### Gibbs sampling

Review of the collapsed Gibbs sampler and blocked Gibbs sampler for DP mixtures. Blocked Gibbs sampler outshines collapsed Gibbs sampler when $$G_0$$ is not conjugate.

#### Collapesd Gibbs sampling

The `collapsed Gibbs sampler` for a DP mixture with conjugate base distribution integrates out the random measure $$G$$ and distinct parameter values $$\{\eta^{*}_{1},\ldots,\eta^{*}_{\lvert \boldsymbol{c} \rvert }\}$$. The Markov chain is thus defined only on the latent partition $$\boldsymbol{c} = \{c_1,...,c_ {N}\}$$, where $$\lvert\boldsymbol{c}\rvert$$ denote the number of cells in the partition. The algorithm iteratively samples each assignment variable $$C_n$$, for $$n\in \{1,\ldots,N\}$$, conditional on the other cells in the partition, $$\boldsymbol{c_{-n}}$$. The assignment $$C_n$$ can be one of $$\lvert \boldsymbol{c_{-n}}\rvert +1$$ values: either the $$n$$th data point is in a cell with other data points, or in a cell by itself. 

Exchangeability implies that $$C_n$$ has the following multinomial distribution:

$$
p(c_n=k\mid \boldsymbol{x},\boldsymbol{c}_ {-n},\lambda,\alpha)\propto p(x_n\mid \boldsymbol{x}_ {-n}, \boldsymbol{c}_ {-n}, c_n=k, \lambda)p(c_n=k\mid \boldsymbol{c}_ {-n}, \alpha)
$$

The first term is a ratio of normalizing constants of the posterior distribution of the $$k$$th parameter, one including and one excluding the $$n$$th data point:

$$
p(x_n\mid \boldsymbol{x}_ {-n}, \boldsymbol{c}_ {-n}, c_n=k, \lambda)=\frac{\text{exp}\lbrace a(\lambda_1+\sum_{m\neq n} \mathbf{1} \lbrack c_m =k \rbrack x_m +x_n, \lambda_2 +\sum_{m\neq n} \mathbf{1} \lbrack c_m =k \rbrack +1)\rbrace}{\text{exp}\lbrace a(\lambda_1+\sum_{m\neq n} \mathbf{1} \lbrack c_m =k \rbrack x_m, \lambda_2 +\sum_{m\neq n} \mathbf{1} \lbrack c_m =k \rbrack)\rbrace}
$$

The second term is given by the Polya urn scheme:

$$
p(c_n=k\mid \boldsymbol{c}_ {-n}, \alpha) = \begin{cases} \vert\lbrace j:c_{-n,j}=k \rbrace\vert & \text{if } k \text{ is an existing cell in the partition}\\
\alpha & \text{if } k \text{ is a new cell in the partition}
\end{cases}
$$

where $$\vert\lbrace j:c_{-n,j}=k \rbrace\vert$$ denotes the number of data points in the kth cell of the partition $$\boldsymbol{c}_{-n}$$.

After the chain has reached stationary distribution, $$B$$ samples $$\lbrace \boldsymbol{c}_1, \dots, \boldsymbol{c}_B \rbrace$$ are collected to approximate the posterior. The approximate predictive distribution is an average of the predictive distributions across the Monte Carlo samples:

$$
p(x_{N+1}\mid x_1, \ldots, x_N,\alpha,\lambda)=\frac{1}{N} \sum_{b=1}^{B} p(x_{N+1}\mid \boldsymbol{c}_b,\boldsymbol{x},\alpha,\lambda)
$$

For a given sample, that distribution is:

$$
p(x_{N+1}\mid \boldsymbol{c}_ {b},\boldsymbol{x},\alpha,\lambda) = \sum_{k=1}^{\vert \boldsymbol{c}_ b \vert+1} p(c_{N+1}=k \mid \boldsymbol{c}_ b,\alpha)p(x_{N+1}\mid \boldsymbol{c}_ b, \boldsymbol{x}, c_{N+1}=k, \lambda)
$$

#### Blocked Gibbs sampling
Ishwaran and James <d-cite key="ishwaran2001gibbs"></d-cite> developed a `blocked Gibbs sampling` algorithm based on the `stick-breaking` representation. A truncated Dirichlet process (TDP) is defined by setting $$v_{K-1}=1$$ and $$\pi_i(\boldsymbol{v})=0$$ for $$i\geq K$$, and showed that the truncated process closely approximates a true Dirichlet process when the truncation level is chosen large relative to the number of data points.

In the TDP mixture, the state of the Markov chain consists of the beta variables $$\boldsymbol{V}=\{V_1,\ldots,V_{K-1}\}$$, the mixture component parameters $$\boldsymbol{\eta}^*=\{\eta_1^*,\ldots,\eta_K^*\}$$, and the indicator variables $$\boldsymbol{Z}=\{Z_1,\ldots,Z_N\}$$. The blocked Gibbs sampler iterates between the following three steps:

1. For $$n\in\{1,\ldots,N\}$$, independently sample $$Z_n$$ from:

   $$
   p(z_n=k\mid \boldsymbol{v},\boldsymbol{\eta}^*,\boldsymbol{x}) = \pi_{k}(\boldsymbol{v})p(x_n\mid \eta^* _k)
   $$

2. For $$k\in\{1,\ldots,K\}$$, independently sample $$V_k$$ from $$\text{Beta}(\gamma_{k,1},\gamma_{k,2})$$, where:

   $$
   \begin{aligned} 
   \gamma_{k,1} & = 1 + \sum_{n=1}^{N} \mathbf{1}\lbrack z_n=k \rbrack \\
   \gamma_{k,2} & = \alpha + \sum_{i=k+1}^{K} \sum_{n=1}^{N} \mathbf{1}\lbrack z_n=i \rbrack 
   \end{aligned}
   $$

3. For $$k\in\{1,\ldots,K\}$$, independently sample $$\eta_k^*$$ from $$p(\eta^ *_k \mid \tau_k)$$. This distribution is in the same family as the base distribution, with parameters:

   $$
   \begin{aligned} 
   \tau_{k,1} & = \lambda_1 + \sum_{i\neq n} \mathbf{1}\lbrack z_i=k \rbrack x_i\\
   \tau_{k,2} & = \lambda_2 + \sum_{i\neq n} \mathbf{1}\lbrack z_i=k \rbrack 
   \end{aligned}
   $$

   where as before, the hyperparameter $$\lambda$$ of the conjugate exponential family is decomposeed, such that $$\lambda_1$$ contains the first $$\dim(\eta^*)$$ components and $$\lambda_2$$ is a scalar.
   
After the chain has reached its stationary distribution, $$B$$ samples are collected and an approximate predictive distribution can be constructed. For a particular sample:

$$
p(x_{N+1}\mid \boldsymbol{z},\boldsymbol{x},\alpha,\lambda) = \sum_{k=1}^{K} \mathrm{E}\lbrack \pi_{i}(\boldsymbol{V} \mid \gamma_1,\ldots,\gamma_k) \rbrack p(x_{N+1}\mid \tau_k)
$$

### Variational inference

Consider a model with hyperparameters $$\theta$$, latent variables $$\boldsymbol{W}=\lbrace W_1,\ldots,W_M\rbrace$$, and observations $$\boldsymbol{x}=\lbrace x_1,\ldots,x_N\rbrace$$. The posterior distribution of the latent variables $$p(\boldsymbol{w}\mid\boldsymbol{x},\theta)=\frac{p(\boldsymbol{x},\boldsymbol{w}\mid \theta)}{p(\boldsymbol{x}\mid \theta)}=\text{exp}\lbrace\log p(\boldsymbol{x},\boldsymbol{w}\mid \theta)-\log p(\boldsymbol{x}\mid \theta)\rbrace$$ is difficult to compute, because the latent variables become dependent when conditioning on observed data, then $$\log p(\boldsymbol{x}\mid \theta)=\log \int p(\boldsymbol{x},\boldsymbol{w}\mid \theta)d\boldsymbol{w}$$ is hard to compute.

A class of variational methods known as `mean-field methods` are based on optimizing Kullback-Leibler (KL) divergence with respect to a so-called variational distribution. Let $$q_{\nu}(\boldsymbol{w})$$ be a family of distributions indexed by a variational parameter $$\nu$$, the aim is to minimize the KL divergence between $$q_{\nu}(\boldsymbol{w})$$ and $$p(\boldsymbol{w}\mid\boldsymbol{x},\theta)$$:

$$
\mathbf{D}(q_{\nu}(\boldsymbol{w}) \Vert p(\boldsymbol{w}\mid\boldsymbol{x},\theta)) = \mathbf{E}_ {q} \lbrack \log q_{\nu}(\boldsymbol{W}) \rbrack - \mathbf{E}_ {q} \lbrack \log p(\boldsymbol{W}\mid\boldsymbol{x},\theta) \rbrack + \log p(\boldsymbol{x}\mid \theta)
$$

As the marginal probability does not depend on the variational parameters, it can be ignored in the optimization. To minimize the KL divergence can be cast alternatively as to compute the maximization of a lower bound on the log marginal likelihood:

$$
\log p(\boldsymbol{x}\mid \theta) \geq \mathbf{E}_ {q} \lbrack \log p(\boldsymbol{W}\mid\boldsymbol{x},\theta) \rbrack - \mathbf{E}_ {q} \lbrack \log q_{\nu}(\boldsymbol{W}) \rbrack 
$$

To constructe the family $$q_{\nu}(\boldsymbol{w})$$, it is in need to break some of dependencies between latent variables which make the true posterior difficult to compute.

#### Mean field variational inference in exponential families

For each latent variable, assume that the conditional distribution is a member of the exponential family:

$$
p(w_{i}\mid \boldsymbol{w}_ {-i},\boldsymbol{x},\theta)=h(w_{i})\text{exp} \lbrace {g_{i}(\boldsymbol{w}_ {-i},\boldsymbol{x},\theta)}^T w_{i} - a(g_{i}(\boldsymbol{w}_ {-i},\boldsymbol{x},\theta)) \rbrace
$$

where $$g_{i}(\boldsymbol{w}_ {-i},\boldsymbol{x},\theta)$$ is the natural parameter for $${w}_{i}$$ when conditioning on the remaining latent variables and the observations.

Consider the following family of distributions as mean field variational approximations:

$$
q_{\boldsymbol{\nu}}(\boldsymbol{w})=\prod_{i=1}^{M} q_{\nu_i}(w_i) =\prod_{i=1}^{M} \text{exp} \lbrace \nu_{i}^T w_{i} - a(w_{i}) \rbrace
$$

where $$\boldsymbol{\nu}=\lbrace \nu_1,\ldots, \nu_M \rbrace$$ are variational parameters and each distribution is in the exponential family. Then it is shown that the optimization of KL divergence with respect to a single variational parameter $$\nu_{i}$$ is achieved by computing the following expectation:

$$
\nu_{i} = \mathbf{E}_ {q} \lbrack g_{i}(\boldsymbol{w}_ {-i},\boldsymbol{x},\theta) \rbrack
$$

In a coordinate ascent algorithm, the bound with respect to each $$\nu_i$$ is iteratively maximized, holding the other variational parameters fixed.

Using the chain rule, the bound can be rewriten:

$$
\log p(\boldsymbol{x}\mid \theta) \geq \log p(\boldsymbol{x}\mid \theta) + \sum_{m=1}^{M} \mathbf{E}_ {q} \lbrack \log p(W_m \mid W_1,\ldots, W_{m-1} \boldsymbol{x},\theta) \rbrack - \sum_{m=1}^{M} \mathbf{E}_ {q} \lbrack \log q_{\nu_m}(W_m) \rbrack
$$

To optimize with respect to $$\nu_{i}$$, the part depend on $$\nu_{i}$$ is :

$$
l_{i}= \mathbf{E}_ {q} \lbrack \log p(W_i \mid \boldsymbol{W}_ {-i}, \boldsymbol{x},\theta) \rbrack - \mathbf{E}_ {q} \lbrack \log q_{\nu_i}(W_i) \rbrack
$$

Given that the variational distribution $$q_{\nu_i}(w_i)$$ is in the exponential family:

$$
q_{\nu_i}(w_i)=h(w_i)\text{exp}\lbrace \nu_i^T w_i - a(\nu_i) \rbrace
$$

as in the exponential family $$\mathbf{E}_ {q} \lbrack W_i\rbrack = a'(\nu_i)$$, it is easy to see,

$$
\begin{aligned} 
l_{i} &= \mathbf{E}_ {q} \lbrack \log p(W_i \mid \boldsymbol{W}_ {-i}, \boldsymbol{x},\theta) - \log h(W_i) - \nu_i^T w_i + a(\nu_i) \rbrack \\
&= \mathbf{E}_ {q} \lbrack \log p(W_i \mid \boldsymbol{W}_ {-i}, \boldsymbol{x},\theta) \rbrack - \mathbf{E}_ {q} \lbrack \log h(W_i)  \rbrack - \nu_i^T a'(\nu_i) + a(\nu_i) 
\end{aligned}
$$

The derivative with respect to $$\nu_i$$ is:

$$
\frac{\partial l_{i}}{\partial \nu_i} = \frac{\partial}{\partial \nu_i} \lparen \mathbf{E}_ {q} \lbrack \log p(W_i \mid \boldsymbol{W}_ {-i}, \boldsymbol{x},\theta) \rbrack - \mathbf{E}_ {q} \lbrack \log h(W_i)  \rbrack \rparen - \nu_i^T a''(\nu_i)
$$

let partial derivative equals $$0$$, the optimal $$\nu_i$$ satisfies:

$$
\nu_i = \lbrack a''(\nu_i) \rbrack ^{-1} \lparen \frac{\partial}{\partial \nu_i}\mathbf{E}_ {q} \lbrack \log p(W_i \mid \boldsymbol{W}_ {-i}, \boldsymbol{x},\theta) \rbrack - \frac{\partial}{\partial \nu_i}\mathbf{E}_ {q} \lbrack \log h(W_i)  \rbrack \rparen
$$

as we assumed that the conditional distribution is a member of the exponential family:

$$
p(w_{i}\mid \boldsymbol{w}_ {-i},\boldsymbol{x},\theta)=h(w_{i})\text{exp} \lbrace {g_{i}(\boldsymbol{w}_ {-i},\boldsymbol{x},\theta)}^T w_{i} - a(g_{i}(\boldsymbol{w}_ {-i},\boldsymbol{x},\theta)) \rbrace
$$

where $$g_{i}(\boldsymbol{w}_ {-i},\boldsymbol{x},\theta)$$ is the natural parameter for $$w_{i}$$ when conditioning on the remaining latent variables and the observations. We have the expected log probability of $$W_i$$ and its first derivative:

$$
\begin{aligned} 
\mathbf{E}_ {q} \lbrack \log p(W_i \mid \boldsymbol{W}_ {-i}, \boldsymbol{x},\theta) \rbrack & = \mathbf{E}_ {q} \lbrack \log h(W_{i})\rbrack + \mathbf{E}_ {q} \lbrack g_{i}(\boldsymbol{W}_ {-i},\boldsymbol{x},\theta)}^T \rbrack a'(\nu_i) - \mathbf{E}_ {q} \lbrack a(g_{i}(\boldsymbol{W}_ {-i},\boldsymbol{x},\theta)) \rbrack \\
\frac{\partial}{\partial \nu_i} \mathbf{E}_ {q} \lbrack \log p(W_i \mid \boldsymbol{W}_ {-i}, \boldsymbol{x},\theta) \rbrack & = \frac{\partial}{\partial \nu_i} \mathbf{E}_ {q} \lbrack \log h(W_{i})\rbrack + \mathbf{E}_ {q} \lbrack g_{i}(\boldsymbol{W}_ {-i},\boldsymbol{x},\theta)}^T \rbrack a''(\nu_i)
\end{aligned}
$$

#### Coordinate ascent algorithm for DP mixtures 

## Implementation 
Citation <d-cite key="blei2017variational"></d-cite> <d-cite key="blei2006variational"></d-cite>

---
