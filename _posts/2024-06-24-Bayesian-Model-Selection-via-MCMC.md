---
layout: distill
title: Bayesian Model Selection via MCMC
description: Bayesian model selection
tags: Bayesian-model/variable-selection
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
  - name: Implementation 
    subsections:
       - name: Example
       - name: In project
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

Citation <d-cite key="carlin1995bayesian"></d-cite>. 

Choose between $$K$$ models with corresponding parameter vector $$\boldsymbol{\theta}_j$$, $$j=1,...K$$. 

Let $$M$$ be an integer-valued parameter that indexes the model, for model $$j$$, we have a likelihood $$f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,M=j)$$ and a prior $$p(\boldsymbol{\theta}_j\mid M=j)$$. Given $$M=j$$, $$\boldsymbol{y}$$ is independent of $$\{\boldsymbol{\theta_{i\neq j}}\}$$. Assume that given the indicator $$M$$, $$\boldsymbol{\theta}_j$$ are independent of each other, we can complete the Bayesian model specification by choosing proper `pseudopriors` $$p(\boldsymbol{\theta}_j\mid M\neq j)$$, which is a conveniently chosen linking density. Reason is shown below, let $$\boldsymbol{\theta}=\{\boldsymbol{\theta}_1,\ldots,\boldsymbol{\theta}_K\}$$,
$$
p(\boldsymbol{y} \mid M=j)=\int f(\boldsymbol{y}\mid \boldsymbol{\theta},M=j)p(\boldsymbol{\theta}\mid M=j)d\boldsymbol{\theta}=\int f(\boldsymbol{y}\mid \boldsymbol{\theta}_{j},M=j)p(\boldsymbol{\theta}_{j}\mid M=j)d\boldsymbol{\theta}_j
$$
Given prior model probabilities $$\pi_{j}\equiv P(M=j)$$ such that $$\sum_{j=1}^{K}\pi_{j}=1$$, when $$M=j$$, the joint distribution of $$\boldsymbol{y}$$ and $$\boldsymbol{\theta}$$ is 

$$
\begin{aligned} 
p(\boldsymbol{y},\boldsymbol{\theta},M=j) & = f(\boldsymbol{y}\mid \boldsymbol{\theta},m=j)p(\boldsymbol{\theta}\mid M=j)p(M=j) \\
 & = f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,m=j)p(\boldsymbol{\theta}\mid M=j)p(M=j) \\
 & = f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,m=j) \left\{\prod_{i=1}^{K} p(\boldsymbol{\theta}_i\mid M=j)\right\} p(M=j)\\
 & = f(\boldsymbol{y}\mid \boldsymbol{\theta}_j,m=j) \left\{\prod_{i=1}^{K} p(\boldsymbol{\theta}_i\mid M=j)\right\} \pi_{j}
\end{aligned}
$$

To implement Gibbs sampler the full conditional distributions of each $$\boldsymbol{\theta}_j$$ and $$M$$.
For $$\boldsymbol{\theta}_j$$, when $$M=j$$, we generate from the usual model $$j$$ full conditional; when $$M\neq j$$, we generate from the linking function (`pseudoprior`).

$$
p(\boldsymbol{\theta}_j \mid \boldsymbol{\theta}_{i\neq j},M,\boldsymbol{y}) \propto \begin{cases}
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
B_{ji}=\frac{p(\boldsymbol{y}\mid M=j)}{p(\boldsymbol{y}\mid M=i)}
$$

between any two of the models.

--- 
## Implementation 

Citation <d-cite key="carlin1995bayesian"></d-cite>.

Poor choices of the linking density (`pseudopriors`) $$p(\boldsymbol{\theta}_j\mid M\neq j)$$ will make jumps between models extremely unlikely, so that the convergence of the Gibbs sampling may trapped to one model, which might not be the true one in fact. Good choices will produce $$\boldsymbol{\theta}_j^{(g)}$$-values that are consistent with the data, so that $$p(M=j\mid \boldsymbol{\theta},\boldsymbol{y})$$ will still be reasonably large at the next $$M$$ update step. 

If for a particular data set one of the $$p(M=j\mid \boldsymbol{y})$$ is extremely large, the $$\pi_j$$ may be adjusted to correct the imbalance during the early stage of the algorithm, so that the final value of $$B_{ji}$$ reflect the true odds in favour of $$M=j$$ suggested by the data.

Key point: Use the data to help to select the `pseudopriors` but `not` the prior, match the `pseudopriors` as nearly as possible to the true model-specific posteriors. 

### Example 

Citation <d-cite key="jauch2021mixture"></d-cite>.

Let $I=(a,b)$ with $a,b\in \mathbb{R}\cup \lbrace -\infty,\infty \rbrace$. Suppose $f$ and $g$ are density functions with $\int_{I}f(x)dx=1$, $\int_{I}g(x)dx=1$, and $g>0$ on $I$, with $F$ and $G$ be the corresponding distribution functions. For each $s\in I$, let $g^s$ denote the truncated density function with $g^s(x)=g(x)\mathbf{1}_ {(-\infty,s]}(x)/G(s)$ for $x\in I$. We say that $F$ is smaller than $G$ in the likelihood ratio order, denote $F\leq_{LR} G$, if $f/g$ is monotone non-increasing. It is shown that, $f/g$ is monotone non-increasing and locally absolutely continuous on $I$ if and only if there exists $\theta \in \lbrack 0,1 \rbrack$ and an absolutely continuous distribution function $U$ with density $u$, where $U(a+)=0$ and $U(b-)=1$ such that for $x\in I$, 

$$
f(x)=\theta g(x) + (1-\theta) \int_{a}^{b} g^{s}(x)u(s)ds
$$

When this mixture representation exists, $\theta=\lim_{x\uparrow b} f(x)/g(x)$, if $\theta\in[0,1)$, $U$ is uniquely determined with $U(x)=\frac{G(x)}{1-\theta}\lbrace \frac{F(x)}{G(x)}-\frac{f(x)}{g(x)} \rbrace$, $x\in I$. 

If we have $I=\mathbb{R}$, densities $g$ and $u$ can be easily modeled using DP mixtures $\mathrm{DP}(P_0,\alpha)$ together with a Gaussian kernel $\phi_{\mu,\sigma^2}$,

$$
\begin{aligned}
g(\cdot\mid P_1) &= \int_{\mathbb{R}\times \mathbb{R}^{+}} \phi_{\mu,\sigma^2}(\cdot) P_1(d\mu,d\sigma^2), \quad P_1\sim \mathrm{DP}(P_{1,0}, \alpha_1) \\
u(\cdot\mid P_2) &= \int_{\mathbb{R}\times \mathbb{R}^{+}} \phi_{\mu,\sigma^2}(\cdot) P_2(d\mu,d\sigma^2), \quad P_2\sim \mathrm{DP}(P_{2,0}, \alpha_2)
\end{aligned} 
$$

Under the truncated stick-breaking representation of DP, 

$$
P_k(\cdot)=\sum_{j=1}^{N} v_{k,j} \left\lbrace \prod_{l=1}^{j-1} (1-v_{k,l}) \right\rbrace \delta_{\mu_{k,j}, \sigma_{k,j}^2} (\cdot), \quad k\in \lbrace 1,2 \rbrace
$$

where the truncation level $N$ is fixed, $v_{k,j}$ are independent beta random variables for each $j\in \lbrace 1,\ldots, N-1 \rbrace$ with $v_{k,N}=1$, and the atoms $(\mu_{k,j}, \sigma_{k,j}^2)^T$ are independent random vectors distributed according to the base measure $P_{0,k}$. Let $\vec{v}_ k=(v_{k,1}, \ldots, v_{k,N-1})^T$, $\vec{\mu}_ k=(\mu_{k,1}, \ldots, \mu_{k,N})^T$, $\vec{\sigma}_ k ^2=(\sigma_{k,1}^2, \ldots, \sigma_{k,N}^2)^T$, the densities of $G$ and $U$ are

$$
\begin{aligned}
g(x \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2) &= \sum_{j=1}^{N} v_{1,j} \left\lbrace \prod_{l=1}^{j-1} (1-v_{1,l}) \right\rbrace \delta_{\mu_{1,j}, \sigma_{1,j}^2} (x) \\
u(x \mid \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2) &= \sum_{j=1}^{N} v_{2,j} \left\lbrace \prod_{l=1}^{j-1} (1-v_{2,l}) \right\rbrace \delta_{\mu_{2,j}, \sigma_{2,j}^2} (x)
\end{aligned} 
$$

Choosing a spike-and-slab prior that assigns positive probability to the event $\theta=1$ enable us to do model selection, using the posterior probability of $H_0: F=G$, equivalent to $\theta=1$, versus $H_1: F\leq_{LR} G$, equivalent to $\theta\in[0,1)$. Setting $\theta=(1-\gamma)\tilde{\theta}+\gamma$, $H_0: F=G$ and $H_1: F\leq_{LR} G$ can be identified with the events $\gamma=1$ and $\gamma=0$, respectively. The density of $F$ can be expressed as

$$
f(x \mid \gamma, \tilde{\theta}, \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2, \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2) = \theta g(x \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2) + (1-\theta) \int_{-\infty}^{\infty} g^{s}(x \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2)u(s \mid \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2)ds
$$

the joint distribution of the data and parameters is given by

$$
\begin{aligned}
X_i \mid \gamma, \tilde{\theta}, \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2, \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2 & \stackrel{ind}{\sim} f(\cdot \mid \gamma, \tilde{\theta}, \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2, \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2), \quad i\in \lbrace 1,\ldots, n \rbrace \\
Y_i \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2 & \stackrel{ind}{\sim} g(\cdot \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2), \quad i\in \lbrace 1,\ldots, m \rbrace \\
v_{1,j} & \stackrel{ind}{\sim} \mathrm{Beta}(1,\alpha), \quad j\in \lbrace 1,\ldots, N-1 \rbrace \\
(\mu_{1,j}, \sigma_{1,j}^2)^T & \stackrel{ind}{\sim} \mathrm{Normal\space Inv-Gamma}(m,c,a_1,a_2), \quad j\in \lbrace 1,\ldots, N \rbrace \\
\tilde{\theta} \mid \gamma = 0 & \sim \mathrm{Beta}(b_1,b_2) \\
v_{2,j} \mid \gamma = 0 & \stackrel{ind}{\sim} \mathrm{Beta}(1,\alpha), \quad j\in \lbrace 1,\ldots, N-1 \rbrace \\
(\mu_{2,j}, \sigma_{2,j}^2)^T \mid \gamma = 0 & \stackrel{ind}{\sim} \mathrm{Normal\space Inv-Gamma}(m,c,a_1,a_2), \quad j\in \lbrace 1,\ldots, N \rbrace \\
\tilde{\theta} \mid \gamma = 1 & \sim \mathrm{Beta}(\breve{b}_{1},\breve{b}_{2}) \\
v_{2,j} \mid \gamma = 1 & \stackrel{ind}{\sim} \mathrm{Beta}(1,\breve{\alpha}), \quad j\in \lbrace 1,\ldots, N-1 \rbrace \\
(\mu_{2,j}, \sigma_{2,j}^2)^T \mid \gamma = 1 & \stackrel{ind}{\sim} \breve{p}_{2,0}(\cdot), \quad j\in \lbrace 1,\ldots, N \rbrace \\
\gamma & \sim \mathrm{Bernoulli}(p_0)
\end{aligned} 
$$

The density of $\mathrm{Normal\space Inv-Gamma}(m,c,a_1,a_2)$ is 

$$
\pi_{\mathrm{NI}}(\mu,\sigma^2 \mid m,c,a_1,a_2) = \frac{\sqrt{c}}{\sigma\sqrt{2\pi}}\frac{a_2^{a_1}}{\Gamma(a_1)}(\frac{1}{\sigma^2})^{a_1+1} \mathrm{exp}(-\frac{2a_2+c(\mu-m)^2}{2\sigma^2})
$$

The density of $\mathrm{Beta}(b_1,b_2)$ is 

$$
\pi_{\mathrm{Beta}}(v \mid b_1, b_2) = \frac{\Gamma(b_1+b_2)}{\Gamma(b_1)\Gamma(b_2)}v^{b_1-1}(1-v)^{b_2-1}
$$

The priors for $\tilde{\theta}$, $v_{2,j}$, and $(\mu_{2,j}, \sigma_{2,j}^2)^T$ are defined conditionally on $\gamma$ for computational purposes. The priors given $\gamma = 1$ are `pseudopriors`. The full conditional distributions for the slice-within-Gibbs sampler are given, with a dash as shorthand notation, 

1. Update $(\vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2)^T$.
   
   $$
   \begin{aligned}
   \pi(\vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2 \mid \gamma = 0, -) & \propto \left\lbrace \prod_{i=1}^{n} f(X_i \mid \gamma = 0, \tilde{\theta}, \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2, \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2) \right\rbrace \times \left\lbrace \prod_{i=1}^{m} g(Y_i \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2) \right\rbrace \\ & \times \left\lbrace \prod_{j=1}^{N} \pi_{\mathrm{NI}}(\mu_{1,j}, \sigma_{1,j} ^2 \mid m,c,a_1,a_2) \right\rbrace \times \left\lbrace \prod_{j=1}^{N-1} \pi_{\mathrm{Beta}}(v_{1,j} \mid b_1, b_2) \right\rbrace
   \end{aligned}
   $$

   $$
   \begin{aligned}
   \pi(\vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2 \mid \gamma = 1, -) & \propto \left\lbrace \prod_{i=1}^{n} g(X_i \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2) \right\rbrace \times \left\lbrace \prod_{i=1}^{m} g(Y_i \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2) \right\rbrace \\ & \times \left\lbrace \prod_{j=1}^{N} \pi_{\mathrm{NI}}(\mu_{1,j}, \sigma_{1,j} ^2 \mid m,c,a_1,a_2) \right\rbrace \times \left\lbrace \prod_{j=1}^{N-1} \pi_{\mathrm{Beta}}(v_{1,j} \mid b_1, b_2) \right\rbrace
   \end{aligned}
   $$

   $f(X_i \mid \gamma = 0, \tilde{\theta}, \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2, \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2)$ is evaluated using numerical integration.

2. Update $(\vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2)^T$.
   
   (1) If $\gamma = 0$, the conditional density
   
    $$
   \begin{aligned}
   \pi(\vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2 \mid \gamma = 0, -) & \propto \left\lbrace \prod_{i=1}^{n} f(X_i \mid \gamma = 0, \tilde{\theta}, \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2, \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2) \right\rbrace \\ & \times \left\lbrace \prod_{j=1}^{N} \pi_{\mathrm{NI}}(\mu_{2,j}, \sigma_{2,j} ^2 \mid m,c,a_1,a_2) \right\rbrace \times \left\lbrace \prod_{j=1}^{N-1} \pi_{\mathrm{Beta}}(v_{2,j} \mid 1,\breve{\alpha}) \right\rbrace
   \end{aligned}
   $$ 
   
   (2) If $\gamma = 1$, sample

   $$
   \begin{aligned}
   v_{2,j} \mid \gamma = 1, -  & {\sim} \mathrm{Beta}(1,\breve{\alpha}), \quad j\in \lbrace 1,\ldots, N-1 \rbrace \\
(\mu_{2,j}, \sigma_{2,j}^2)^T \mid \gamma = 1, - & {\sim} \breve{p}_{2,0}(\cdot), \quad j\in \lbrace 1,\ldots, N \rbrace
   \end{aligned}
   $$ 

3. Update $\tilde{\theta}$. For each $i \in \lbrace 1,\ldots,n \rbrace$, a latent variable $R_i$ that associates $X_i$ with one of the two components in the mixture representation is introduced,

   $$
   \begin{aligned}
   X_i \mid R_i = 1, \gamma \in \lbrace 0,1 \rbrace, - & \sim g(\cdot \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2) \\
   X_i \mid R_i = 0, \gamma = 0, - & \sim \int_{-\infty}^{\infty} g^{s}(\cdot \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2)u(s \mid \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2)ds \\
   R_i \mid \gamma = 0 & \sim \mathrm{Bernoulli}(\tilde{\theta}) \\
   \tilde{\theta} \mid \gamma = 0 & \sim \mathrm{Beta}(b_1,b_2) \\
   R_i \mid \gamma = 1 & \sim \delta_1(\cdot) \\
   \tilde{\theta} \mid \gamma = 1 & \sim \mathrm{Beta}(\breve{b}_1,\breve{b}_2)
   \end{aligned} 
   $$

   As $\mathrm{Pr}(R_i = 0, \gamma = 1) = 0$, $R_i$ is updated with

   $$
   \begin{aligned}
   \mathrm{Pr}(R_i = 1 \mid \gamma = 0, -) &= \frac{\tilde{\theta}g(X_i \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2)}{\tilde{\theta}g(X_i \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2) + (1- 
   \tilde{\theta})\int_{-\infty}^{\infty} g^{s}(X_i \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2)u(s \mid \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2)ds}\\
   \mathrm{Pr}(R_i = 0 \mid \gamma = 0, -) &= 1-\mathrm{Pr}(R_i = 1 \mid \gamma = 0, -)\\
   \mathrm{Pr}(R_i = 1 \mid \gamma = 1, -) &= 1
   \end{aligned} 
   $$

   and $\tilde{\theta}$ is updated with

   $$
   \begin{aligned}
   \tilde{\theta} \mid \gamma = 0, - & \sim \mathrm{Beta}(b_1+\sum_{i=1}^{n}R_i, b_2+n-\sum_{i=1}^{n}R_i) \\
   \tilde{\theta} \mid \gamma = 1, - & \sim \mathrm{Beta}(\breve{b}_1,\breve{b}_2)
   \end{aligned}
   $$

4. Update $\gamma$. The full conditional distribution for $\gamma$ is Bernoulli with
   
   $$
   \begin{aligned}
   \mathrm{Pr}(\gamma = 0 \mid -) & = \frac{(1-p_0) \prod_{i=1}^{n} f(X_i \mid \gamma = 0, \tilde{\theta}, \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2, \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2)}{\mathnormal{Z}_ {\gamma}}\\
   \mathrm{Pr}(\gamma = 1 \mid -) & = \frac{p_0 \prod_{i=1}^{n} g(X_i \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2)}{\mathnormal{Z}_ {\gamma}}\\
   \end{aligned}
   $$

   where 

   $$
   \mathnormal{Z}_ {\gamma} = p_0 \prod_{i=1}^{n} g(X_i \mid \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2) + (1-p_0) \prod_{i=1}^{n} f(X_i \mid \gamma = 0, \tilde{\theta}, \vec{v}_ 1, \vec{\mu}_ 1, \vec{\sigma}_ 1 ^2, 
   \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2)
   $$


`pseudopriors` (conditional on $\gamma = 1$) $\mathrm{Beta}(\breve{b}_ 1, \breve{b}_ 2)$, $\mathrm{Beta}(1,\breve{\alpha})$ and $\breve{p}_ {2,0}(\cdot)$ are defined to resemble the posterior distribution of $\tilde{\theta}$, $\vec{v}_ 2$, and $(\vec{\mu}_ 2, \vec{\sigma}_ 2 ^2)^T$ conditional on $\gamma = 0$. First, fix $\gamma = 0$ and run first three steps in a sampler for $Q$ iterations, the Markov chain output $(\tilde{\theta}_ {(1)}, \vec{v}_ {2,(1)}, \vec{\mu}_ {2,(1)}, \vec{\sigma}_ {2,(1)} ^2)^T, \ldots, (\tilde{\theta}_ {(Q)}, \vec{v}_ {2,(Q)}, \vec{\mu}_ {2,(Q)}, \vec{\sigma}_ {2,(Q)} ^2)^T$ provides an approximation of the posterior distribution of $\tilde{\theta}$, $\vec{v}_ 2$, and $(\vec{\mu}_ 2, \vec{\sigma}_ 2 ^2)^T$ conditional on $\gamma = 0$. Then we use the Markvo chain output to determine the `pseudopriors`. 

For the `pseudopriors` of $\tilde{\theta}$,

$$
(\breve{b}_ 1, \breve{b}_ 2) = \arg\max_ {(a,b)} \prod_{q=1}^{Q} \pi_{\mathrm{Beta}}(\tilde{\theta}_{(q)} \mid a, b)
$$

For the `pseudopriors` of $\vec{v}_ 2$,

$$
\breve{\alpha} = \frac{1}{Q}\sum_{q=1}^{Q} \arg\max _ {a} \prod_{j=1}^{N-1} \pi_{\mathrm{Beta}}(v_{2,j(q)} \mid 1, a)
$$

For the `pseudopriors` of $(\vec{\mu}_ 2, \vec{\sigma}_ 2 ^2)^T$, $\breve{p}_ {2,0}(\cdot)$ is set as a kernel density estimate using the function $\it{kde}$ in the $\boldsymbol{R}$ package $\it{ks}$ computed from $(m_1,s_1^2)^T, \ldots, (m_Q,s_Q^2)^T$, where 

$$
(m_q,s_q^2) \sim u_{(q)}(\cdot \mid \vec{v}_ 2, \vec{\mu}_ 2, \vec{\sigma}_ 2 ^2) = \sum_{j=1}^{N} v_{2,j(q)} \left\lbrace \prod_{l=1}^{j-1} (1-v_{2,l(q)}) \right\rbrace \delta_{\mu_{2,j(q)}, \sigma_{2,j(q)}^2} (\cdot)
$$

### In project

To test the correlation between different components in different simplexes is equivalent to do model selection between these two models:

1. Dependent simplexes:

$$
g_1(\cdot\mid k_1,k_2,F)
$$


2. Independent simplexes:

---
