---
layout: distill
title: High-Dimensional Statistics Homework Solution
description: Back up for my TA work
tags: High-Dimensiona Sparsity
giscus_comments: false
date: 2024-10-01
featured: true

bibliography: 

toc:
  - name: HW1 
    subsections:
       - name: Problem 1
       - name: Problem 2
       - name: Problem 3
       - name: Problem 4

  - name: HW2 
    subsections:
       - name: Problem 1

  - name: Midterm
    subsections:
       - name: Problem 1
       - name: Problem 2



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

## HW1 


  
--- 


## HW2 

### Problem 1

For a dataset $\{ y_i, \boldsymbol{X}_ i \}_{i=1}^N$ with $y_i \in \mathbb{R}$, $\boldsymbol{X}_i \in \mathbb{R}^p$, define $\boldsymbol{y}=(y_1,\ldots,y_N)$ and $\boldsymbol{X}=(\boldsymbol{X}_1^T,\ldots,\boldsymbol{X}_N^T)^T\in \mathbb{R}^{N\times p}$. Consider the LASSO estimator

$$
\hat{\boldsymbol{\beta}} = \arg \min_{\boldsymbol{\beta} \in \mathbb{R}^p} \frac{1}{2N} \sum_{i=1}^{N} \omega_i (y_i - \boldsymbol{X}_i^T \boldsymbol{\beta})^2 + \lambda \|\boldsymbol{\beta}\|_1,
$$

where $\omega_i$ are positive constants and $\sum_{i=1}^N \omega_i = 1$.

#### 1. Write down a `proximal gradient descent algorithm` for solving this problem

`Algorithm`

The proximal gradient descent algorithm solves the problem by iterating between gradient descent and applying the proximal operator for the $\ell_1$-norm regularization term. 

1. **Initialize** $\beta^{0}$.
2. **Repeat** until convergence:

   For function $f(\boldsymbol{\beta})= \frac{1}{2N} \sum_{i=1}^{N} \omega_i (y_i - \boldsymbol{X}_i^T \boldsymbol{\beta})^2 + \lambda \|\boldsymbol{\beta}\|_1$, we can write $f(\boldsymbol{\beta})= g(\boldsymbol{\beta}) + h(\boldsymbol{\beta})$, where $g(\boldsymbol{\beta})=\frac{1}{2N} \sum_{i=1}^{N} \omega_i (y_i - \boldsymbol{X}_i^T \boldsymbol{\beta})^2$ is convex and differentiable, $h(\boldsymbol{\beta}) = \lambda \|\boldsymbol{\beta}\|_1$ is convex.

   We make quadratic approximation to $g$, for iteration $t+1$, the `proximal gradient` update is 
   
   $$
   \boldsymbol{\beta}^{t+1} = \text{prox}_{h,s^t}\left(\boldsymbol{\beta}^{t}-s^t \cdot \nabla g(\boldsymbol{\beta}^{t})\right)
   $$
   where 
   
   $$
   \nabla g(\boldsymbol{\beta}^{t}) = -\frac{1}{N} \sum_{i=1}^{N} \omega_i X_i (y_i - X_i^T \boldsymbol{\beta}^{t})
   $$
   
   
   then, the `proximal gradient` update takes the form:
   $$
   \boldsymbol{\beta}^{t+1} = S_{s^t \lambda}\left(\beta^{t} + s^t \frac{1}{N} \sum_{i=1}^{N} \omega_i X_i (y_i - X_i^T \boldsymbol{\beta}^{t})\right)
   $$
   where $s^t > 0$ is the step size and $S_{\lambda}(z)$ is the soft-thresholding operator:
   
   $$
   S_{\lambda}(z) = \text{sgn}(z) \max\{|z| - \lambda, 0\}.
   $$

#### 2. Write down a `coordinate descent algorithm` for solving this problem

`Algorithm`

1. **Initialize** $\beta^{0}$.
2. **Repeat** until convergence:

   For iteration $t+1$, each coordinate $j \in \{1, \dots, p\}$, the loss function can be expressed below w.r.t $\beta_j$,
   
   $$
   \begin{eqnarray*}
   L(\boldsymbol{\beta}) 
   &=& \frac{1}{2N} \sum_{i=1}^{N} \omega_i (y_i - \boldsymbol{X}_i^T \boldsymbol{\beta})^2 + \lambda \|\boldsymbol{\beta}\|_1 \\
   &=& \frac{1}{2N} \sum_{i=1}^{N} \omega_i (y_i - \sum_{k \neq j} x_{ik} \beta_k - x_{ij} \beta_j)^2 + \lambda \sum_{k \neq j} |\beta_{k}| + \lambda |\beta_{j}|
   \end{eqnarray*}
   $$
   Find the minimizer w.r.t $\beta_j$ with partical derivative,

   $$
   \begin{eqnarray*}
   \frac{\partial L(\boldsymbol{\beta})}{\partial \beta_j} &=& -\frac{1}{N} \sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k - x_{ij} \beta_j)+  \text{sgn}(\beta_j) \lambda \\
   &=& -\frac{1}{N} \sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k) +  \frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2 \beta_j +  \text{sgn}(\beta_j) \lambda
   \end{eqnarray*}
   $$
   Let $f(\beta_j)= L(\boldsymbol{\beta})$, consider 3 cases.

   (i). $\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t}) > \lambda$.

   When $\beta_j > 0$, we have

   $$
   f'(\beta_j) = -\frac{1}{N} \sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k) +  \frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2 \beta_j + \lambda
   $$

   Let $f'(\beta_j) = 0$, we have $\hat{\beta}_j = \frac{\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t})-\lambda}{\frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2} > 0$, and $f''(\beta_j) > 0$.

   When $\beta_j < 0$, we have

   $$
   f'(\beta_j) = -\frac{1}{N} \sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k) +  \frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2 \beta_j - \lambda < 0,
   $$

   notice that $f(\beta_j)$ is decreasing.

   When $\beta_j = 0$, consider the interval $(-\epsilon, \epsilon)$, where $0 < \epsilon < \hat{\beta}_j = \frac{\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t})-\lambda}{\frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2}$. From the former two conditions, we notice that $f(\beta_j)$ is decreasing on $(-\epsilon, 0)$ and $(0, \epsilon)$, since $f(\beta_j)$ is continuous, we conclude that $\beta_j = 0$ can not minimize $f(\beta_j)$.

   We now conclude, given $\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t}) > \lambda$, $\hat{\beta}_j = \frac{\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t})-\lambda}{\frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2} > 0$ minimizes $f(\beta_j)$.

   (ii). $\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t}) < -\lambda$.

   When $\beta_j > 0$, we have

   $$
   f'(\beta_j) = -\frac{1}{N} \sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k) +  \frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2 \beta_j + \lambda > 0.
   $$

   notice that $f(\beta_j)$ is increasing.
   
   When $\beta_j < 0$, we have

   $$
   f'(\beta_j) = -\frac{1}{N} \sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k) +  \frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2 \beta_j - \lambda.
   $$

   Let $f'({\beta}_j) = 0$, we have $\hat{\beta}_j = \frac{\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t})+\lambda}{\frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2} < 0$, and $f''(\beta_j) > 0$.

   When $\beta_j = 0$, consider the interval $(-\epsilon, \epsilon)$, where $0 > -\epsilon > \hat{\beta}_j = \frac{\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t})+\lambda}{\frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2}$. From the former two conditions, we notice that $f(\beta_j)$ is increasing on $(-\epsilon, 0)$ and $(0, \epsilon)$, since $f(\beta_j)$ is continuous, we conclude that $\beta_j = 0$ can not minimize $f(\beta_j)$.

   We now conclude, given $\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t}) < -\lambda$, $\hat{\beta}_j = \frac{\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t})+\lambda}{\frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2} < 0$ minimizes $f(\beta_j)$.

   (iii). $|\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t}) | \leq \lambda$.

   When $\beta_j > 0$, we have

   $$
   f'(\beta_j) = -\frac{1}{N} \sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k) +  \frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2 \beta_j + \lambda > 0.
   $$

   notice that $f(\beta_j)$ is increasing.

   When $\beta_j < 0$, we have

   $$
   f'(\beta_j) = -\frac{1}{N} \sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k) +  \frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2 \beta_j - \lambda < 0.
   $$

   notice that $f(\beta_j)$ is decreasing.

   Since $f(\beta_j)$ is continuous, we now conclude that given $|\frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t}) | \leq \lambda$, $\hat{\beta}_j = 0$ minimizes $f(\beta_j)$.
   
   
   In conclusion, the $t+1$ update of $\beta_j$ is, 
   
   $$
   \beta_j^{t+1} = \frac{S_{\lambda} \left( \frac{1}{N}\sum_{i=1}^{N} \omega_i x_{ij} (y_i - \sum_{k \neq j} x_{ik} \beta_k^{t}) \right)}{\frac{1}{N}\sum_{i=1}^{N}\omega_i x_{ij}^2}
   $$
---
