---
layout: distill
title: Jackknife, Bootstrap and Bayesian Bootstrap
description: Bootstrap
tags: Resampling
giscus_comments: false
date: 2024-07-06
featured: true

bibliography: 2024-07-06-Jackknife-Bootstrap-BayesianBootstrap.bib

toc:
  - name: Jackknife
  - name: Bootstrap
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Bayesian Bootstrap

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
Ways of resampling, comparison of Jackknife, Bootstrap and Bayesian Bootstrap.

## Jackknife <d-cite key="miller1974jackknife"></d-cite>
Given a sample of size $$n$$, a [jackknife estimator](https://en.wikipedia.org/wiki/Jackknife_resampling) can be built by aggregating the parameter estimates from each subsample of size $$(n-1)$$ obtained by omitting one observation.

Useful for bias and variance estimation, a linear approximation of the bootstrap.

## Bootstrap <d-cite key="efron1992bootstrap"></d-cite>

A generalization of the jackknife, 

## Bayesian Bootstrap <d-cite key="rubin1981bayesian"></d-cite>

---
