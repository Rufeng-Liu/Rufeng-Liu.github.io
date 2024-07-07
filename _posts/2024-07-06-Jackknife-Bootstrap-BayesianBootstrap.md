---
layout: distill
title: Variational Inference
description: Variational Bayesian
tags: Optimization, Variational
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
Use optimization rather than use sampling. 
First posit a family of densities, then to find a member of that family which is close to the target density. Use exponential family as an example.

## Jackknife <d-cite key="miller1974jackknife"></d-cite>

## Bootstrap <d-cite key="efron1992bootstrap"></d-cite>

## Bayesian Bootstrap <d-cite key="rubin1981bayesian"></d-cite>

---
