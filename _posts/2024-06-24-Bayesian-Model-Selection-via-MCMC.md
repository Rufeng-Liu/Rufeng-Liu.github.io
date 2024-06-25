---
layout: distill
title: Bayesian Model Selection via MCMC
description: Bayesian model selection
tags: Bayesian model/variable selection
giscus_comments: true
date: 2024-06-24
featured: true

authors:
  - name: Rufeng Liu
    affiliations:
      name: FSU

bibliography: 1995-Carlin.bib

toc:
  - name: Method
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Implementaton

## Method

This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
p(y \mid M=j)=\int f(y\mid \theta,M=j)p(\theta\mid M=j)d\theta=\int f(y\mid \theta_{j},M=j)p(\theta_{j}\mid M=j)d\theta_{j}
$$
  
