---
layout: distill
title: Sampling Weights
description: Sampling weights
tags: sample methods
giscus_comments: false
date: 2024-07-16
featured: true

bibliography: 2024-07-16-sampling-weights.bib

toc:
  - name: Sampling Weights
    subsections:
      - name: For non-response
      - name: For non-coverage
  - name: NHANES 
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
    
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

## Sampling Weights 
Construct a base weight for each sampled unit, the reciprocal of its probability of selection into the sample, to correct for their unequal probabilities of selection, e.g, $$w_i=\frac{1}{p_i}$$.

For multi-stage designs, the base weights must reflect the probabilities of selection at each stage, e.g, $$p_{ij}=p_i\times p_{j(i)}$$. Then, base weight $$w_{ij,b}=\frac{1}{p_{ij}}$$.

The weight for non-response $$w_{ij,nr}$$, and the weight for non-coverage is $$w_{ij,nc}$$, will be explained later. The overall weight is $$w_{ij}=w_{ij,b}\times w_{ij,nr} \times w_{ij,nc}$$.

Some units have duplicates on the frame, then increased probability of selection of such units can be compensated. Suppose the $$i$$-th sampled unit has a probability of selections denoted by $$p_{i1},\ldots,p_{ik}$$,  the adjusted probability of selection of the sampled unit is $$p_i=1-(1-p_{i1})(1-p_{i2})\cdots(1-p_{ik})$$, then $$w_i=\frac{1}{p_i}$$.

### For non-response

In a survey, participants may provide no data at all (`total non-response`) or only partial data (`item non-response`). If there are any systematic differences between the respondents and non-respondents, then naive estimates based solely on the respondents will be biased. The size of the non-response bias for a sample mean, for instance, is a function of two factors: (1) the proportion of the population that does not respond, (2) the size of the difference in population means between respondent and nonrespondent groups. Reducing the bias due to non-response therefore requires that either the non-response rate be small, or that there are small differences between responding and non-responding households and persons. 

For `total non-response`, there are three basic procedures for compensation:
1.
2.
3.

### For non-coverage

## NHANES

