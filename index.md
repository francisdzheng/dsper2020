---
layout: home
title: Home
nav_order: 0
description: >-
    Data Science for Practical Economic Research
latex: true
---

# {{ site.description }}
{: .mb-2 }
{{ site.semester }}
{: .fs-6 .fw-300 }

{% if site.announcements %}
{{ site.announcements.last }}
[Announcements &#124; お知らせ]({{ site.baseurl }}{% link announcements.md %}){: .btn .btn-outline .fs-3 }
{% endif %}

## Important information

This website serves as a supplement to Professor Konstantin Kucheryavyy's Data Science for Practical Economic Research course.

- Please check [Piazza](https://piazza.com/class/k8pcxfiwkxf2ec) regularly for announcements. 
- Lectures are held every Tuesday, 14:55 - 16:40 (4限). The Zoom link can be found on UTAS. 

## Goals

In this course, we will study the fundamentals of machine learning, with a focus on economic applications. Topics include:
- Supervised machine learning: under-fitting and over-fitting, regularization, cross-validation, data augmentation
- Unsupervised machine learning: clustering, factor analysis, principal component analysis, independent component analysis
- Semi-supervised learning
