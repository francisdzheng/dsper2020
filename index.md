---
layout: home
title: Home
nav_order: 0
description: >-
    Just the Class is a modern, highly customizable, responsive Jekyll theme
    for developing course websites.
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

<!-- 
### Local development environment

Just the Class is built for [Jekyll](https://jekyllrb.com), a static site generator. View the [quick start guide](https://jekyllrb.com/docs/) for more information. Just the Docs requires no special Jekyll plugins and can run on GitHub Pages' standard Jekyll compiler.

1. Follow the GitHub documentation for [Setting up your GitHub Pages site locally with Jekyll](https://help.github.com/en/articles/setting-up-your-github-pages-site-locally-with-jekyll).
1. Start your local Jekyll server.
```bash
$ bundle exec jekyll serve
```
1. Point your web browser to [http://localhost:4000](http://localhost:4000)
1. Reload your web browser after making a change to preview its effect.

For more information, refer to [Just the Docs](https://pmarsceill.github.io/just-the-docs/). -->
