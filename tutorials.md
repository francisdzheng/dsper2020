---
layout: page
title: Tutorials
nav_order: 2
description: An embedded Google Calendar displaying the weekly event schedule.
has_children: true
---

# Tutorials

These tutorials are meant to accompany the homework assignments in this class, and provide you with the necessary fundamentals to succeed. All tutorials will be done in Python. 

## Modules

{% for module in site.modules %}
{{ module }}
{% endfor %}
