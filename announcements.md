---
layout: page
title: Announcements
nav_exclude: true
description: A feed containing all of the class announcements.
---

# Announcements

[Piazza](https://piazza.com/class/k8pcxfiwkxf2ec) is our main platform for announcements, but important announcements will also be summarized here. When in doubt, please check Piazza.

{% assign announcements = site.announcements | reverse %}
{% for announcement in announcements %}
{{ announcement }}
{% endfor %}
