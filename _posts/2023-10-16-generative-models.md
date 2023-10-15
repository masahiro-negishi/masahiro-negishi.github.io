---
layout: post
title: Overview of Generative Models
date: 2023-10-16
description: motivation and a basic framework of generative models
tags: generative-model
categories: ML
related_posts: false
---

## Table of Contents
- [Motivation](#1-motivation)
- [Learning](#2-learning)
- [Reference](#3-reference)

## 1. Motivation
We assume that there is a true data distribution $$ p_{data}(x) $$, which is only accessible through $$ \lbrace x_1, x_2, ..., x_N \rbrace $$ that are sampled from $$ p_{data}(x) $$. The goal of generative models is to find an approximation of $$ p_{data}(x) $$:

$$
p_{\theta}(x) \approx p_{data}(x).
$$

A generative model is composed of its architecture and its parameter $$ \theta $$. The architecture reflects people's thought on how $$ p_{data}(x) $$ looks like. Parameter $$ \theta $$ determines remaining things. There are many applications of generative models including:
- generation of new samples
- abnormal detection, outlier detection
- denoising, missing value completion

## 2. Learning
As written in Section 1, the goal is to learn $$ p_{\theta}(x) $$ that approximates $$ p_{data}(x) $$. There are two issues to achieve this goal.
- Issue 1: $$ p_{data}(x) $$ is unknown
- Issue 2: It is unclear how to measure the "distance" between $$ p_{data}(x) $$ and $$ p_{\theta}(x) $$.

The first issue can be solved by approximating $$ p_{data}(x) $$ with an empirical distribution $$ \hat{p}_{data}^{N}(x) \coloneqq \frac{1}{N}\sum_{i=1}^{N}\delta(x-x_i) $$. The second issue can be solved by introducing KL divergence $$ D_{KL}[p(x) \Vert q(x)] \coloneqq \mathbb{E}_{p(x)}[\log\frac{p(x)}{q(x)}] $$. As a result, the learning objective is to derive the following $$ \hat{\theta} $$:

$$
\begin{aligned}
\hat{\theta} &\coloneqq \argmin_{\theta} D_{KL}[\hat{p}_{data}^{N}(x) \Vert p_{\theta}(x)] \\
&= \argmin_{\theta} \mathbb{E}_{\hat{p}_{data}^{N}(x)}[\log \hat{p}_{data}^{N}(x)] - \mathbb{E}_{\hat{p}_{data}^{N}(x)}[\log p_{\theta}(x)] \\
&= \argmax_{\theta} \mathbb{E}_{\hat{p}_{data}^{N}(x)}[\log p_{\theta}(x)] \\
&= \argmax_{\theta} \frac{1}{N} \sum_{i=1}^{N}\log p_{\theta}(x_i) \\
\bigl( &= \argmax_{\theta} \prod_{i=1}^{N}p_{\theta}(x_i) \bigr) \\
\end{aligned}
$$

From the above equations, you can understand that minimizing the KL divergence between $$ \hat{p} _ {data}^{N}(x) $$ and $$ p_{\theta}(x) $$ is equivalent to maximum likelihood estimation. Thus, in common maximum likelihood estimation, we should keep in mind that we use KL divergence as a distance metric. Due to the assymmetry property of KL divergence, there may be undesirable effects on learned results. In addition, we use $$ \hat{p}_{data}^{N}(x) $$ instead of $$ p_{data}(x) $$. Therefore, maximum likelihood estimation does not necessarilly lead to generalization. For example, if the number of training samples $N$ is small, you can fall into over-fitting.

## 3. Reference
- Summer seminar "Deep Generative Models" provided by [Matsuo Lab](https://weblab.t.u-tokyo.ac.jp/)