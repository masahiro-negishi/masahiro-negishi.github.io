---
layout: post
title: EM algorithm
date: 2023-10-16
description: derivation of EM algorithm
tags: generative-model
categories: ML
related_posts: false
---

## Table of Contents
- [Problem setting](#1-problem-setting)
- [Background](#2-background)
- [Algorithm](#3-algorithm)
- [Reference](#4-reference)

## 1. Problem setting
Assume there are latent variables $$ z $$, which cannot be observed. The goal is to maximize likelihood $$ \log p _ {\theta} (x) $$, where $x$ is observed variables. More specifically, given the training set $X$, the objective is to maximize likelihood $$ \log p _ {\theta} (X) $$. We assume two things to make the problem solvable with EM algorithm:
- $$ p _ {\theta} (x, z) $$ is tractable (For M step)
- $$ p _ {\theta} (z \vert x) $$ is tractable (For E step)



## 2. Background
You cannot directly maxmize $$ \log p _ {\theta} (x) $$, thus we consider evidence lower bound (ELBO) as follows:

$$
\begin{aligned}
\log p _ {\theta} (x) &= \log p _ {\theta} (x) \int q(z)dz \\
&= \int q(z) \log \frac{q(z) p _ {\theta} (x, z)}{q(z) p _ {\theta} (z \vert x)}dz \\
&= \int q(z) \log \frac{p _ {\theta} (x, z)}{q(z)}dz + \int q(z) \log \frac{q(z)}{p _ {\theta} (z \vert x)}dz \\
&= \mathcal{L} (q, \theta; x) + D _ {KL} [q(z) \Vert p _ {\theta} (z \vert x)].
\end{aligned}
$$

You can also derive that $$ \mathcal{L} (q, \theta; x) $$ is a lower bound of $$ \log p _ {\theta} (x) $$ by utilizing Jensen's inequality:

$$
\begin{aligned}
\log p _ {\theta} (x) &= \log \int p _ {\theta} (x, z) dz \\
&= \log \int q(z) \frac{p _ {\theta} (x, z)}{q(z)} dz \\
&\ge \int q(z) \log \frac{p _ {\theta} (x, z)}{q(z)} dz \\
&= \mathcal{L} (q, \theta; x).
\end{aligned}
$$

## 3. Algorithm
EM algorithm iterates two steps to maximize $$ \mathcal{L} (q, \theta; x) = \log p _ {\theta} (x) - D _ {KL} [q(z) \Vert p _ {\theta} (z \vert x)] $$.

### 3.1. E step
E step maximizes $$ \mathcal{L} (q, \theta _ t; x) $$ in terms of $$ q $$. This corresponds to minimizing $$ D _ {KL} [q(z) \Vert p _ {\theta _ t} (z \vert x)] $$ in terms of $$ q $$. Thus, E step updates $$ q(z) $$ as follows:

$$q(z) = p _ {\theta _ t} (z \vert x)$$

### 3.2. M step
M step maximizes $$ \mathcal{L} (q, \theta; x) $$ in terms of $$ \theta $$. Note that $$ q(x) $$ is set to $$ p _ {\theta _ t} (z \vert x) $$ in the last E step.

$$
\begin{aligned}
\theta _ {t+1} &\coloneq \underset{\theta}{\text{argmax}} \mathcal{L} (p _ {\theta _ t} (z \vert x), \theta; x) \\
&= \underset{\theta}{\text{argmax}} \int p _ {\theta _ t} (z \vert x) \log \frac{p _ {\theta} (x, z)}{p _ {\theta _ t} (z \vert x)}dz \\
&= \underset{\theta}{\text{argmax}} \int p _ {\theta _ t} (z \vert x) \log p _ {\theta} (x, z)dz \\
\end{aligned}
$$

Note: $$ \log p _ \theta (x) $$ monotonically increases, thus $$ \theta _ t $$ will converge to a local optima.

## 4. Reference
- Summer seminar "Deep Generative Models" provided by [Matsuo Lab](https://weblab.t.u-tokyo.ac.jp/)
- [A website provides a mathematical details](https://academ-aid.com/ml/em)