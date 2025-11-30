# Data & Knowledge Representation – Graph ML Tutorial

### **Applying Graph Convolutional and Attention Networks to Paper Classification in Citation Networks**

**Live Tutorial Website:**\
[https://dkr-gcn-gat-tutorial.vercel.app](https://dkr-gcn-gat-tutorial.vercel.app)

---

## Overview

This repository contains an interactive tutorial for the **Data and Knowledge Representation** course, demonstrating how to apply **Graph Neural Networks (GNNs)** to a real-world machine learning problem:

### *Classifying scientific papers in a citation network.*

We build and compare two foundational GNN architectures:

* **Graph Convolutional Network (GCN)**
* **Graph Attention Network (GAT)**

Both are implemented in **PyTorch Geometric (PyG)** and trained on the **Cora citation dataset**, one of the most widely used benchmarks in graph machine learning.

The project consists of:

* A **fully interactive tutorial website** explaining concepts step-by-step
* **Jupyter notebooks** (Colab-ready) implementing GCN and GAT
* Visualizations and interpretation of learned embeddings
* A discussion comparing both architectures

---

## Project Goal

Our aim is to create a beginner-friendly yet technically rigorous tutorial that assumes the reader:

✔ knows Python, PyTorch, and basic deep learning
✖️ but is *new to* Graph Machine Learning and PyTorch Geometric

By the end of the tutorial, users should understand:

* What graph-structured data is and why graphs matter
* How message passing works in GNNs
* The math behind GCN and GAT
* How to load and process Cora in PyG
* How to train and evaluate graph neural networks
* How GCN and GAT differ in performance and complexity

---

## Application Domain: Paper Classification in Citation Networks

Scientific papers form natural graphs: each paper cites others, forming a rich structure.
The **Cora dataset** models this relationship:

| Property      | Value                               |
| ------------- | ----------------------------------- |
| Nodes         | 2,708 papers                        |
| Edges         | 5,429 citation links                |
| Node features | 1,433 words (bag-of-words)          |
| Classes       | 7 research topics                   |
| Task          | Semi-supervised node classification |

---

## Models Covered

### - Graph Convolutional Network (GCN)

Learns node representations via neighborhood feature aggregation using:

$$
H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A}\tilde{D}^{-1/2} H^{(l)} W^{(l)})
$$

### - Graph Attention Network (GAT)

Extends GCN with a learnable attention mechanism:

$$
h'_i = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j\right)
$$

where attention weights ( $\alpha_{ij}$, $\mathcal{N}(i)$ ) depend on the importance of neighbor (j).

---

## Tutorial Website

Our tutorial website walks through:

* Cora dataset breakdown
* Message passing intuition
* GCN math and PyG implementation
* GAT math and PyG implementation
* Training curves, loss plots, and accuracy comparisons
* Embedding visualization
* Final evaluation and discussion

 **Visit the site:**
[https://dkr-gcn-gat-tutorial.vercel.app](https://dkr-gcn-gat-tutorial.vercel.app)

---

## Colab Notebooks


| Notebook              | Link            |
| --------------------- | --------------- |
| GCN on Cora           | *(coming soon)* |
| GAT on Cora           | *(coming soon)* |
| GCN vs GAT comparison | *(coming soon)* |

---

## Team

This project was developed as part of the
**Data and Knowledge Representation – Machine Learning for Graphs** course.

**Team members:**

* **Yasmina Mamadalieva** — GCN implementation & experiments
* **Sofa Goryunova** — GAT implementation & experiments
* **Ekaterina Akimenko** — Tutorial writing, website development, comparison analysis

---

## Deployment

The site is deployed using **Vercel** with a React + Vite setup.