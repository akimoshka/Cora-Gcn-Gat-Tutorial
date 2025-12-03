# Data & Knowledge Representation – Graph ML Tutorial

### **Applying Graph Convolutional and Attention Networks to Paper Classification in Citation Networks**

**Live Tutorial Website:**  
[https://dkr-gcn-gat-tutorial.vercel.app](https://dkr-gcn-gat-tutorial.vercel.app)

---

## Overview

This repository contains an interactive tutorial for the **Data and Knowledge Representation** course, demonstrating how to apply **Graph Neural Networks (GNNs)** to a real-world machine learning problem:

### _Classifying scientific papers in a citation network._

We build and compare two foundational GNN architectures:

- **Graph Convolutional Network (GCN)**
- **Graph Attention Network (GAT)**

Both models are implemented in **PyTorch Geometric (PyG)** and trained on the **Cora citation dataset**, one of the most widely used benchmarks in graph machine learning.

The project consists of:

- A **fully interactive tutorial website** explaining concepts step-by-step  
- **Jupyter notebooks** (Colab-ready) implementing GCN and GAT  
- Visualizations of the dataset and learned node embeddings  
- A discussion comparing both architectures in terms of performance and behavior

---

## Project Goal

Our aim is to create a beginner-friendly, yet technically rigorous tutorial that assumes the reader:

- ✔ knows Python, PyTorch, and basic deep learning  
- ✖️ but is _new to_ Graph Machine Learning and PyTorch Geometric  

By the end of the tutorial, users should understand:

- What graph-structured data is and why graphs matter  
- How message passing works in GNNs  
- The core ideas and math behind GCN and GAT  
- How to load and process Cora in PyG  
- How to train and evaluate graph neural networks on citation networks  
- How GCN and GAT differ in performance, complexity, and interpretability  

---

## Application Domain: Paper Classification in Citation Networks

Scientific papers naturally form graphs: each paper cites other papers, creating a rich structure that goes beyond individual documents. The **Cora dataset** models this relationship and is a standard benchmark in graph learning.

| Property      | Value                               |
|--------------|-------------------------------------|
| Nodes         | 2,708 papers                        |
| Edges         | 5,429 citation links                |
| Node features | 1,433 words (bag-of-words)          |
| Classes       | 7 research topics                   |
| Task          | Semi-supervised node classification |

Each node (paper) has:

- a high-dimensional bag-of-words feature vector, and  
- a label indicating its research topic (e.g., Neural Networks, Probabilistic Methods, Theory, etc.).

The goal is to predict the topic of each paper using **both** its content and its position in the citation graph.

---

## Models Covered

### Graph Convolutional Network (GCN)

GCN learns node representations via neighborhood feature aggregation using the layer-wise update:

$$
H^{(l+1)} = \sigma\big(\tilde{D}^{-1/2} \tilde{A}\tilde{D}^{-1/2} H^{(l)} W^{(l)}\big)
$$

where:

- $\tilde{A} = A + I$ is the adjacency matrix with self-loops  
- $\tilde{D}$ is the diagonal degree matrix of $\tilde{A}$  
- $H^{(l)}$ is the matrix of node features at layer $l$  
- $W^{(l)}$ is a learnable weight matrix  
- $\sigma$ is a non-linear activation (e.g. ReLU)  

On Cora, a simple **two-layer GCN** is often sufficient and deeper GCNs may suffer from over-smoothing of node representations.

---

### Graph Attention Network (GAT)

GAT extends GCN with a **learnable attention mechanism** on edges. Instead of aggregating all neighbors uniformly (after normalization), GAT learns to assign different importance to each neighbor.

The update rule for node $i$ in a GAT layer is:

$$
h'_i = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j\right)
$$

where:

- $h_i$ is the input feature vector of node $i$  
- $W$ is a learnable weight matrix  
- $\alpha_{ij}$ are attention coefficients for neighbor $j$ in the neighborhood $\mathcal{N}(i)$  

The attention coefficients are computed as:

$$
\alpha_{ij} = \frac{\exp\big(\text{LeakyReLU}(a^\top [Wh_i \Vert Wh_j])\big)}{\sum_{k \in \mathcal{N}(i)} \exp\big(\text{LeakyReLU}(a^\top [Wh_i \Vert Wh_k])\big)}
$$

In the notebook, we experiment with:

- different learning rates (`0.005`, `0.001`),  
- weight decay (`5e-4`, `1e-3`),  
- early-stopping patience (`10`, `20`),  
- hidden dimension (`4`, `8`),  
- number of attention heads (`4`, `8`).

The final configuration is chosen based on **validation accuracy** to balance performance and overfitting.

---

## Tutorial Website

The tutorial website walks through the entire pipeline:

- Cora dataset structure and class distribution  
- Intuition behind message passing on graphs  
- GCN math and its implementation with `GCNConv`  
- GAT math and its implementation with `GATConv`  
- Training curves, validation accuracy, and early stopping  
- t-SNE and PCA visualizations of learned embeddings  
- A qualitative and quantitative comparison between GCN and GAT  

**Visit the site:**  
👉 [https://dkr-gcn-gat-tutorial.vercel.app](https://dkr-gcn-gat-tutorial.vercel.app)

---

## Colab Notebooks

You can run all experiments directly in Google Colab:

| Notebook              | Link |
|-----------------------|------|
| GCN on Cora           | [Open in Colab](https://colab.research.google.com/drive/1Jv7DPatVyO61ydvaOjDJAkjiowV_aKNx?authuser=0#scrollTo=dAIZ3kkY3lTE) |
| GAT on Cora           | [Open in Colab](https://colab.research.google.com/drive/1y2B-9Dsn8JBB8q5jR75p1GInkFc-7i8i?usp=sharing#scrollTo=BH5iYWvhLnBN) |
| GCN vs GAT comparison | _(coming soon)_ |

Each notebook:

- installs PyTorch + PyTorch Geometric,  
- loads the Cora dataset,  
- defines and trains the model (GCN or GAT),  
- tracks training loss and validation accuracy,  
- evaluates test accuracy and visualizes embeddings.

---

## Team

This project was developed as part of the  
**Data and Knowledge Representation – Machine Learning for Graphs** course.

**Team members:**

- **Yasmina Mamadalieva** — GCN implementation & experiments  
- **Sofa Goryunova** — GAT implementation, experiments, and attention analysis  
- **Ekaterina Akimenko** — Tutorial writing, website development, comparison notebook, and visualizations  

---

## Deployment

The site is deployed using **Vercel** with a **React + Vite** setup.
