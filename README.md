# Data & Knowledge Representation – Graph ML Tutorial

### **Applying Graph Neural Networks to Paper and Author Classification in Scientific Networks**

**Live Tutorial Website:**  
[https://dkr-gcn-gat-tutorial.vercel.app](https://dkr-gcn-gat-tutorial.vercel.app)

---

## Overview

This repository contains an interactive tutorial for the **Data and Knowledge Representation** course, demonstrating how to apply **Graph Neural Networks (GNNs)** to real-world graph-structured data.

### Main tasks

1. **Classifying scientific papers in a citation network (Cora)**  
2. **Classifying authors into research fields in a co-authorship graph (Coauthor-CS)**  

We build and compare several GNN architectures:

- **Graph Convolutional Network (GCN)**
- **Graph Attention Network (GAT)**
- **Graph Attention Network v2 (GATv2 / GAT+)**
- **GraphSAGE**
- **APPNP** (Approximate Personalized Propagation of Neural Predictions)

Most models are implemented in **PyTorch Geometric (PyG)** and trained on:

- the **Cora citation dataset** (semi-supervised node classification), and  
- the **Coauthor-CS dataset** (larger co-authorship graph with 15 classes).

The project consists of:

- A **fully interactive tutorial website** explaining concepts step-by-step  
- **Jupyter/Colab notebooks** implementing all models  
- Visualizations of datasets and learned node embeddings (PCA, UMAP, t-SNE)  
- A discussion comparing models in terms of performance and representation quality  

---

## Project Goal

Our aim is to create a beginner-friendly, yet technically rigorous tutorial that assumes the reader: 

By the end of the tutorial, users should understand:

- What graph-structured data is and why graphs matter  
- How **message passing** and **neighborhood aggregation** work in GNNs  
- The core ideas and math behind **GCN, GAT, GATv2, GraphSAGE, and APPNP**  
- How to load and process **Cora** and **Coauthor-CS** in PyG  
- How to train and evaluate GNNs for **node classification**  
- How different architectures compare in terms of:
  - performance (accuracy),
  - complexity (parameters, training stability),
  - and representation quality (learned embeddings).

---

## Application Domain 1: Paper Classification in Citation Networks (Cora)

Scientific papers naturally form graphs: each paper cites other papers, creating a rich structure that goes beyond individual documents. The **Cora dataset** models this relationship and is a standard benchmark in graph learning.

| Property      | Value                               |
|--------------|--------------------------------------|
| Nodes        | 2,708 papers                         |
| Edges        | 5,429 citation links                 |
| Node features| 1,433 words (bag-of-words)           |
| Classes      | 7 research topics                    |
| Task         | Semi-supervised node classification  |

Each node (paper) has:

- a high-dimensional bag-of-words feature vector, and  
- a label indicating its research topic (e.g., Neural Networks, Probabilistic Methods, Theory, etc.).

The goal is to predict the topic of each paper using **both** its content and its position in the citation graph.

On Cora we focus on:

- **GCN** as a strong baseline  
- **GAT** as an attention-based extension

---

## Application Domain 2: Author Classification in Co-Authorship Networks (Coauthor-CS)

To go beyond the small Cora graph, we also experiment on the **Coauthor-CS** dataset from PyG:

- Nodes represent **authors**  
- Edges represent **co-authorship** relations  
- Node features are bag-of-words vectors built from paper keywords  
- Labels correspond to **15 research subfields** in computer science  

Main properties:

| Property      | Value                                          |
|--------------|-------------------------------------------------|
| Nodes        | 18,333 authors                                 |
| Edges        | 81,894 co-authorship links                     |
| Node features| 6,805-dimensional keyword bag-of-words vectors |
| Classes      | 15 CS research fields                          |
| Task         | Node classification (author’s main research area)|

On Coauthor-CS we evaluate more expressive GNNs:

- **GATv2 (GAT+)**
- **GraphSAGE**
- **APPNP**

We also include **interactive PCA and UMAP visualizations** of Coauthor-CS node features (with hoverable topics) to help understand the structure of the graph and the class distribution.

---

## Models Covered

### 1. Graph Convolutional Network (GCN)

GCN learns node representations via neighborhood feature aggregation using the layer-wise update:

$$
H^{(l+1)} = \sigma\big(\tilde{D}^{-1/2} \tilde{A}\tilde{D}^{-1/2} H^{(l)} W^{(l)}\big)
$$

where:

- $\tilde{A} = A + I$ is the adjacency matrix with self-loops  
- $\tilde{D}$ is the diagonal degree matrix of $\tilde{A}$  
- $H^{(l)}$ is the matrix of node features at layer $l$  
- $W^{(l)}$ is a learnable weight matrix  
- $\sigma$ is a non-linear activation (typically ReLU)  

On Cora, a simple **two-layer GCN** is sufficient: deeper GCNs may suffer from **over-smoothing**, where node embeddings become too similar across the graph.

**Cora results (GCN):**

- 2-layer GCN with hidden dim 64  
- Early stopping on validation accuracy  
- **Test accuracy ≈ 0.81**

---

### 2. Graph Attention Network (GAT)

GAT extends GCN with a **learnable attention mechanism** on edges. Instead of aggregating all neighbors uniformly (up to degree normalization), GAT learns to assign different importance to each neighbor.

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

We experiment with:

- different learning rates (`0.005`, `0.001`),  
- weight decay (`5e-4`, `1e-3`),  
- early-stopping patience (`10`, `20`),  
- hidden dimension (`4`, `8`),  
- number of attention heads (`4`, `8`).  

The final configuration is chosen based on **validation accuracy**.

**Cora results (GAT):**

- 2-layer GAT with multi-head attention  
- Early stopping based on validation accuracy  
- **Test accuracy ≈ 0.772**

Interestingly, on the small and clean Cora graph, the simpler GCN slightly outperforms GAT, illustrating that more expressive models do not always win when data is limited.

---

### 3. GATv2 (GAT+) on Coauthor-CS

**GATv2** (often referred to as GAT+) improves the original GAT by changing where the non-linearity is applied in the attention mechanism.

Original GAT attention:

$$
\alpha_{ij} = \text{softmax}_j\big(\text{LeakyReLU}(a^\top [W h_i \Vert W h_j])\big)
$$

GATv2 attention:

$$
\alpha_{ij} = \text{softmax}_j\big(a^\top \text{LeakyReLU}(W [h_i \Vert h_j])\big)
$$

This subtle reordering makes the attention mechanism **universally expressive** and fully differentiable with respect to the input features.

**Coauthor-CS configuration (GATv2):**

- 2 layers with `GATv2Conv`  
- Hidden dimension = 8, attention heads = 8 in the first layer  
- Dropout = 0.3–0.6  
- Adam optimizer, learning rate = 0.01, weight decay = 1e-4  
- Early stopping with patience = 20  

**Result:**

- **Test accuracy (GATv2 on Coauthor-CS): 0.919**

---

### 4. GraphSAGE on Coauthor-CS

**GraphSAGE** (Sample and Aggregate) is an **inductive** GNN designed to generalize to unseen nodes. It learns a neighborhood aggregation function that can be applied to new graphs or new nodes without retraining from scratch.

Two key steps:

1. **Neighborhood aggregation**  
   $$
   h_{\mathcal{N}(i)}^{(l+1)} = \text{aggregate}\big( \{ h_j^{(l)} : j \in \mathcal{N}(i) \} \big)
   $$
2. **Node update**  
   $$
   h_i^{(l+1)} = \sigma\left( W \cdot [h_i^{(l)} \Vert h_{\mathcal{N}(i)}^{(l+1)}] \right)
   $$

In our implementation we use `SAGEConv` with **mean aggregation**, which fits the homophilous structure of Coauthor-CS (authors in the same field tend to co-author).

**Coauthor-CS configuration (GraphSAGE):**

- 2 layers with `SAGEConv` (mean aggregation)  
- Hidden dimension = 64  
- ReLU + dropout in the hidden layer  
- Adam optimizer, learning rate = 0.01, weight decay = 5e-4  
- Early stopping with patience = 20  

**Result:**

- **Test accuracy (GraphSAGE on Coauthor-CS): 0.93**

GraphSAGE achieves the best accuracy among our models on Coauthor-CS.

---

### 5. APPNP on Coauthor-CS

**APPNP** (Approximate Personalized Propagation of Neural Predictions) separates **prediction** from **propagation**:

1. First, a small neural network (here: a 2-layer MLP) predicts logits:
   $$
   Z = f_\theta(X)
   $$
2. Then, APPNP performs k steps of personalized PageRank-style propagation:
   $$
   H^{(0)} = Z, \quad H^{(k+1)} = (1 - \alpha)\,\tilde{A} H^{(k)} + \alpha Z
   $$

Here:

- $\alpha$ is the teleport probability  
- $\tilde{A}$ is the normalized adjacency with self-loops  

This formulation allows information to travel over many hops while still keeping the original predictions $Z$ in the mixture, helping to mitigate oversmoothing.

**Coauthor-CS configuration (APPNP):**

- MLP: input → 64 hidden units → num_classes  
- Propagation: `APPNP(K=10, alpha=0.1)`  
- Dropout = 0.5  
- Adam optimizer, learning rate = 0.01, weight decay = 5e-4  
- Early stopping with patience = 20  

**Result:**

- **Test accuracy (APPNP on Coauthor-CS): 0.9289**

APPNP performs on par with GraphSAGE, with slightly smoother and more well-separated t-SNE clusters in the learned node embeddings.

---

## Loss Functions and Activations

All models are trained with **cross-entropy loss** on the labeled nodes (standard for multi-class classification):

- Targets: integer class labels (`0..C-1`)  
- Outputs: class logits from the final layer  
- Loss: `torch.nn.functional.cross_entropy`  

This choice is natural because we care about assigning each node to exactly one topic/field and evaluate using **accuracy**.

For activations, we use:

- **ReLU** in GCN, GraphSAGE, and the MLP part of APPNP:
  - $\text{ReLU}(x) = \max(0, x)$
  - Simple, efficient, but completely zeroes out negative activations.
- **LeakyReLU / ELU** in attention-based models:
  - In GAT/GATv2, **LeakyReLU** is used inside the attention mechanism to avoid “dead” negative responses and keep a small gradient for negative inputs.
  - ELU/LeakyReLU help stabilize the attention scores, especially when many neighbors have similar features.

Intuitively:

- For standard feature transformations we prefer ReLU (fast, simple).  
- For **attention logits**, we prefer LeakyReLU to allow negative inputs to still influence the gradient and avoid degenerate attention patterns.

---

## Results Overview

### Cora (paper classification)

| Model            | Dataset | Test Accuracy |
|------------------|---------|---------------|
| GCN (2 layers)   | Cora    | ~0.81         |
| GAT              | Cora    | ~0.772        |

### Coauthor-CS (author classification)

| Model    | Dataset     | Test Accuracy |
|----------|-------------|---------------|
| GATv2    | Coauthor-CS | 0.919         |
| GraphSAGE| Coauthor-CS | 0.930         |
| APPNP    | Coauthor-CS | 0.9289        |

Qualitatively, t-SNE plots of learned embeddings show:

- GCN on Cora: compact clusters, good class separation.  
- GAT on Cora: still clustered but with more overlap for some classes.  
- On Coauthor-CS:
  - **APPNP** produces especially smooth, well-structured clusters,
  - **GraphSAGE** yields strong separation consistent with its top accuracy,
  - **GATv2** captures more complex decision boundaries thanks to its expressive attention.

---

## Tutorial Website

The tutorial website walks through the entire pipeline:

- Cora dataset structure and class distribution  
- Coauthor-CS description and interactive PCA/UMAP projections  
- Intuition behind message passing on graphs  
- GCN math and implementation with `GCNConv`  
- GAT and GATv2 math and implementation with `GATConv` / `GATv2Conv`  
- GraphSAGE and APPNP architectures and their inductive/propagation behavior  
- Training curves, validation accuracy, and early stopping  
- t-SNE, PCA, and UMAP visualizations of learned embeddings  
- A qualitative and quantitative comparison across all models  

**Visit the site:**  
[https://dkr-gcn-gat-tutorial.vercel.app](https://dkr-gcn-gat-tutorial.vercel.app)

---

## Colab Notebooks

You can run all experiments directly in Google Colab.

### Cora (GCN & GAT)

| Notebook              | Link |
|-----------------------|------|
| GCN on Cora           | [Open in Colab](https://colab.research.google.com/drive/1Jv7DPatVyO61ydvaOjDJAkjiowV_aKNx?authuser=0#scrollTo=dAIZ3kkY3lTE) |
| GAT on Cora           | [Open in Colab](https://colab.research.google.com/drive/1y2B-9Dsn8JBB8q5jR75p1GInkFc-7i8i?usp=sharing#scrollTo=BH5iYWvhLnBN) |
| GCN vs GAT comparison | [Open in Colab](https://colab.research.google.com/drive/1ROgAUnXgXtReR3-tfGvHZkIC9iUdAq03?usp=sharing) |

### Coauthor-CS (Extended Experiments)

| Notebook        | Link |
|-----------------|------|
| GATv2 (GAT+)    | [Open in Colab](https://colab.research.google.com/drive/1SWOraiV1drvd3f5NE13uYjuKTwg4J7XY?usp=sharing) |
| GraphSAGE       | [Open in Colab](https://colab.research.google.com/drive/1XhQG5dKLJH0jmxToxfsL51Ik0XwvEwbM?usp=sharing) |
| APPNP           | [Open in Colab](https://colab.research.google.com/drive/1MqEnp4woLOKpCZgZy2Z8_nyhcbkeHnfW?usp=sharing) |

Each notebook:

- installs PyTorch + PyTorch Geometric,  
- loads the corresponding dataset (Cora or Coauthor-CS),  
- defines and trains the chosen model,  
- tracks training loss and validation accuracy with early stopping,  
- evaluates test accuracy and visualizes node embeddings.

---

## References

- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017.  
- Veličković et al., *Graph Attention Networks*, ICLR 2018.  
- Brody, Alon & Yahav, *How Attentive are Graph Attention Networks?*, ICLR 2022 (GATv2).  
- Hamilton, Ying & Leskovec, *Inductive Representation Learning on Large Graphs*, NeurIPS 2017 (GraphSAGE).  
- Klicpera, Bojchevski & Günnemann, *Predict then Propagate: Graph Neural Networks meet Personalized PageRank*, ICLR 2019 (APPNP).

---

## Team

This project was developed as part of the  
**Data and Knowledge Representation – Machine Learning for Graphs** course.

**Team members:**

- **Yasmina Mamadalieva** - GCN & GraphSAGE implementation & experiments  
- **Sofa Goryunova** - GAT & GATv2 implementation, experiments, and attention analysis  
- **Ekaterina Akimenko** - APPNP implementation, Tutorial writing, website development, comparison notebook, and visualizations  

---

## Deployment

The site is deployed using **Vercel** with a **React + Vite** frontend.  
All figures and interactive HTML visualizations (PCA/UMAP/t-SNE) are built from the Colab notebooks and integrated into the website.
