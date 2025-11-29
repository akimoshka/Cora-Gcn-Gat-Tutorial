
# 1. Introduction: Why Graph ML?

Deep learning usually works on grids (images), sequences (text), or tables.
But many real-world systems are **networks**:

* citation networks
* social networks
* knowledge graphs
* molecules
* transaction networks
* recommendation systems

These structures can’t be captured by CNNs or transformers directly.
This is where **Graph Neural Networks** come in.

In this tutorial, we focus on the classic benchmark:

### Classifying scientific papers using their citations + text features

Each paper is a **node**, each citation is an **edge**, and each node has a **1433-dimensional bag-of-words feature vector**.
The goal is to predict the **research topic** of each paper.

---

# 2. Dataset Overview: The Cora Citation Network

We use the Cora dataset from Planetoid (built into PyG):

```python
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]
```

### Dataset summary

| Property        | Value                               |
| --------------- | ----------------------------------- |
| Number of nodes | 2708 papers                         |
| Number of edges | 5429 citations                      |
| Node features   | 1433 words                          |
| Classes         | 7 research topics                   |
| Task            | Semi-supervised node classification |

Why Cora?

* It is small and fast to train
* It has meaningful community structure
* Perfect for GCN/GAT comparisons
* Very widely used in graph ML tutorials and papers

---

# 3. What We Build

We train and compare **two fundamental GNN architectures**:

### - Graph Convolutional Network (GCN)

* Performs **weighted neighborhood averaging**
* Fast, simple, widely used
* Ideal baseline

### -  Graph Attention Network (GAT)

* Learns **attention weights** on edges
* Allows the model to focus on the most relevant neighbors
* More expressive than GCN

Both are implemented with PyTorch Geometric.

---

# 4. Understanding the Models (Intuitively)

This tutorial is designed so someone new to graph ML can still follow everything.

---

## 4.1 GCN — Graph Convolutional Network

GCN performs a “smoothed feature aggregation”:

[
H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A}\tilde{D}^{-1/2} H^{(l)} W^{(l)})
]

In plain language:

> Every node updates its representation by taking a **normalized average** of its neighbors’ features (including itself).

This works extremely well when the graph is **homophilous** (connected nodes tend to have similar labels) — and Cora is such a graph.

### GCN model (from `01_gcn_cora.ipynb`)

```python
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 64)
        self.conv2 = GCNConv(64, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

It’s simple but surprisingly strong.

---

## 4.2 GAT — Graph Attention Network

Instead of averaging all neighbors equally, GAT learns **attention scores** for each edge:

[
\alpha_{ij} = \text{softmax}_j(a^\top [Wh_i ,\Vert, Wh_j])
]

This answers the question:

> “Which neighboring papers are most relevant when classifying this one?”

GAT can capture:

* important citations
* asymmetric influence
* local heterogeneity in neighborhoods

We use `GATConv` from PyG.

---

# 5. Running the Tutorial

## Option A — Google Colab (recommended)

No installation needed. Each notebook works entirely in Colab:

| Notebook              | Link          |
| --------------------- | ------------- |
| GCN Tutorial          | *to be added* |
| GAT Tutorial          | *to be added* |
| GCN vs GAT Comparison | *to be added* |

---

## Option B — Run locally

### Clone the repository

```bash
git clone https://github.com/yourusername/cora-gcn-gat-tutorial.git
cd cora-gcn-gat-tutorial
```

### Install requirements

```
pip install -r requirements.txt
```

### Launch Jupyter

```
jupyter notebook
```

Open any notebook from the `/notebooks/` folder.

---

# 6. GCN Results (early experiments)

After experimenting with several GCN architectures:

* deeper GCNs → worse performance
* residual GCNs → no improvement
* batchnorm + dropout → did not beat simple model

This is expected because **Cora is small** and **homophilous** — deep GNNs tend to **over-smooth** features.

A simple **2-layer GCN** works best.

### Plots from the notebook

(Your real plots go here later.)

* Training loss curve
* Validation accuracy
* t-SNE visualization of learned node embeddings

These will be added from `/figures/`.

---

# 7. GCN vs GAT — Comparison (planned)

This section will be updated once the GAT results are ready.

We will compare:

* ✔ Test accuracy
* ✔ Training speed
* ✔ Attention weights visualization
* ✔ Embedding quality (t-SNE)
* ✔ Model capacity vs overfitting

Expected outcome:

* GAT may slightly outperform GCN
* but GCN is faster and simpler
* both models are strong baselines for citation networks

---

# 8. Team

* **Yasmina Mamadalieva** – GCN implementation
* **Sofa Goryunova** – GAT implementation
* **Ekaterina Akimenko** – Project structure, tutorial writing, comparison notebook & plots

---

# 10. Conclusion

In this tutorial, we demonstrated how to:

* load a graph dataset with PyG
* implement GCN and GAT from scratch
* train them on the Cora citation network
* visualize node embeddings
* compare two foundational GNN architectures

Our goal was to make graph ML **easy to learn**, **practical**, and **well-visualized**, even if you are completely new to PyTorch Geometric.

If you're following along, feel free to extend our work:

* Add GraphSAGE
* Try different datasets
* Add hyperparameter sweeps
* Explore interpretability of attention scores