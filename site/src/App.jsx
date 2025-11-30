// src/App.jsx
import React from "react";
import Section from "./components/Section";

function App() {
  return (
    <div className="app">
      <header className="hero">
        <h1 className="title-main">
          Data and Knowledge Representation Course Tutorial
        </h1>

        <h2 className="title-sub">
          Applying Graph Convolutional and Attention Networks to Paper
          Classification in Citation Networks
        </h2>

        <h3 className="authors">
          By: Yasmina Mamadalieva — Ekaterina Akimenko — Sofa Goryunova
        </h3>

        <nav className="nav">
          <a href="#intro">Intro</a>
          <a href="#cora">Cora Dataset</a>
          <a href="#gcn">GCN</a>
          <a href="#gat">GAT</a>
          <a href="#comparison">Comparison</a>
          <a href="#colab">Colab</a>
          <a href="#team">Team</a>
        </nav>
      </header>

      <main>
        {/* 1. Intro */}
        <Section id="intro" title="1. Introduction: Graph ML for Citation Networks">
          <p>
            In many machine learning tasks, we assume that data points are
            independent: each image, each sentence, each tabular row is treated
            separately. However, in a lot of real-world scenarios this is not
            true. Scientific papers cite each other, users in social networks
            are connected, products are co-purchased together. In all these
            cases, the structure between objects carries important information.
          </p>

          <p>
            Graph Neural Networks (GNNs) are a family of models that can exploit
            this structure. They operate on graphs, where:
          </p>
          <ul>
            <li>
              <strong>Nodes</strong> represent entities (here: scientific
              papers),
            </li>
            <li>
              <strong>Edges</strong> represent relationships (here: citation
              links),
            </li>
            <li>
              <strong>Node features</strong> describe each entity (here:
              bag-of-words features of the paper text).
            </li>
          </ul>

          <p>
            In this tutorial, we focus on a classic benchmark problem:{" "}
            <strong>
              scientific paper classification in a citation network
            </strong>
            . The task is:
          </p>

          <p className="emphasis">
            Given a graph where each node is a paper and edges are citation
            links, predict the research topic of each paper using both its text
            features and its position in the citation graph.
          </p>

          <p>
            We will work with the Cora dataset and implement two fundamental
            GNN architectures:
          </p>
          <ul>
            <li>
              <strong>Graph Convolutional Network (GCN)</strong> — aggregates
              information from neighbors using a normalized graph convolution.
            </li>
            <li>
              <strong>Graph Attention Network (GAT)</strong> — extends GCN by
              learning <em>attention weights</em> for each edge, so some
              neighbors can be more important than others.
            </li>
          </ul>

          <p>
            The goal of this site is to guide you step by step, assuming you are
            familiar with PyTorch and deep learning, but new to graph machine
            learning and PyTorch Geometric.
          </p>
        </Section>

        {/* 2. Cora */}
        <Section id="cora" title="2. The Cora Citation Dataset">
          <p>
            The <strong>Cora</strong> dataset is one of the most widely used
            benchmarks in graph learning. It is a citation network of machine
            learning papers, and it is conveniently available in{" "}
            <code>torch_geometric.datasets.Planetoid</code>. Each node is a
            scientific publication, and each directed edge represents a citation
            from one paper to another.
          </p>

          <p>The main properties of the dataset are:</p>

          <ul>
            <li>
              <strong>Nodes:</strong> 2,708 scientific papers
            </li>
            <li>
              <strong>Edges:</strong> 5,429 citation links
            </li>
            <li>
              <strong>Node features:</strong> 1,433-dimensional bag-of-words
              vectors
            </li>
            <li>
              <strong>Labels:</strong> 7 research topics (e.g. Neural Networks,
              Probabilistic Methods, Theory, etc.)
            </li>
          </ul>

          <p>
            Our prediction task is <strong>node classification</strong>: predict
            the topic label of each paper. We will use{" "}
            <strong>classification accuracy</strong> on the test set as the
            main evaluation metric.
          </p>

          <p>
            Loading Cora in PyTorch Geometric is only a few lines of code:
          </p>

          <pre>
{`from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

print(data)
print("Classes:", dataset.num_classes)
print("Features per node:", dataset.num_node_features)`}
          </pre>

          <p>
            The <code>data</code> object is a single graph containing:
          </p>
          <ul>
            <li>
              <code>data.x</code> — node feature matrix (2708 × 1433),
            </li>
            <li>
              <code>data.edge_index</code> — list of edges in COO format,
            </li>
            <li>
              <code>data.y</code> — node labels (0–6),
            </li>
            <li>
              <code>data.train_mask</code>, <code>val_mask</code>,{" "}
              <code>test_mask</code> — boolean masks for semi-supervised
              training.
            </li>
          </ul>

          <p>
            Cora is small, easy to visualize, and directly integrated into PyG.
            This makes it ideal for a detailed, step-by-step tutorial and for a
            fair comparison between different message-passing layers such as GCN
            and GAT.
          </p>
        </Section>

        {/* 3. GCN */}
        <Section id="gcn" title="3. Graph Convolutional Network (GCN)">
          <p>
            We start with the Graph Convolutional Network (GCN), which is one of
            the foundational architectures in graph deep learning. The key idea
            is to update the representation of each node by aggregating
            normalized features from its neighbors.
          </p>

          <p>
            A single GCN layer can be written as the following update rule:
          </p>

          <p className="formula">
            H<sup>(l+1)</sup> = σ( D̃<sup>-1/2</sup> Ã D̃<sup>-1/2</sup> H
            <sup>(l)</sup> W<sup>(l)</sup> )
          </p>

          <p>Here:</p>
          <ul>
            <li>
              Ã = A + I is the adjacency matrix with self-loops (each node is
              connected to itself),
            </li>
            <li>
              D̃ is the diagonal degree matrix of Ã (D̃<sub>ii</sub> =
              ∑<sub>j</sub> Ã<sub>ij</sub>),
            </li>
            <li>
              H<sup>(l)</sup> is the matrix of node features at layer{" "}
              <code>l</code>,
            </li>
            <li>W<sup>(l)</sup> is a learnable weight matrix,</li>
            <li>
              σ is a non-linear activation function such as ReLU.
            </li>
          </ul>

          <p>
            The normalization D̃<sup>-1/2</sup> Ã D̃<sup>-1/2</sup> ensures that
            features are scaled in a way that stabilizes training and prevents
            high-degree nodes from dominating the aggregation.
          </p>

          <p>
            In PyTorch Geometric, this entire operation is implemented inside{" "}
            <code>GCNConv</code>, so the model code stays compact:
          </p>

          <pre>
{`import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # 1st layer: propagate + non-linearity + dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # 2nd layer: propagate to output dimension (num_classes)
        x = self.conv2(x, edge_index)
        return x`}
          </pre>

          <p>
            For Cora, we use a simple two-layer GCN with hidden dimension 64.
            Despite its simplicity, this model already achieves strong
            performance. In practice, making the model deeper on Cora often{" "}
            <em>hurts</em> performance due to{" "}
            <strong>over-smoothing</strong>: after many layers, node
            representations tend to become too similar across the graph and lose
            class-specific information.
          </p>

          <p>
            During training, we only use the nodes marked by{" "}
            <code>data.train_mask</code>, while{" "}
            <code>val_mask</code> and <code>test_mask</code> are used to monitor
            validation accuracy and compute the final test accuracy.
          </p>
        </Section>

        {/* 4. GAT */}
        <Section id="gat" title="4. Graph Attention Network (GAT)">
          <p>
            Graph Convolutional Networks treat all neighbors of a node equally
            (up to the degree-based normalization). However, in real citation
            networks, not all neighbors are equally informative. Some references
            might be central to the topic of a paper, while others are only
            loosely related.
          </p>

          <p>
            Graph Attention Networks (GAT) address this by learning{" "}
            <strong>attention weights</strong> on edges. Instead of averaging all
            neighbors uniformly, the model learns which neighbors are more
            important for each node.
          </p>

          <p>
            The update rule for node <code>i</code> in a GAT layer can be
            written as:
          </p>

          <p className="formula">
            h′<sub>i</sub> = σ ( ∑<sub>j ∈ 𝒩(i)</sub> α<sub>ij</sub> W h
            <sub>j</sub> )
          </p>

          <p>where:</p>
          <ul>
            <li>h<sub>i</sub> is the input feature vector of node i,</li>
            <li>W is a learnable weight matrix,</li>
            <li>
              α<sub>ij</sub> is the attention coefficient for edge (i, j).
            </li>
          </ul>

          <p>
            The attention coefficients α<sub>ij</sub> are computed as:
          </p>

          <p className="formula">
            α<sub>ij</sub> = softmax<sub>j</sub>( LeakyReLU( aᵀ [W h
            <sub>i</sub> ∥ W h<sub>j</sub>] ) )
          </p>

          <p>
            Here, <code>a</code> is a learnable vector, and{" "}
            <code>[· ∥ ·]</code> denotes concatenation. The softmax is taken
            over all neighbors of node <code>i</code>, so the attention weights
            for each node sum to 1.
          </p>

          <p>
            In PyTorch Geometric, this mechanism is implemented in{" "}
            <code>GATConv</code>. A typical GAT layer might look like:
          </p>

          <pre>
{`from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.6)
        # concat heads from layer 1 -> hidden_dim * heads
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x`}
          </pre>

          <p>
            By comparing GAT to GCN on the same dataset, we can study when
            learning attention weights actually improves performance, and how it
            changes the learned node embeddings.
          </p>
        </Section>

        {/* 5. Comparison */}
        <Section id="comparison" title="5. Experimental Setup & Comparison">
          <p>
            To compare GCN and GAT fairly, we keep the experimental setup as
            similar as possible:
          </p>

          <ul>
            <li>Same Cora dataset and train/val/test split (from Planetoid)</li>
            <li>Same optimizer (Adam) and similar learning rate</li>
            <li>Early stopping based on validation accuracy</li>
            <li>Classification accuracy on the test set as the main metric</li>
          </ul>

          <p>
            For each model, we will record and visualize the following:
          </p>
          <ul>
            <li>Training loss over epochs</li>
            <li>Validation accuracy over epochs</li>
            <li>
              Final test accuracy (best model according to validation accuracy)
            </li>
            <li>
              2D t-SNE plots of the final node embeddings, colored by class
            </li>
          </ul>

          <p>
            The 2D embeddings allow us to visually inspect how well each model
            separates the different research topics in representation space.
            Tight, well-separated clusters indicate that the model learned useful
            structure from both node features and the citation graph.
          </p>

          <p>
            In many published results, GAT slightly improves over GCN on Cora,
            but at the cost of higher computational complexity. Our comparison
            aims to illustrate this trade-off between <strong>expressiveness</strong> (via attention) and <strong>simplicity</strong> (via plain convolution).
          </p>
        </Section>

        {/* 6. Colab */}
        <Section id="colab" title="6. Run the Code in Google Colab">
          <p>
            All experiments in this tutorial are implemented as Jupyter
            notebooks. To make them easy to run and reproduce, we provide
            Google Colab links (to be filled with actual URLs):
          </p>

          <ul>
            <li>GCN on Cora — Colab</li>
            <li>GAT on Cora — Colab</li>
            <li>GCN vs GAT comparison — Colab</li>
          </ul>

          <p>
            Each notebook:
          </p>
          <ul>
            <li>installs the correct PyTorch and PyG versions,</li>
            <li>loads the Cora dataset,</li>
            <li>defines the model (GCN or GAT),</li>
            <li>trains it with early stopping,</li>
            <li>reports accuracy and generates plots.</li>
          </ul>
        </Section>

        {/* 7. Team */}
        <Section id="team" title="7. Team & Course Context">
          <p>
            This project was developed as part of the{" "}
            <strong>
              Data and Knowledge Representation – Machine Learning for Graphs
            </strong>{" "}
            course. The goal of the assignment is to create a tutorial-style
            case study that teaches how to apply state-of-the-art graph ML
            models to a real-world dataset using PyTorch Geometric.
          </p>

          <p>Team members and roles:</p>
          <ul>
            <li>
              <strong>Yasmina Mamadalieva</strong> — Graph Convolutional Network
              (GCN) implementation and experiments
            </li>
            <li>
              <strong>Sofa Goryunova</strong> — Graph Attention Network (GAT)
              implementation and experiments
            </li>
            <li>
              <strong>Ekaterina Akimenko</strong> — website design, comparison
              notebook, visualizations, and tutorial writing
            </li>
          </ul>

          <p>
            The final outcome is both a set of runnable notebooks and this
            interactive tutorial site, which together form a complete learning
            resource for graph-based paper classification using GCN and GAT.
          </p>
        </Section>
      </main>

      <footer className="footer">
        <p>
          © {new Date().getFullYear()} DKR Course — Graph Neural Networks
          Tutorial
        </p>
      </footer>
    </div>
  );
}

export default App;
