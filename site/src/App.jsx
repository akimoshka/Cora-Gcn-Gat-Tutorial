import React from "react";
import Section from "./components/Section";
import MathBlock from "./components/MathBlock";

// Figures and interactives
import ClassDistrCora from "./assets/ClassDistrCora.png";
import InteractiveEmbeddings from "./components/InteractiveEmbeddings";
import CoraPCAInteractive from "./components/CoraPCAInteractive";
import GCNTrainingLoss from "./assets/GCNTrainingLoss.png";
import GCNValAcc from "./assets/GCNValAcc.png";
import GCNLearnedNodeEmb from "./assets/GCNLearnedNodeEmb.png";
import GATTrainingLoss from "./assets/GATTrainingLoss.png";
import GATValAcc from "./assets/GATValAcc.png";
import GATLearnedNodeEmb from "./assets/GATLearnedNodeEmb.png";

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
          <a href="#comparison">Comparison &amp; Results</a>
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
            We work with the Cora dataset and implement two fundamental GNN
            architectures:
          </p>
          <ul>
            <li>
              <strong>Graph Convolutional Network (GCN)</strong> - aggregates
              information from neighbors using a normalized graph convolution.
            </li>
            <li>
              <strong>Graph Attention Network (GAT)</strong> - extends GCN by
              learning <em>attention weights</em> for each edge, so some
              neighbors can be more important than others.
            </li>
          </ul>
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
            the topic label of each paper. We use{" "}
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
            Before training, we inspect the dataset statistically and visually.
            The figure below shows the class distribution over the seven topics.
            We can already see that classes are slightly imbalanced but all of
            them are reasonably represented, which makes accuracy a meaningful
            metric.
          </p>

          <div className="figure-wrapper">
            <img
              src={ClassDistrCora}
              alt="Class distribution in the Cora dataset"
              className="figure"
            />
            <p className="figure-caption">
              Figure: Class distribution in the Cora dataset. Each bar
              corresponds to one research topic.
            </p>
          </div>

          <p>
            Next, we project the high-dimensional node features onto two
            principal components (PCA) and plot the nodes in 2D, colored by
            class. This does not use the graph structure yet, but it gives a
            first impression of how separable the classes are based only on
            bag-of-words features.
          </p>

          <CoraPCAInteractive />

          <p>
            In the following sections we will see how GCN and GAT use both these
            features and the citation edges to construct more discriminative
            node embeddings.
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

          <p>A single GCN layer can be written as the following update rule:</p>

          <MathBlock>
            {
              "H^{(l+1)} = \\sigma\\left(\\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2} H^{(l)} W^{(l)}\\right)"
            }
          </MathBlock>

          <p>Here:</p>
          <ul>
            <li>
              <MathBlock>{"\\tilde{A} = A + I"}</MathBlock>
              <span>
                {" "}
                is the adjacency matrix with self-loops (each node is connected
                to itself),
              </span>
            </li>
            <li>
              <MathBlock>{"\\tilde{D}"}</MathBlock>
              <span> is the diagonal degree matrix of </span>
              <MathBlock>{"\\tilde{A}"}</MathBlock>
              <span>,</span>
            </li>
            <li>
              <MathBlock>{"H^{(l)}"}</MathBlock>
              <span>
                {" "}
                is the matrix of node features at layer <code>l</code>,
              </span>
            </li>
            <li>
              <MathBlock>{"W^{(l)}"}</MathBlock>
              <span> is a learnable weight matrix,</span>
            </li>
            <li>
              <MathBlock>{"\\sigma"}</MathBlock>
              <span> is a non-linear activation function such as ReLU.</span>
            </li>
          </ul>

          <p>
            The normalization{" "}
            <MathBlock>
              {"\\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2}"}
            </MathBlock>{" "}
            ensures that features are scaled in a way that stabilizes training
            and prevents high-degree nodes from dominating the aggregation.
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
            For Cora, we use a simple two-layer GCN with a hidden dimension of
            64 and early stopping based on validation accuracy. Despite its
            simplicity, this model already achieves strong performance, reaching
            a test accuracy of <strong>0.81</strong>. With early stopping, the
            training converged after <strong>46 epochs</strong>.
          </p>

          <div className="figure-grid">
            <div className="figure-wrapper">
              <img
                src={GCNTrainingLoss}
                alt="GCN training loss curve"
                className="figure"
              />
              <p className="figure-caption">
                GCN training loss over epochs. The loss decreases smoothly and
                stabilizes, which indicates that the model fits the training
                nodes without severe instability.
              </p>
            </div>

            <div className="figure-wrapper">
              <img
                src={GCNValAcc}
                alt="GCN validation accuracy curve"
                className="figure"
              />
              <p className="figure-caption">
                GCN validation accuracy curve. Early stopping selects the epoch
                with the highest validation accuracy to avoid overfitting.
              </p>
            </div>
          </div>

          <p>
            After training, we take the logits of the final layer and embed them
            into 2D using t-SNE. This allows us to visually inspect the
            separability of the classes in the learned representation space.
          </p>

          <div className="figure-wrapper">
            <img
              src={GCNLearnedNodeEmb}
              alt="GCN learned node embeddings visualized with t-SNE"
              className="figure"
            />
            <p className="figure-caption">
              t-SNE visualization of node embeddings learned by the GCN. Most
              classes form compact, well-separated clusters (for example, the
              red and yellow groups), and even the denser brown cluster remains
              relatively localized. This structure is consistent with the higher
              test accuracy of 0.81.
            </p>
          </div>
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

          <MathBlock>
            {
              "\\mathbf{h}'_i = \\sigma\\left( \\sum_{j \\in \\mathcal{N}(i)} \\alpha_{ij} \\, W \\, \\mathbf{h}_j \\right)"
            }
          </MathBlock>

          <p>where:</p>
          <ul>
            <li>
              <MathBlock>{"\\mathbf{h}_i"}</MathBlock>
              <span> is the input feature vector of node i,</span>
            </li>
            <li>
              <MathBlock>{"W"}</MathBlock>
              <span> is a learnable weight matrix,</span>
            </li>
            <li>
              <MathBlock>{"\\alpha_{ij}"}</MathBlock>
              <span>
                {" "}
                is the attention coefficient for edge (i, j), telling the model how much
                information to take from neighbor j.
              </span>
            </li>
          </ul>

          <p>The attention coefficients are computed as:</p>

          <MathBlock>
            {
              "\\alpha_{ij} = \\operatorname{softmax}_j\\left( \\operatorname{LeakyReLU}\\left( \\mathbf{a}^\\top [ W \\mathbf{h}_i \\Vert W \\mathbf{h}_j ] \\right) \\right)"
            }
          </MathBlock>

          <p>
            Here, <MathBlock>{"\\mathbf{a}"}</MathBlock> is a learnable vector, and{" "}
            <code>[· ∥ ·]</code> denotes concatenation. The softmax is taken
            over all neighbors of node <code>i</code>, so the attention weights
            for each node sum to 1. Intuitively, neighbors that are more
            relevant to predicting the label of node <code>i</code> get higher
            attention.
          </p>

          <p>
            In PyTorch Geometric, this mechanism is implemented in{" "}
            <code>GATConv</code>. A typical GAT model in our experiments looks
            like:
          </p>

          <pre>
            {`from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x_res = x  # optional residual
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_res.shape[1] == x.shape[1]:
            x = x + x_res

        x = self.gat2(x, edge_index)
        return x`}
          </pre>

          <p>
            We fix random seeds for reproducibility and train the GAT model
            using the Adam optimizer with learning rate <code>0.005</code> and
            weight decay <code>5e-4</code>. We also explore several
            configurations:
          </p>
          <ul>
            <li>hidden dimension values (4, 8),</li>
            <li>number of attention heads (4 and 8),</li>
            <li>learning rates (0.001 vs 0.005),</li>
            <li>weight decay (5e-4 vs 1e-3),</li>
            <li>
              early-stopping patience (10 vs 20 epochs based on validation
              accuracy).
            </li>
          </ul>

          <p>
            The final configuration is chosen by monitoring validation accuracy,
            selecting the model that generalizes best without overfitting. With
            this setup, GAT reaches a test accuracy of{" "}
            <strong>0.772</strong>. Early stopping halted training after{" "}
            <strong>18 epochs</strong>.
          </p>

          <div className="figure-grid">
            <div className="figure-wrapper">
              <img
                src={GATTrainingLoss}
                alt="GAT training loss curve"
                className="figure"
              />
              <p className="figure-caption">
                GAT training loss over epochs. The curve is slightly noisier
                than for GCN due to the added complexity of attention weights,
                but it still converges to a low loss.
              </p>
            </div>

            <div className="figure-wrapper">
              <img
                src={GATValAcc}
                alt="GAT validation accuracy curve"
                className="figure"
              />
              <p className="figure-caption">
                GAT validation accuracy curve. The best checkpoint (used for
                testing) is selected based on the peak validation accuracy.
              </p>
            </div>
          </div>

          <p>
            As with GCN, we visualize the final node embeddings via t-SNE,
            coloring nodes by their true label:
          </p>

          <div className="figure-wrapper">
            <img
              src={GATLearnedNodeEmb}
              alt="GAT learned node embeddings visualized with t-SNE"
              className="figure"
            />
            <p className="figure-caption">
              t-SNE visualization of node embeddings learned by the GAT model.
              Clusters are still clearly present, but some regions (especially
              the brown class) spread over a larger area and overlap more with
              other colors than in the GCN plot. This increased overlap is
              consistent with the slightly lower test accuracy of 0.772.
            </p>
          </div>
        </Section>

        {/* 5. Comparison & Results */}
        <Section
          id="comparison"
          title="5. Experimental Setup, Results & Discussion"
        >
          <p>
            To compare GCN and GAT fairly, we keep the experimental setup as
            similar as possible:
          </p>

          <ul>
            <li>Same Cora dataset and train/val/test split (from Planetoid)</li>
            <li>Same optimizer family (Adam) with comparable learning rates</li>
            <li>Early stopping based on validation accuracy</li>
            <li>Classification accuracy on the test set as the main metric</li>
          </ul>

          <p>
            The table below summarizes the final test accuracies obtained with
            the best configuration for each model:
          </p>

          <table className="results-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Test Accuracy</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>GCN (2 layers)</td>
                <td>0.81</td>
              </tr>
              <tr>
                <td>GAT</td>
                <td>0.772</td>
              </tr>
            </tbody>
          </table>

          <p>
            Visually, the GCN embedding (Figure in the GCN section) shows more
            compact clusters with cleaner boundaries between topics, whereas the
            GAT embedding exhibits larger mixed-color regions where several
            classes overlap. This suggests that on Cora, the uniform neighborhood
            aggregation of GCN is already well aligned with the label structure,
            and the extra flexibility from attention does not translate into
            better separation.
          </p>

          <p>
            There are several plausible explanations for why the simpler
            two-layer GCN slightly outperforms the more expressive GAT model on
            Cora:
          </p>
          <ul>
            <li>
              The Cora dataset is relatively small and well-structured. A simple
              convolutional architecture is already strong enough to exploit the
              graph structure without needing attention.
            </li>
            <li>
              GAT introduces many additional parameters (per-head projections and
              attention vectors), which makes it more prone to overfitting in
              low-data regimes.
            </li>
            <li>
              Optimization for attention-based models can be more delicate; the
              best configuration may require more careful tuning or additional
              regularization.
            </li>
          </ul>

          <p>
            Overall, this case study illustrates an important lesson in graph
            machine learning: more complex architectures such as GAT are not
            guaranteed to win on every benchmark. On small, clean citation
            graphs like Cora, a well-regularized GCN can be a very strong and
            robust baseline.
          </p>

          <InteractiveEmbeddings />
        </Section>

        {/* 6. Colab */}
        <Section id="colab" title="6. Run the Code in Google Colab">
          <p>
            All experiments in this tutorial are implemented as Jupyter
            notebooks. To make them easy to run and reproduce, we provide
            Google Colab links:
          </p>

          <ul>
            <li>
              <a
                href="https://colab.research.google.com/drive/1Jv7DPatVyO61ydvaOjDJAkjiowV_aKNx?authuser=0#scrollTo=dAIZ3kkY3lTE"
                target="_blank"
                rel="noreferrer"
              >
                GCN on Cora — Colab
              </a>
            </li>
            <li>
              <a
                href="https://colab.research.google.com/drive/1y2B-9Dsn8JBB8q5jR75p1GInkFc-7i8i?usp=sharing#scrollTo=BH5iYWvhLnBN"
                target="_blank"
                rel="noreferrer"
              >
                GAT on Cora — Colab
              </a>
            </li>
            <li>
              <a
                href="https://colab.research.google.com/drive/1ROgAUnXgXtReR3-tfGvHZkIC9iUdAq03?usp=sharing"
                target="_blank"
                rel="noreferrer"
              >
                GCN vs GAT comparison — Colab
              </a>
            </li>
          </ul>

          <p>Each notebook:</p>
          <ul>
            <li>installs the correct PyTorch and PyG versions,</li>
            <li>loads the Cora dataset,</li>
            <li>defines the model (GCN or GAT),</li>
            <li>trains it with early stopping,</li>
            <li>reports accuracy and generates visualizations.</li>
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
              implementation, experiments, and attention analysis
            </li>
            <li>
              <strong>Ekaterina Akimenko</strong> — Website design, comparison
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
