import React from "react";
import Section from "./components/Section";
import MathBlock from "./components/MathBlock";

// Figures and interactives (Cora)
import ClassDistrCora from "./assets/ClassDistrCora.png";
import InteractiveEmbeddings from "./components/InteractiveEmbeddings";
import CoraPCAInteractive from "./components/CoraPCAInteractive";
import GCNTrainingLoss from "./assets/GCNTrainingLoss.png";
import GCNValAcc from "./assets/GCNValAcc.png";
import GCNLearnedNodeEmb from "./assets/GCNLearnedNodeEmb.png";
import GATTrainingLoss from "./assets/GATTrainingLoss.png";
import GATValAcc from "./assets/GATValAcc.png";
import GATLearnedNodeEmb from "./assets/GATLearnedNodeEmb.png";

// New figures (Coauthor-CS models)
import GATv2TrainingLoss from "./assets/GATv2TrainingLoss.png";
import GATv2ValAcc from "./assets/GATv2ValAcc.png";
import GATv2LearnedNodeEmb from "./assets/GATv2LearnedNodeEmb.png";

import GraphSAGETrainingLoss from "./assets/GraphSAGETrainingLoss.png";
import GraphSAGEValAcc from "./assets/GraphSAGEValAcc.png";
import GraphSAGELearnedNodeEmb from "./assets/GraphSAGELearnedNodeEmb.png";

import APPNPTrainingLoss from "./assets/APPNPTrainingLoss.png";
import APPNPValAcc from "./assets/APPNPValAcc.png";
import APPNPLearnedNodeEmb from "./assets/APPNPLearnedNodeEmb.png";

function App() {
  return (
    <div className="app">
      <header className="hero">
        <h1 className="title-main">
          Applying Graph Neural Networks to Paper and Author Classification in
          Citation and Co-Authorship Networks
        </h1>

        <h2 className="title-sub">
          Data and Knowledge Representation – Machine Learning for Graphs
        </h2>

        <h3 className="authors">
          By: Yasmina Mamadalieva, Ekaterina Akimenko, Sofa Goryunova
        </h3>

        <nav className="nav">
          <a href="#intro">Intro</a>
          <a href="#cora">Cora Dataset</a>
          <a href="#gcn">GCN</a>
          <a href="#gat">GAT</a>
          <a href="#coauthor">Coauthor-CS Dataset</a>
          <a href="#gatv2">GAT+</a>
          <a href="#graphsage">GraphSAGE</a>
          <a href="#appnp">APPNP</a>
          <a href="#comparison">Comparison &amp; Results</a>
          <a href="#colab">Colab</a>
          <a href="#team">Team</a>
          <a href="#references">References</a>
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
              papers or authors),
            </li>
            <li>
              <strong>Edges</strong> represent relationships (here: citation
              links or co-authorship),
            </li>
            <li>
              <strong>Node features</strong> describe each entity (here:
              bag-of-words features of paper text or publication keywords).
            </li>
          </ul>

          <p>
            We start from a classic benchmark problem:{" "}
            <strong>scientific paper classification in a citation network</strong>{" "}
            using the Cora dataset. Then we extend our study to a larger,
            more realistic dataset –{" "}
            <strong>Coauthor-CS</strong>, where nodes are authors and edges are
            co-authorship links.
          </p>

          <p className="emphasis">
            Our goal is to compare several GNN architectures on these graphs and
            understand when more complex models (attention, propagation,
            inductive aggregation) actually help.
          </p>

          <p>
            On Cora we implement two fundamental GNN architectures:
          </p>
          <ul>
            <li>
              <strong>Graph Convolutional Network (GCN)</strong> – aggregates
              information from neighbors using a normalized graph convolution.
            </li>
            <li>
              <strong>Graph Attention Network (GAT)</strong> – extends GCN by
              learning <em>attention weights</em> for each edge, so some
              neighbors can be more important than others.
            </li>
          </ul>

          <p>
            On Coauthor-CS we go beyond the scope of the course and evaluate
            three additional models:
          </p>
          <ul>
            <li>
              <strong>GATv2 (GAT+)</strong> – a more expressive version of GAT
              with improved attention formulation,
            </li>
            <li>
              <strong>GraphSAGE</strong> – an inductive architecture based on
              neighborhood sampling and aggregation,
            </li>
            <li>
              <strong>APPNP</strong> – a model that combines an MLP with
              personalized PageRank propagation.
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
              style={{ width: "90%", maxWidth: "1200px" }}
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
                is the attention coefficient for edge (i, j), telling the model
                how much information to take from neighbor j.
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
            configurations (hidden dimension, heads, learning rate, weight
            decay, early-stopping patience) and select the best one based on
            validation accuracy.
          </p>

          <p>
            With this setup on Cora, GAT reaches a test accuracy of{" "}
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

        {/* 5. Coauthor-CS Dataset */}
        <Section id="coauthor" title="5. The Coauthor-CS Dataset">
          <p>
            To go beyond the original assignment scope and test more expressive
            GNN models, we turned to a larger benchmark:{" "}
            <strong>Coauthor-CS</strong>. This is a co-authorship graph derived
            from Microsoft Academic Graph, where:
          </p>
          <ul>
            <li>
              <strong>Nodes:</strong> 18,333 authors
            </li>
            <li>
              <strong>Edges:</strong> 81,894 co-authorship links
            </li>
            <li>
              <strong>Node features:</strong> 6,805-dimensional bag-of-words
              vectors built from paper keywords
            </li>
            <li>
              <strong>Labels:</strong> 15 research subfields in computer
              science
            </li>
          </ul>

          <p>
            The task is again <strong>node classification</strong>: predict the
            main research area of each author. Unlike Cora, Coauthor-CS does not
            come with predefined train/validation/test masks, so we construct a
            random split with 10% training, 10% validation and 80% test nodes.
          </p>

          <p>
            The graph exhibits strong <strong>homophily</strong>: authors in the
            same research area tend to publish together, which is a favourable
            setting for neighborhood-aggregation GNNs.
          </p>

          <p>
            To understand the structure of the data, we first visualize node
            features projected onto two dimensions using PCA and UMAP. Below are
            interactive plots, where each point is an author colored by their
            research field.
          </p>

          <div className="interactive-wrapper">
            <iframe
              src="/graph_pca_interactive.html"
              title="Coauthor-CS PCA visualization"
              className="interactive-frame"
            />
          </div>

          <div className="interactive-wrapper">
            <iframe
              src="/graph_umap_interactive.html"
              title="Coauthor-CS UMAP visualization"
              className="interactive-frame"
            />
          </div>


          <p>
            Coauthor-CS is substantially larger and more complex than Cora. This
            makes it a good testbed for modern architectures such as GATv2,
            GraphSAGE, and APPNP, which are designed to handle larger graphs and
            more realistic neighborhood patterns.
          </p>
        </Section>

        {/* 6. GAT+ (GATv2) */}
        <Section id="gatv2" title="6. GAT+ (GATv2) on Coauthor-CS">
          <p>
            <strong>GATv2</strong> (often referred to as GAT+) is an improved
            version of the original Graph Attention Network. The key idea is to
            make the attention mechanism <strong>more expressive</strong> by
            changing where the non-linearity is applied.
          </p>

          <p>
            In the original GAT, the attention coefficient between nodes{" "}
            <MathBlock>{"i"}</MathBlock> and <MathBlock>{"j"}</MathBlock> is:
          </p>

          <MathBlock>
            {
              "\\alpha_{ij} = \\operatorname{softmax}_j\\big(\\operatorname{LeakyReLU}(\\mathbf{a}^\\top [W\\mathbf{h}_i \\Vert W\\mathbf{h}_j])\\big)"
            }
          </MathBlock>

          <p>
            In GATv2, the order of operations is changed so that the non-linearity
            is applied <em>after</em> combining the features:
          </p>

          <MathBlock>
            {
              "\\alpha_{ij} = \\operatorname{softmax}_j\\big(\\mathbf{a}^\\top \\operatorname{LeakyReLU}(W[\\mathbf{h}_i \\Vert \\mathbf{h}_j])\\big)"
            }
          </MathBlock>

          <p>
            This subtle change makes the attention mechanism{" "}
            <strong>universally expressive</strong> and fully differentiable with
            respect to the input features. In practice, it allows GATv2 to learn
            more complex patterns of importance across edges.
          </p>

          <p>We start by loading Coauthor-CS and creating our own masks:</p>

          <pre>{`from torch_geometric.datasets import Coauthor
        import torch_geometric.transforms as T
        import numpy as np
        import torch

        # Load and normalize features
        dataset = Coauthor(root="data/", name="CS", transform=T.NormalizeFeatures())
        data = dataset[0]

        def create_masks(data, train_ratio=0.1, val_ratio=0.1, seed=42):
            num_nodes = data.num_nodes
            num_train = int(num_nodes * train_ratio)
            num_val = int(num_nodes * val_ratio)
            num_test = num_nodes - num_train - num_val

            indices = np.random.RandomState(seed).permutation(num_nodes)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[indices[:num_train]] = True
            val_mask[indices[num_train:num_train+num_val]] = True
            test_mask[indices[num_train+num_val:]] = True

            return train_mask, val_mask, test_mask

        train_mask, val_mask, test_mask = create_masks(data)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask`}</pre>

          <p>
            The core of GATv2 is implemented using <code>GATv2Conv</code>. Below is
            the exact model we used:
          </p>

          <pre>{`import torch.nn.functional as F
        from torch_geometric.nn import GATv2Conv

        class GATv2(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels,
                        heads=8, dropout=0.6):
                super().__init__()
                self.dropout = dropout

                # First attention layer (multi-head)
                self.gat1 = GATv2Conv(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout
                )

                # Second attention layer (single head, no concat)
                self.gat2 = GATv2Conv(
                    hidden_channels * heads,
                    out_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout
                )

            def forward(self, x, edge_index):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.elu(self.gat1(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.gat2(x, edge_index)
                return x`}</pre>

          <p>
            Training is standard supervised node classification with cross-entropy
            on the labeled nodes, Adam optimizer, and early stopping:
          </p>

          <pre>{`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GATv2(
            in_channels=dataset.num_features,
            hidden_channels=8,
            out_channels=dataset.num_classes,
            heads=8,
            dropout=0.3
        ).to(device)

        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        best_val_acc = 0.0
        patience = 20
        counter = 0

        for epoch in range(200):
            # Train step
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation step
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break`}</pre>

          <p>Our final configuration on Coauthor-CS is:</p>
          <ul>
            <li>2 layers with <code>GATv2Conv</code></li>
            <li>Hidden dimension = 8, attention heads = 8 in the first layer</li>
            <li>Dropout = 0.3–0.6</li>
            <li>Adam optimizer, learning rate = 0.01, weight decay = 1e-4</li>
            <li>Early stopping based on validation accuracy (patience = 20)</li>
          </ul>

          <p>
            Early stopping occurred at epoch <strong>166</strong>. The final
            performance on the test set is:
          </p>

          <p className="emphasis">
            <strong>Test accuracy (GATv2 on Coauthor-CS): 0.919</strong>
          </p>

          <div className="figure-grid">
            <div className="figure-wrapper">
              <img
                src={GATv2TrainingLoss}
                alt="GATv2 training loss curve"
                className="figure"
              />
              <p className="figure-caption">
                GATv2 training loss on Coauthor-CS. The curve is noisier than on
                Cora due to the larger graph and more complex attention
                mechanism, but it still converges steadily.
              </p>
            </div>
            <div className="figure-wrapper">
              <img
                src={GATv2ValAcc}
                alt="GATv2 validation accuracy curve"
                className="figure"
              />
              <p className="figure-caption">
                GATv2 validation accuracy over epochs. The best checkpoint is
                selected based on the maximum validation accuracy before early
                stopping.
              </p>
            </div>
          </div>
        </Section>


        {/* 7. GraphSAGE */}
        <Section id="graphsage" title="7. GraphSAGE on Coauthor-CS">
          <p>
            <strong>GraphSAGE</strong> (Sample and Aggregate) is an{" "}
            <strong>inductive</strong> GNN architecture designed to generalize to
            unseen nodes. Instead of relying on the full graph during training, it
            learns a <em>neighborhood aggregation function</em> that can later be
            applied to new graphs or new nodes.
          </p>

          <p>The basic update rule has two steps:</p>

          <p>
            <strong>1. Neighborhood aggregation</strong>
          </p>
          <MathBlock>
            {
              "h_{\\mathcal{N}(i)}^{(l+1)} = \\text{aggregate}\\left( \\{ h_j^{(l)} : j \\in \\mathcal{N}(i) \\} \\right)"
            }
          </MathBlock>

          <p>
            <strong>2. Node update</strong>
          </p>
          <MathBlock>
            {
              "h_i^{(l+1)} = \\sigma\\left( W \\cdot [h_i^{(l)} \\Vert h_{\\mathcal{N}(i)}^{(l+1)}] \\right)"
            }
          </MathBlock>

          <p>
            In our code we first load Coauthor-CS and create train/val/test masks,
            just like for GATv2:
          </p>

          <pre>{`from torch_geometric.datasets import Coauthor
        import numpy as np
        import torch

        dataset = Coauthor(root="/tmp/CS", name="CS")
        data = dataset[0]

        def create_masks(data, train_ratio=0.1, val_ratio=0.1, seed=42):
            num_nodes = data.num_nodes
            num_train = int(num_nodes * train_ratio)
            num_val = int(num_nodes * val_ratio)
            num_test = num_nodes - num_train - num_val

            indices = np.random.RandomState(seed).permutation(num_nodes)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[indices[:num_train]] = True
            val_mask[indices[num_train:num_train+num_val]] = True
            test_mask[indices[num_train+num_val:]] = True

            return train_mask, val_mask, test_mask

        train_mask, val_mask, test_mask = create_masks(data)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask`}</pre>

          <p>
            The GraphSAGE model itself is very compact when using{" "}
            <code>SAGEConv</code> from PyTorch Geometric:
          </p>

          <pre>{`from torch_geometric.nn import SAGEConv
        import torch.nn.functional as F

        class GraphSAGE(torch.nn.Module):
            def __init__(self, hidden_channels=64):
                super().__init__()
                # First layer: from raw features to hidden space
                self.conv1 = SAGEConv(dataset.num_features, hidden_channels)
                # Second layer: from hidden space to class logits
                self.conv2 = SAGEConv(hidden_channels, dataset.num_classes)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index)
                return x`}</pre>

          <p>
            Training again uses cross-entropy on labeled nodes, Adam optimizer, and
            early stopping:
          </p>

          <pre>{`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GraphSAGE().to(device)
        data = data.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=5e-4
        )

        best_val_acc = 0.0
        patience = 20
        counter = 0

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)

            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                val_acc = (
                    (pred[data.val_mask] == data.y[data.val_mask]).sum()
                    / data.val_mask.sum()
                ).item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break`}</pre>

          <p>Our GraphSAGE configuration on Coauthor-CS:</p>
          <ul>
            <li>2 layers with <code>SAGEConv</code> (mean aggregation)</li>
            <li>Hidden dimension = 64</li>
            <li>ReLU activation after the first layer</li>
            <li>Dropout for regularization</li>
            <li>Adam optimizer, learning rate = 0.01, weight decay = 5e-4</li>
            <li>Early stopping with patience = 20</li>
          </ul>

          <p>
            Early stopping triggered at epoch <strong>100</strong>. The final
            performance is:
          </p>

          <p className="emphasis">
            <strong>Test accuracy (GraphSAGE on Coauthor-CS): 0.93</strong>
          </p>

          <div className="figure-grid">
            <div className="figure-wrapper">
              <img
                src={GraphSAGETrainingLoss}
                alt="GraphSAGE training loss"
                className="figure"
              />
              <p className="figure-caption">
                GraphSAGE training loss on Coauthor-CS. The loss curve is smooth
                and shows no signs of severe overfitting.
              </p>
            </div>
            <div className="figure-wrapper">
              <img
                src={GraphSAGEValAcc}
                alt="GraphSAGE validation accuracy"
                className="figure"
              />
              <p className="figure-caption">
                GraphSAGE validation accuracy over epochs. The best checkpoint is
                chosen via early stopping.
              </p>
            </div>
          </div>
        </Section>


        {/* 8. APPNP */}
        <Section id="appnp" title="8. APPNP: Predict then Propagate">
          <p>
            <strong>APPNP</strong> (Approximate Personalized Propagation of Neural
            Predictions) separates <em>feature transformation</em> from{" "}
            <em>graph propagation</em>. It first applies a small neural network
            (e.g. an MLP) to compute initial logits and then performs several
            rounds of personalized PageRank-style propagation.
          </p>

          <p>Conceptually, APPNP works in two stages:</p>

          <ol>
            <li>
              <strong>Prediction:</strong> compute initial node predictions{" "}
              <MathBlock>{"Z = f_\\theta(X)"}</MathBlock>
            </li>
            <li>
              <strong>Propagation:</strong> diffuse these predictions over the
              graph using a personalized PageRank scheme
            </li>
          </ol>

          <MathBlock>
            {
              "H^{(0)} = Z, \\quad H^{(k+1)} = (1 - \\alpha) \\tilde{A} H^{(k)} + \\alpha Z"
            }
          </MathBlock>

          <p>
            Here <MathBlock>{"\\alpha"}</MathBlock> is the teleport probability and{" "}
            <MathBlock>{"\\tilde{A}"}</MathBlock> is the normalized adjacency
            matrix with self-loops. This recurrence allows information to travel
            over many hops while still keeping the original prediction{" "}
            <MathBlock>{"Z"}</MathBlock> in the mixture, which helps avoid
            oversmoothing.
          </p>

          <p>We again start by creating masks on Coauthor-CS:</p>

          <pre>{`from torch_geometric.datasets import Coauthor
        import numpy as np
        import torch

        dataset = Coauthor(root="/tmp/CS", name="CS")
        data = dataset[0]

        def create_masks(data, train_ratio=0.1, val_ratio=0.1, seed=42):
            num_nodes = data.num_nodes
            num_train = int(num_nodes * train_ratio)
            num_val = int(num_nodes * val_ratio)
            num_test = num_nodes - num_train - num_val

            rng = np.random.RandomState(seed)
            indices = rng.permutation(num_nodes)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[indices[:num_train]] = True
            val_mask[indices[num_train:num_train+num_val]] = True
            test_mask[indices[num_train+num_val:]] = True

            return train_mask, val_mask, test_mask

        train_mask, val_mask, test_mask = create_masks(data)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask`}</pre>

          <p>
            The APPNP model is implemented as a two-layer MLP followed by an{" "}
            <code>APPNP</code> propagation layer from PyG:
          </p>

          <pre>{`import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import APPNP

        class APPNPNet(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels,
                        K=10, alpha=0.1, dropout=0.5):
                super().__init__()
                # MLP part
                self.lin1 = nn.Linear(in_channels, hidden_channels)
                self.lin2 = nn.Linear(hidden_channels, out_channels)
                # Propagation part (personalized PageRank)
                self.prop = APPNP(K=K, alpha=alpha)
                self.dropout = dropout

            def forward(self, x, edge_index):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.lin1(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.lin2(x)
                # APPNP propagation step
                x = self.prop(x, edge_index)
                return x`}</pre>

          <p>Training again follows the same pattern:</p>

          <pre>{`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)

        model = APPNPNet(
            in_channels=dataset.num_features,
            hidden_channels=64,
            out_channels=dataset.num_classes,
            K=10,
            alpha=0.1,
            dropout=0.5,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        best_val_acc = 0.0
        patience = 20
        counter = 0

        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)

            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
                val_total = data.val_mask.sum()
                val_acc = (val_correct.float() / val_total).item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break`}</pre>

          <p>
            We train APPNP on Coauthor-CS with standard cross-entropy loss and
            early stopping based on validation accuracy. As in GraphSAGE, we use a
            10%/10%/80% train/validation/test split.
          </p>

          <p>
            Early stopping occurs at epoch <strong>58</strong>. The final test
            accuracy is:
          </p>

          <p className="emphasis">
            <strong>Test accuracy (APPNP on Coauthor-CS): 0.9289</strong>
          </p>

          <div className="figure-grid">
            <div className="figure-wrapper">
              <img
                src={APPNPTrainingLoss}
                alt="APPNP training loss"
                className="figure"
              />
              <p className="figure-caption">
                APPNP training loss on Coauthor-CS. The loss decreases
                consistently until early stopping.
              </p>
            </div>
            <div className="figure-wrapper">
              <img
                src={APPNPValAcc}
                alt="APPNP validation accuracy"
                className="figure"
              />
              <p className="figure-caption">
                APPNP validation accuracy over epochs. The selected checkpoint
                corresponds to the peak validation performance.
              </p>
            </div>
          </div>
        </Section>


        {/* 9. Comparison & Results */}
        <Section
          id="comparison"
          title="9. Experimental Setup, Results & Discussion"
        >
          <p>
            To compare the models fairly, we keep the experimental setup as
            consistent as possible within each dataset:
          </p>

          <ul>
            <li>Fixed random seeds for reproducibility</li>
            <li>Same train/validation/test splits per dataset</li>
            <li>Adam optimizer with tuned but comparable learning rates</li>
            <li>Early stopping based on validation accuracy</li>
            <li>
              <strong>Cross-entropy loss</strong> as the main objective for
              multi-class node classification
            </li>
          </ul>

          <h3>Cora (Citation Network)</h3>
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
            Visually, the GCN embedding on Cora shows more compact clusters with
            cleaner boundaries between topics, whereas the GAT embedding exhibits
            larger mixed-color regions where several classes overlap. This
            suggests that on Cora, the uniform neighborhood aggregation of GCN is
            already well aligned with the label structure, and the extra
            flexibility from attention does not translate into better separation.
          </p>

          <h3>Coauthor-CS (Co-authorship Network)</h3>
          <table className="results-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Test Accuracy</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>GATv2 (GAT+)</td>
                <td>0.919</td>
              </tr>
              <tr>
                <td>GraphSAGE</td>
                <td>0.93</td>
              </tr>
              <tr>
                <td>APPNP</td>
                <td>0.9289</td>
              </tr>
            </tbody>
          </table>

          <p>
            On Coauthor-CS, all three advanced architectures perform very well,
            but their strengths are slightly different:
          </p>
          <ul>
            <li>
              <strong>GraphSAGE</strong> achieves the highest test accuracy
              (0.93). The mean aggregator is a good match for the strong
              homophily in co-authorship graphs, and the model is relatively
              simple and stable to optimize.
            </li>
            <li>
              <strong>APPNP</strong> is only slightly behind (0.9289). Its
              propagation step effectively spreads label information over longer
              distances while keeping the original MLP predictions in the mix.
            </li>
            <li>
              <strong>GATv2</strong> benefits from a more expressive attention
              mechanism and significantly outperforms vanilla GAT, but on this
              dataset it still remains slightly below GraphSAGE and APPNP.
            </li>
          </ul>

          <p>
            Overall, our experiments illustrate that the relative advantage of
            different GNN architectures is dataset-dependent. On a small, clean
            citation graph such as Cora, a well-regularized two-layer GCN is
            already a very strong baseline. On a larger and more realistic
            co-authorship graph such as Coauthor-CS, more advanced architectures
            like GraphSAGE, APPNP, and GATv2 can fully exploit the richer
            structure and achieve substantially higher accuracy.
          </p>

          <InteractiveEmbeddings />

          <div className="figure-grid figure-grid-large">
            <div className="figure-wrapper">
              <img
                src={GATv2LearnedNodeEmb}
                alt="GATv2 learned node embeddings (t-SNE, Coauthor-CS)"
                className="figure"
              />
              <p className="figure-caption">
                GATv2 on Coauthor-CS. The attention mechanism produces
                meaningful clusters but some boundaries remain fuzzy, especially
                in regions where several subfields interact.
              </p>
            </div>

            <div className="figure-wrapper">
              <img
                src={GraphSAGELearnedNodeEmb}
                alt="GraphSAGE learned node embeddings (t-SNE, Coauthor-CS)"
                className="figure"
              />
              <p className="figure-caption">
                GraphSAGE on Coauthor-CS. Clusters become tighter and more
                elongated along meaningful directions, reflecting the
                neighborhood aggregation that leverages homophily in
                co-authorship patterns.
              </p>
            </div>

            <div className="figure-wrapper">
              <img
                src={APPNPLearnedNodeEmb}
                alt="APPNP learned node embeddings (t-SNE, Coauthor-CS)"
                className="figure"
              />
              <p className="figure-caption">
                APPNP on Coauthor-CS. The clusters look particularly clean: many
                subfields occupy well-shaped regions with smooth boundaries and
                fewer “stray” points. The personalized PageRank propagation
                spreads information over longer paths while the teleport term
                keeps predictions anchored, which avoids over-smoothing and
                explains why APPNP visually looks the most structured, even
                though its accuracy is only slightly below GraphSAGE.
              </p>
            </div>
          </div>

        </Section>

        {/* 10. Colab */}
        <Section id="colab" title="10. Run the Code in Google Colab">
          <p>
            All experiments in this tutorial are implemented as Jupyter
            notebooks. To make them easy to run and reproduce, we provide Google
            Colab links:
          </p>

          <ul>
            <li>
              <a
                href="https://colab.research.google.com/drive/1Jv7DPatVyO61ydvaOjDJAkjiowV_aKNx?authuser=0#scrollTo=dAIZ3kkY3lTE"
                target="_blank"
                rel="noreferrer"
              >
                GCN on Cora – Colab (runs in about 1–2 minutes)
              </a>
            </li>
            <li>
              <a
                href="https://colab.research.google.com/drive/1y2B-9Dsn8JBB8q5jR75p1GInkFc-7i8i?usp=sharing#scrollTo=BH5iYWvhLnBN"
                target="_blank"
                rel="noreferrer"
              >
                GAT on Cora – Colab (runs in about 1–2 minutes)
              </a>
            </li>
            <li>
              <a
                href="https://colab.research.google.com/drive/1ROgAUnXgXtReR3-tfGvHZkIC9iUdAq03?usp=sharing"
                target="_blank"
                rel="noreferrer"
              >
                GCN vs GAT comparison on Cora – Colab (includes interactive
                visualizations, takes longer)
              </a>
            </li>
            <li>
              <a
                href="https://colab.research.google.com/drive/1SWOraiV1drvd3f5NE13uYjuKTwg4J7XY?usp=sharing"
                target="_blank"
                rel="noreferrer"
              >
                GATv2 (GAT+) on Coauthor-CS – Colab
              </a>
            </li>
            <li>
              <a
                href="https://colab.research.google.com/drive/1XhQG5dKLJH0jmxToxfsL51Ik0XwvEwbM?usp=sharing"
                target="_blank"
                rel="noreferrer"
              >
                GraphSAGE on Coauthor-CS – Colab
              </a>
            </li>
            <li>
              <a
                href="https://colab.research.google.com/drive/1MqEnp4woLOKpCZgZy2Z8_nyhcbkeHnfW?usp=sharing"
                target="_blank"
                rel="noreferrer"
              >
                APPNP on Coauthor-CS – Colab
              </a>
            </li>
          </ul>

          <p>Each notebook:</p>
          <ul>
            <li>installs the correct PyTorch and PyG versions,</li>
            <li>loads the corresponding dataset (Cora or Coauthor-CS),</li>
            <li>defines the model architecture,</li>
            <li>trains it with early stopping,</li>
            <li>reports accuracy and generates visualizations.</li>
          </ul>
        </Section>

        {/* 11. Team */}
        <Section id="team" title="11. Team & Course Context">
          <p>
            This project was developed as part of the{" "}
            <strong>
              Data and Knowledge Representation – Machine Learning for Graphs
            </strong>{" "}
            course. The goal of the assignment is to create a tutorial-style
            case study that teaches how to apply state-of-the-art graph ML
            models to real-world datasets using PyTorch Geometric.
          </p>

          <p>Team members and roles:</p>
          <ul>
            <li>
              <strong>Yasmina Mamadalieva</strong> – Graph Convolutional Network
              (GCN) implementation and experiments on Cora, Coauthor-CS experiments with GraphSAGE.
            </li>
            <li>
              <strong>Sofa Goryunova</strong> – Graph Attention Network (GAT),
              Coauthor-CS experiments with GAT+, and attention analysis.
            </li>
            <li>
              <strong>Ekaterina Akimenko</strong> – Website design, comparison
              notebooks, Coauthor-CS experiments with APPNP, and
              visualization/tutorial writing.
            </li>
          </ul>

          <p>
            The final outcome is both a set of runnable notebooks and this
            interactive tutorial site, which together form a complete learning
            resource for graph-based node classification on citation and
            co-authorship networks.
          </p>
        </Section>

        {/* 12. References */}
        <Section id="references" title="12. References">
          <ul className="references-list">
            <li>
              T. N. Kipf and M. Welling, &quot;Semi-Supervised Classification
              with Graph Convolutional Networks,&quot; ICLR, 2017.
            </li>
            <li>
              P. Veličković et al., &quot;Graph Attention Networks,&quot; ICLR,
              2018.
            </li>
            <li>
              S. Brody, U. Alon, and E. Yahav, &quot;How Attentive are Graph
              Attention Networks?&quot; ICLR, 2022. (GATv2)
            </li>
            <li>
              W. Hamilton, Z. Ying, and J. Leskovec, &quot;Inductive
              Representation Learning on Large Graphs,&quot; NIPS, 2017.
              (GraphSAGE)
            </li>
            <li>
              J. Klicpera, A. Bojchevski, and S. Günnemann, &quot;Predict then
              Propagate: Graph Neural Networks meet Personalized PageRank,&quot;
              ICLR, 2019. (APPNP)
            </li>
            <li>PyTorch Geometric documentation and dataset descriptions.</li>
          </ul>
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
