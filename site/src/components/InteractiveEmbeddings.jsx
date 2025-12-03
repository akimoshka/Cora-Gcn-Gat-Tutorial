import React from "react";

export default function InteractiveEmbeddings() {
  return (
    <section className="interactive-section">
      <h3 className="section-subtitle">
        Interactive t-SNE embeddings: GCN vs GAT
      </h3>
      <p>
        Use the buttons in the top panel to switch between GCN and GAT. Each
        point is a paper in the Cora graph, colored by its ground-truth topic.
      </p>
      <div className="iframe-wrapper">
        <iframe
          src="/gcn_gat_tsne_interactive.html"
          title="GCN vs GAT t-SNE embeddings on Cora"
          className="interactive-iframe"
        />
      </div>
    </section>
  );
}
