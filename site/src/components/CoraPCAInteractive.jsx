import React from "react";

export default function CoraPCAInteractive() {
  return (
    <div className="figure-wrapper">
      <div className="iframe-wrapper">
        <iframe
          src="/cora_pca_interactive.html"
          title="Cora PCA interactive visualization"
          className="interactive-iframe"
          loading="lazy"
        />
      </div>
      <p className="figure-caption">
        Interactive PCA visualization of Cora node features. Hover over a node
        to see its paper id, class index, and research topic. Edges are shown as
        faint gray lines in the background.
      </p>
    </div>
  );
}
