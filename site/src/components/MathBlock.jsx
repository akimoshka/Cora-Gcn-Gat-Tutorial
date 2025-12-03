import React from "react";
import { BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

export default function MathBlock({ children }) {
  return (
    <div className="formula">
      <BlockMath math={children} />
    </div>
  );
}
