# causal-graph-rs

Rust port of [causal-graph](https://github.com/SuperInstance/causal-graph) — causal discovery and inference with PC algorithm, do-calculus, and counterfactual reasoning.

## Overview

- **CausalGraph**: DAG data structure with cycle detection, topological sort, ancestors/descendants
- **PCAlgorithm**: Causal discovery from observational data using conditional independence tests
- **d-separation**: Bayes-Ball algorithm for checking conditional independence in graphs
- **DoCalculus**: Intervention operations (do-operator), backdoor criterion
- **Counterfactual**: Counterfactual reasoning on structural causal models

## Usage

```rust
use causal_graph::{CausalGraph, Node, Edge, PCAlgorithm, DataSet, d_separated, DoCalculus, Intervention};

// Build a causal graph
let mut g = CausalGraph::new();
g.add_node(Node::new("Treatment"));
g.add_node(Node::new("Outcome"));
g.add_node(Node::new("Confounder"));
g.add_edge(Edge::new("Confounder", "Treatment")).unwrap();
g.add_edge(Edge::new("Confounder", "Outcome")).unwrap();
g.add_edge(Edge::new("Treatment", "Outcome")).unwrap();

// D-separation
assert!(!d_separated(&g, "Treatment", "Outcome", &[])); // connected
assert!(d_separated(&g, "Treatment", "Outcome", &["Confounder"])); // blocked

// Intervention
let dc = DoCalculus::new(g);
let intervened = dc.intervene(&Intervention::do_set("Treatment"));
assert!(intervened.parents("Treatment").is_empty()); // parents removed

// PC algorithm discovery
let mut ds = DataSet::new();
// ... add columns ...
let pc = PCAlgorithm::new(0.05);
let discovered = pc.discover(&ds).unwrap();
```

## License

MIT
