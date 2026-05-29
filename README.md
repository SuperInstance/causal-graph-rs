# causal-graph-rs

Causal discovery and inference — DAG data structures, PC algorithm for observational discovery, d-separation via Bayes-Ball, do-calculus interventions, and counterfactual reasoning on structural causal models.

## What This Gives You

- **`CausalGraph`** — DAG with cycle detection, topological sort, ancestor/descendant queries
- **PC Algorithm** — Causal discovery from observational data using conditional independence tests
- **d-separation** — Bayes-Ball algorithm for checking conditional independence in graphs
- **Do-calculus** — Intervention operations (do-operator), backdoor criterion identification
- **Counterfactual reasoning** — Counterfactual queries on structural causal models (SCMs)

## Quick Start

### Build and query a causal graph

```rust
use causal_graph::{CausalGraph, Node, Edge, d_separated};

let mut g = CausalGraph::new();
g.add_node(Node::new("Treatment"));
g.add_node(Node::new("Outcome"));
g.add_node(Node::new("Confounder"));

g.add_edge(Edge::new("Confounder", "Treatment")).unwrap();
g.add_edge(Edge::new("Confounder", "Outcome")).unwrap();
g.add_edge(Edge::new("Treatment", "Outcome")).unwrap();

// d-separation: Treatment and Outcome connected unless we condition on Confounder
assert!(!d_separated(&g, "Treatment", "Outcome", &[]));
assert!(d_separated(&g, "Treatment", "Outcome", &["Confounder"]));
```

### Intervene with the do-operator

```rust
use causal_graph::{DoCalculus, Intervention};

let dc = DoCalculus::new(g);
let intervened = dc.intervene(&Intervention::do_set("Treatment"));
// do(Treatment) removes incoming edges to Treatment
assert!(intervened.parents("Treatment").is_empty());
```

### Discover causal structure from data

```rust
use causal_graph::PCAlgorithm;

let mut ds = DataSet::new();
// ... add observations ...
let pc = PCAlgorithm::new(0.05);  // significance level α
let discovered = pc.discover(&ds).unwrap();
```

## API Reference

### `CausalGraph`

| Method | Description |
|--------|-------------|
| `new()` | Empty DAG |
| `add_node(node)` | Add a variable |
| `add_edge(edge)` | Add causal edge (rejects cycles) |
| `topological_sort()` | Topological ordering |
| `ancestors(node)` / `descendants(node)` | Reachability |
| `parents(node)` / `children(node)` | Direct neighbors |

### `PCAlgorithm`

```rust
PCAlgorithm::new(alpha)           // set significance level
pc.discover(&data_set)            // learn DAG from observations
```

### `DoCalculus`

```rust
DoCalculus::new(graph)
dc.intervene(&Intervention::do_set(variable))  // cut incoming edges
```

## How It Fits

- **[conservation-protocol](https://github.com/SuperInstance/conservation-protocol)** — Spectral fingerprints provide the graph structure; causal-graph discovers the causal relationships
- **[cocapn-explain-rs](https://github.com/SuperInstance/cocapn-explain-rs)** — Feature importance explanations use causal structure to determine what's truly causal vs confounded
- **[constraint-dsl](https://github.com/SuperInstance/constraint-dsl)** — Constraint pipelines can encode causal structure as DAG dependencies

## Testing

29 tests covering DAG construction, cycle detection, topological sort, d-separation, do-operator interventions, PC algorithm discovery, and counterfactual queries.

```bash
cargo test
```

## Installation

```toml
[dependencies]
causal-graph = { git = "https://github.com/SuperInstance/causal-graph-rs" }
```

```bash
git clone https://github.com/SuperInstance/causal-graph-rs.git
cd causal-graph-rs
cargo build
```

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance) ecosystem.
