# causal-graph-rs

Rust port of [causal-graph](https://github.com/SuperInstance/causal-graph) — DAG-based causal graph.

## Features

- **Acyclic enforcement**: cycle detection prevents invalid edges
- **Topological sort**: Kahn's algorithm
- **Reachability queries**: BFS-based path detection
- **Ancestor/descendant queries**: transitive closure
- **Lowest common ancestor**: find LCA of two nodes
- Generic node data with zero dependencies

## Usage

```rust
use causal_graph::CausalGraph;

let mut g = CausalGraph::new();
let a = g.add_node("event_a");
let b = g.add_node("event_b");
let c = g.add_node("event_c");

g.add_edge(a, b);  // a causes b
g.add_edge(b, c);  // b causes c

assert!(g.reachable(a, c));    // transitive
assert!(!g.reachable(c, a));   // not reversible
assert!(!g.add_edge(c, a));    // would create cycle

let order = g.topological_sort();
```

## License

MIT
