use crate::graph::CausalGraph;
use std::collections::HashSet;

/// Check if X and Y are d-separated given Z in the graph.
///
/// D-separation: X ⊥ Y | Z iff every path between X and Y is blocked.
/// A path is blocked if it contains:
/// - A chain A → B → C where B ∈ Z
/// - A fork A ← B → C where B ∈ Z
/// - A collider A → B ← C where B ∉ Z and no descendant of B is in Z
pub fn d_separated(graph: &CausalGraph, x: &str, y: &str, z: &[&str]) -> bool {
    let z_set: HashSet<&str> = z.iter().copied().collect();

    // Use the Bayes-Ball algorithm
    let mut visited = HashSet::new();
    let mut reachable = HashSet::new();

    // Start from x, enqueue both directions
    let mut queue: Vec<(&str, bool)> = Vec::new();
    // Enqueue parents (going backward from x) and children (going forward from x)
    for parent in graph.parents(x) {
        queue.push((parent, false));
    }
    for child in graph.children(x) {
        queue.push((child, true));
    }
    while let Some((node, forward)) = queue.pop() {
        let state = (node.to_string(), forward);
        if visited.contains(&state) {
            continue;
        }
        visited.insert(state.clone());

        if node != x {
            reachable.insert(node.to_string());
        }

        if forward {
            // Arrived at node from a parent (going downstream)
            if !z_set.contains(node) {
                // Not observed: continue downstream to children
                for child in graph.children(node) {
                    queue.push((child, true));
                }
            } else {
                // Observed: go upstream to parents (explaining away / collider activation)
                for parent in graph.parents(node) {
                    queue.push((parent, false));
                }
            }
        } else {
            // Arrived at node from a child (going upstream)
            if !z_set.contains(node) {
                // Not observed: continue upstream to parents AND downstream to children (fork)
                for parent in graph.parents(node) {
                    queue.push((parent, false));
                }
                for child in graph.children(node) {
                    queue.push((child, true));
                }
            } else {
                // Observed: block (fork is blocked when center is observed)
                // Do not continue in any direction
            }
        }
    }

    !reachable.contains(y)
}

/// Causal inference engine for answering queries about a causal graph.
#[derive(Debug, Clone)]
pub struct InferenceEngine {
    pub graph: CausalGraph,
}

impl InferenceEngine {
    pub fn new(graph: CausalGraph) -> Self {
        Self { graph }
    }

    /// Check if X is independent of Y given Z.
    pub fn is_independent(&self, x: &str, y: &str, z: &[&str]) -> bool {
        d_separated(&self.graph, x, y, z)
    }

    /// Find all variables that are d-connected to X given Z.
    pub fn d_connected(&self, x: &str, z: &[&str]) -> HashSet<String> {
        let z_set: HashSet<&str> = z.iter().copied().collect();
        let mut visited = HashSet::new();
        let mut reachable = HashSet::new();
        let mut queue = vec![(x, true)];

        while let Some((node, forward)) = queue.pop() {
            let state = (node.to_string(), forward);
            if visited.contains(&state) {
                continue;
            }
            visited.insert(state);

            if node != x {
                reachable.insert(node.to_string());
            }

            if forward {
                if !z_set.contains(node) {
                    for child in self.graph.children(node) {
                        queue.push((child, true));
                    }
                } else {
                    for parent in self.graph.parents(node) {
                        queue.push((parent, false));
                    }
                }
            } else {
                if !z_set.contains(node) {
                    for parent in self.graph.parents(node) {
                        queue.push((parent, false));
                    }
                    for child in self.graph.children(node) {
                        queue.push((child, true));
                    }
                } else {
                    for child in self.graph.children(node) {
                        queue.push((child, true));
                    }
                }
            }
        }
        reachable
    }

    /// Find the Markov blanket of a node: parents + children + other parents of children.
    pub fn markov_blanket(&self, x: &str) -> HashSet<String> {
        let mut blanket = HashSet::new();
        // Parents
        for p in self.graph.parents(x) {
            blanket.insert(p.to_string());
        }
        // Children
        for c in self.graph.children(x) {
            blanket.insert(c.to_string());
            // Other parents of children (co-parents)
            for p in self.graph.parents(c) {
                if p != x {
                    blanket.insert(p.to_string());
                }
            }
        }
        blanket
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    fn chain_graph() -> CausalGraph {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        g.add_node(Node::new("B"));
        g.add_node(Node::new("C"));
        g.add_edge(Edge::new("A", "B")).unwrap();
        g.add_edge(Edge::new("B", "C")).unwrap();
        g
    }

    fn fork_graph() -> CausalGraph {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        g.add_node(Node::new("B"));
        g.add_node(Node::new("C"));
        g.add_edge(Edge::new("B", "A")).unwrap();
        g.add_edge(Edge::new("B", "C")).unwrap();
        g
    }

    fn collider_graph() -> CausalGraph {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        g.add_node(Node::new("B"));
        g.add_node(Node::new("C"));
        g.add_edge(Edge::new("A", "B")).unwrap();
        g.add_edge(Edge::new("C", "B")).unwrap();
        g
    }

    #[test]
    fn test_chain_unblocked() {
        let g = chain_graph();
        // A -> B -> C: A and C are NOT d-separated (unblocked chain)
        assert!(!d_separated(&g, "A", "C", &[]));
    }

    #[test]
    fn test_chain_blocked_by_middle() {
        let g = chain_graph();
        // A -> B -> C: conditioning on B blocks
        assert!(d_separated(&g, "A", "C", &["B"]));
    }

    #[test]
    fn test_fork_unblocked() {
        let g = fork_graph();
        // A <- B -> C: A and C are NOT d-separated
        assert!(!d_separated(&g, "A", "C", &[]));
    }

    #[test]
    fn test_fork_blocked() {
        let g = fork_graph();
        // A <- B -> C: conditioning on B blocks
        assert!(d_separated(&g, "A", "C", &["B"]));
    }

    #[test]
    fn test_collider_blocks() {
        let g = collider_graph();
        // A -> B <- C: A and C ARE d-separated (collider blocks)
        assert!(d_separated(&g, "A", "C", &[]));
    }

    #[test]
    fn test_collider_opens_when_conditioned() {
        let g = collider_graph();
        // A -> B <- C: conditioning on B opens the path
        assert!(!d_separated(&g, "A", "C", &["B"]));
    }

    #[test]
    fn test_markov_blanket() {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        g.add_node(Node::new("B"));
        g.add_node(Node::new("C"));
        g.add_node(Node::new("D"));
        g.add_edge(Edge::new("A", "B")).unwrap();
        g.add_edge(Edge::new("C", "B")).unwrap();
        g.add_edge(Edge::new("B", "D")).unwrap();

        let engine = InferenceEngine::new(g);
        let mb = engine.markov_blanket("B");
        assert!(mb.contains("A"));
        assert!(mb.contains("C"));
        assert!(mb.contains("D"));
        assert_eq!(mb.len(), 3);
    }

    #[test]
    fn test_inference_engine() {
        let g = chain_graph();
        let engine = InferenceEngine::new(g);
        assert!(!engine.is_independent("A", "C", &[]));
        assert!(engine.is_independent("A", "C", &["B"]));
    }
}
