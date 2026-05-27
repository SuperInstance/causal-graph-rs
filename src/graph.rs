use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Semantic type of a causal node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum NodeType {
    Cause,
    Effect,
    #[default]
    Factor,
    Latent,
    Outcome,
}

/// A node in the causal graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    #[serde(default)]
    pub node_type: NodeType,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
}

fn default_confidence() -> f64 {
    1.0
}

impl Node {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            node_type: NodeType::Factor,
            description: String::new(),
            confidence: 1.0,
        }
    }

    pub fn with_type(mut self, t: NodeType) -> Self {
        self.node_type = t;
        self
    }
    pub fn with_description(mut self, d: &str) -> Self {
        self.description = d.to_string();
        self
    }
}

/// A directed edge between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub target: String,
    #[serde(default = "default_confidence")]
    pub strength: f64,
    #[serde(default)]
    pub description: String,
}

impl Edge {
    pub fn new(source: &str, target: &str) -> Self {
        Self {
            source: source.to_string(),
            target: target.to_string(),
            strength: 1.0,
            description: String::new(),
        }
    }

    pub fn with_strength(mut self, s: f64) -> Self {
        self.strength = s;
        self
    }
}

/// Core causal graph data structure with DAG validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraph {
    pub nodes: HashMap<String, Node>,
    pub edges: Vec<Edge>,
    #[serde(skip)]
    pub(crate) adjacency: HashMap<String, HashSet<String>>,
    #[serde(skip)]
    pub(crate) reverse: HashMap<String, HashSet<String>>,
}

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: Node) {
        let id = node.id.clone();
        self.nodes.insert(id, node);
    }

    /// Add a directed edge. Returns error if it would create a cycle.
    pub fn add_edge(&mut self, edge: Edge) -> Result<(), crate::error::CausalError> {
        // Check for cycle: would adding source->target create a path target->...->source?
        if self.would_create_cycle(&edge.source, &edge.target) {
            return Err(crate::error::CausalError::CycleDetected {
                from: edge.source.clone(),
                to: edge.target.clone(),
            });
        }
        self.adjacency
            .entry(edge.source.clone())
            .or_default()
            .insert(edge.target.clone());
        self.reverse
            .entry(edge.target.clone())
            .or_default()
            .insert(edge.source.clone());
        self.edges.push(edge);
        Ok(())
    }

    /// Add edge without cycle checking (for building from discovered structure).
    pub fn add_edge_unchecked(&mut self, edge: Edge) {
        self.adjacency
            .entry(edge.source.clone())
            .or_default()
            .insert(edge.target.clone());
        self.reverse
            .entry(edge.target.clone())
            .or_default()
            .insert(edge.source.clone());
        self.edges.push(edge);
    }

    fn would_create_cycle(&self, source: &str, target: &str) -> bool {
        if source == target {
            return true;
        }
        // BFS from target: can we reach source?
        let mut visited = HashSet::new();
        let mut queue = vec![target];
        while let Some(node) = queue.pop() {
            if node == source {
                return true;
            }
            if visited.insert(node.to_string()) {
                if let Some(children) = self.adjacency.get(node) {
                    for child in children {
                        if !visited.contains(child.as_str()) {
                            queue.push(child);
                        }
                    }
                }
            }
        }
        false
    }

    /// Get parents of a node.
    pub fn parents(&self, id: &str) -> Vec<&str> {
        self.reverse
            .get(id)
            .map(|s| s.iter().map(|x| x.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get children of a node.
    pub fn children(&self, id: &str) -> Vec<&str> {
        self.adjacency
            .get(id)
            .map(|s| s.iter().map(|x| x.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get all ancestors of a node (transitive parents).
    pub fn ancestors(&self, id: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut stack = vec![id];
        while let Some(node) = stack.pop() {
            if let Some(parents) = self.reverse.get(node) {
                for p in parents {
                    if result.insert(p.clone()) {
                        stack.push(p);
                    }
                }
            }
        }
        result
    }

    /// Get all descendants of a node (transitive children).
    pub fn descendants(&self, id: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut stack = vec![id];
        while let Some(node) = stack.pop() {
            if let Some(children) = self.adjacency.get(node) {
                for c in children {
                    if result.insert(c.clone()) {
                        stack.push(c);
                    }
                }
            }
        }
        result
    }

    /// Topological sort of the graph.
    pub fn topological_sort(&self) -> Vec<String> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for id in self.nodes.keys() {
            in_degree.insert(id.as_str(), 0);
        }
        for edge in &self.edges {
            *in_degree.entry(edge.target.as_str()).or_insert(0) += 1;
        }

        let mut queue: Vec<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::new();
        while let Some(node) = queue.pop() {
            result.push(node.to_string());
            if let Some(children) = self.adjacency.get(node) {
                for child in children {
                    let d = in_degree.get_mut(child.as_str()).unwrap();
                    *d -= 1;
                    if *d == 0 {
                        queue.push(child);
                    }
                }
            }
        }
        result
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_nodes_and_edges() {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        g.add_node(Node::new("B"));
        g.add_edge(Edge::new("A", "B")).unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        g.add_node(Node::new("B"));
        g.add_edge(Edge::new("A", "B")).unwrap();
        assert!(g.add_edge(Edge::new("B", "A")).is_err());
    }

    #[test]
    fn test_self_loop() {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        assert!(g.add_edge(Edge::new("A", "A")).is_err());
    }

    #[test]
    fn test_parents_children() {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        g.add_node(Node::new("B"));
        g.add_node(Node::new("C"));
        g.add_edge(Edge::new("A", "C")).unwrap();
        g.add_edge(Edge::new("B", "C")).unwrap();
        let parents = g.parents("C");
        assert_eq!(parents.len(), 2);
        let children = g.children("A");
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn test_ancestors() {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        g.add_node(Node::new("B"));
        g.add_node(Node::new("C"));
        g.add_edge(Edge::new("A", "B")).unwrap();
        g.add_edge(Edge::new("B", "C")).unwrap();
        let anc = g.ancestors("C");
        assert!(anc.contains("A"));
        assert!(anc.contains("B"));
        assert!(!anc.contains("C"));
    }

    #[test]
    fn test_descendants() {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        g.add_node(Node::new("B"));
        g.add_node(Node::new("C"));
        g.add_edge(Edge::new("A", "B")).unwrap();
        g.add_edge(Edge::new("B", "C")).unwrap();
        let desc = g.descendants("A");
        assert!(desc.contains("B"));
        assert!(desc.contains("C"));
    }

    #[test]
    fn test_topological_sort() {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        g.add_node(Node::new("B"));
        g.add_node(Node::new("C"));
        g.add_edge(Edge::new("A", "B")).unwrap();
        g.add_edge(Edge::new("B", "C")).unwrap();
        let order = g.topological_sort();
        assert_eq!(order.len(), 3);
        let pos_a = order.iter().position(|x| x == "A").unwrap();
        let pos_b = order.iter().position(|x| x == "B").unwrap();
        let pos_c = order.iter().position(|x| x == "C").unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_no_edges() {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("X"));
        g.add_node(Node::new("Y"));
        assert!(g.parents("X").is_empty());
        assert!(g.children("X").is_empty());
    }
}
