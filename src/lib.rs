//! DAG-based causal graph for dependency tracking and causal reasoning.

use std::collections::{HashMap, HashSet, VecDeque};

/// A node in the causal graph.
#[derive(Debug, Clone)]
pub struct Node<T> {
    pub id: u64,
    pub data: T,
}

/// Directed acyclic graph for causal relationships.
#[derive(Debug, Clone)]
pub struct CausalGraph<T> {
    nodes: HashMap<u64, Node<T>>,
    edges: HashMap<u64, HashSet<u64>>,      // parent → children
    reverse: HashMap<u64, HashSet<u64>>,    // child → parents
    next_id: u64,
}

impl<T: Clone> Default for CausalGraph<T> {
    fn default() -> Self { Self::new() }
}

impl<T: Clone> CausalGraph<T> {
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), edges: HashMap::new(), reverse: HashMap::new(), next_id: 1 }
    }

    /// Add a node, returning its ID.
    pub fn add_node(&mut self, data: T) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, Node { id, data });
        self.edges.insert(id, HashSet::new());
        self.reverse.insert(id, HashSet::new());
        id
    }

    /// Add a directed edge from `from` to `to`. Returns false if it would create a cycle.
    pub fn add_edge(&mut self, from: u64, to: u64) -> bool {
        if !self.nodes.contains_key(&from) || !self.nodes.contains_key(&to) { return false; }
        if from == to { return false; }
        // Check for cycle: would `from` be reachable from `to`?
        if self.reachable(to, from) { return false; }
        self.edges.get_mut(&from).unwrap().insert(to);
        self.reverse.get_mut(&to).unwrap().insert(from);
        true
    }

    /// Check if `target` is reachable from `source`.
    pub fn reachable(&self, source: u64, target: u64) -> bool {
        if source == target { return true; }
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(source);
        while let Some(node) = queue.pop_front() {
            if node == target { return true; }
            if visited.insert(node) {
                if let Some(children) = self.edges.get(&node) {
                    for &child in children {
                        queue.push_back(child);
                    }
                }
            }
        }
        false
    }

    /// Get all ancestors of a node (transitive parents).
    pub fn ancestors(&self, id: u64) -> HashSet<u64> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        if let Some(parents) = self.reverse.get(&id) {
            for &p in parents { queue.push_back(p); }
        }
        while let Some(node) = queue.pop_front() {
            if result.insert(node) {
                if let Some(parents) = self.reverse.get(&node) {
                    for &p in parents { queue.push_back(p); }
                }
            }
        }
        result
    }

    /// Get all descendants of a node (transitive children).
    pub fn descendants(&self, id: u64) -> HashSet<u64> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        if let Some(children) = self.edges.get(&id) {
            for &c in children { queue.push_back(c); }
        }
        while let Some(node) = queue.pop_front() {
            if result.insert(node) {
                if let Some(children) = self.edges.get(&node) {
                    for &c in children { queue.push_back(c); }
                }
            }
        }
        result
    }

    /// Topological sort (Kahn's algorithm).
    pub fn topological_sort(&self) -> Vec<u64> {
        let mut in_degree: HashMap<u64, usize> = self.nodes.keys().map(|&id| (id, 0)).collect();
        for (_, children) in &self.edges {
            for &child in children {
                *in_degree.get_mut(&child).unwrap() += 1;
            }
        }

        let mut queue: VecDeque<u64> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::new();
        while let Some(node) = queue.pop_front() {
            result.push(node);
            if let Some(children) = self.edges.get(&node) {
                for &child in children {
                    let deg = in_degree.get_mut(&child).unwrap();
                    *deg -= 1;
                    if *deg == 0 { queue.push_back(child); }
                }
            }
        }
        result
    }

    /// Get immediate children of a node.
    pub fn children(&self, id: u64) -> Vec<u64> {
        self.edges.get(&id).map(|s| s.iter().copied().collect()).unwrap_or_default()
    }

    /// Get immediate parents of a node.
    pub fn parents(&self, id: u64) -> Vec<u64> {
        self.reverse.get(&id).map(|s| s.iter().copied().collect()).unwrap_or_default()
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: u64) -> Option<&Node<T>> { self.nodes.get(&id) }

    /// Number of nodes.
    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|s| s.len()).sum()
    }

    /// Find the lowest common ancestors of two nodes.
    pub fn lca(&self, a: u64, b: u64) -> HashSet<u64> {
        let ancestors_a = self.ancestors(a);
        let mut ancestors_a: HashSet<u64> = ancestors_a.into_iter().chain(std::iter::once(a)).collect();
        let ancestors_b = self.ancestors(b);
        let ancestors_b: HashSet<u64> = ancestors_b.into_iter().chain(std::iter::once(b)).collect();

        let common: HashSet<u64> = &ancestors_a & &ancestors_b;
        // Filter to lowest: those whose children are NOT common ancestors
        common.iter().copied().filter(|&c| {
            self.children(c).iter().all(|child| !common.contains(child))
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_nodes_and_edges() {
        let mut g: CausalGraph<&str> = CausalGraph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        assert!(g.add_edge(a, b));
        assert!(g.add_edge(b, c));
        assert_eq!(g.len(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_prevent_cycle() {
        let mut g: CausalGraph<i32> = CausalGraph::new();
        let a = g.add_node(1);
        let b = g.add_node(2);
        let c = g.add_node(3);
        g.add_edge(a, b);
        g.add_edge(b, c);
        assert!(!g.add_edge(c, a)); // would create cycle
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_self_loop_prevented() {
        let mut g: CausalGraph<i32> = CausalGraph::new();
        let a = g.add_node(1);
        assert!(!g.add_edge(a, a));
    }

    #[test]
    fn test_reachable() {
        let mut g: CausalGraph<i32> = CausalGraph::new();
        let a = g.add_node(1);
        let b = g.add_node(2);
        let c = g.add_node(3);
        g.add_edge(a, b);
        g.add_edge(b, c);
        assert!(g.reachable(a, c));
        assert!(!g.reachable(c, a));
    }

    #[test]
    fn test_ancestors_descendants() {
        let mut g: CausalGraph<i32> = CausalGraph::new();
        let a = g.add_node(1);
        let b = g.add_node(2);
        let c = g.add_node(3);
        g.add_edge(a, b);
        g.add_edge(b, c);
        let anc = g.ancestors(c);
        assert!(anc.contains(&a));
        assert!(anc.contains(&b));
        let desc = g.descendants(a);
        assert!(desc.contains(&b));
        assert!(desc.contains(&c));
    }

    #[test]
    fn test_topological_sort() {
        let mut g: CausalGraph<i32> = CausalGraph::new();
        let a = g.add_node(1);
        let b = g.add_node(2);
        let c = g.add_node(3);
        g.add_edge(a, b);
        g.add_edge(b, c);
        let order = g.topological_sort();
        assert_eq!(order.len(), 3);
        let pos_a = order.iter().position(|&x| x == a).unwrap();
        let pos_b = order.iter().position(|&x| x == b).unwrap();
        let pos_c = order.iter().position(|&x| x == c).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_lca() {
        let mut g: CausalGraph<i32> = CausalGraph::new();
        let root = g.add_node(0);
        let a = g.add_node(1);
        let b = g.add_node(2);
        let c = g.add_node(3);
        g.add_edge(root, a);
        g.add_edge(root, b);
        g.add_edge(a, c);
        g.add_edge(b, c);
        let lca = g.lca(a, b);
        assert!(lca.contains(&root));
    }

    #[test]
    fn test_empty_graph() {
        let g: CausalGraph<i32> = CausalGraph::new();
        assert!(g.is_empty());
        assert!(g.topological_sort().is_empty());
    }
}
