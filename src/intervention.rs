use crate::graph::CausalGraph;
use crate::inference::d_separated;
use std::collections::HashSet;

/// Represents a do-intervention on a variable.
#[derive(Debug, Clone)]
pub struct Intervention {
    pub variable: String,
    pub value: Option<f64>,
}

impl Intervention {
    pub fn do_set(variable: &str) -> Self {
        Self {
            variable: variable.to_string(),
            value: None,
        }
    }

    pub fn do_set_to(variable: &str, value: f64) -> Self {
        Self {
            variable: variable.to_string(),
            value: Some(value),
        }
    }
}

/// Do-calculus operations for causal reasoning.
#[derive(Debug, Clone)]
pub struct DoCalculus {
    pub graph: CausalGraph,
}

impl DoCalculus {
    pub fn new(graph: CausalGraph) -> Self {
        Self { graph }
    }

    /// Apply the truncated factorization (Rule 2 of do-calculus).
    /// When we do(X=x), remove all incoming edges to X and compute
    /// P(Y | do(X)) = Σ_{pa(Y)\{X}} P(Y | pa(Y)) · Π P(pa_i | pa(pa_i))
    pub fn intervene(&self, intervention: &Intervention) -> CausalGraph {
        let mut modified = self.graph.clone();
        // Remove all incoming edges to the intervened variable
        modified.edges.retain(|e| e.target != intervention.variable);
        // Rebuild adjacency
        modified.adjacency.clear();
        modified.reverse.clear();
        for edge in &modified.edges {
            modified
                .adjacency
                .entry(edge.source.clone())
                .or_default()
                .insert(edge.target.clone());
            modified
                .reverse
                .entry(edge.target.clone())
                .or_default()
                .insert(edge.source.clone());
        }
        modified
    }

    /// Check if P(Y | do(X)) = P(Y | X) (identifiability via backdoor criterion).
    pub fn is_observable_effect(&self, cause: &str, effect: &str) -> bool {
        // Backdoor criterion: find a set Z that blocks all backdoor paths
        // from X to Y (paths via parents of X)
        let parents: HashSet<String> = self
            .graph
            .parents(cause)
            .iter()
            .map(|s| s.to_string())
            .collect();
        // If X has no parents, the effect is observable
        if parents.is_empty() {
            return true;
        }
        // Check if conditioning on parents blocks all non-causal paths
        let parent_refs: Vec<&str> = parents.iter().map(|s| s.as_str()).collect();
        d_separated(&self.graph, cause, effect, &parent_refs)
    }

    /// Compute the causal effect as the set of variables affected by do(X).
    pub fn causal_effect_set(&self, cause: &str) -> HashSet<String> {
        self.graph.descendants(cause)
    }
}

/// Counterfactual reasoning on causal models.
#[derive(Debug, Clone)]
pub struct Counterfactual {
    pub graph: CausalGraph,
}

impl Counterfactual {
    pub fn new(graph: CausalGraph) -> Self {
        Self { graph }
    }

    /// Compute what would happen if we set X=x in a world where we observed X=x_obs, Y=y_obs.
    /// Returns the set of variables that would change under this counterfactual.
    pub fn counterfactual_effect(
        &self,
        intervened_var: &str,
        _observed_value: f64,
        counterfactual_value: f64,
    ) -> HashSet<String> {
        // The counterfactual effect is the set of descendants that would change
        // Simple version: all descendants whose values depend on the intervened variable
        let descendants = self.graph.descendants(intervened_var);

        // In a more complete implementation, we'd:
        // 1. Abduction: infer exogenous noise from observations
        // 2. Action: replace the structural equation for X
        // 3. Prediction: propagate through the model

        // For now, return descendants that would be affected by the value change
        let _ = counterfactual_value; // used in full implementation
        descendants
    }

    /// Check if a counterfactual query is well-defined (the variable exists and has descendants).
    pub fn is_valid_query(&self, variable: &str) -> bool {
        self.graph.nodes.contains_key(variable) && !self.graph.descendants(variable).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    fn simple_causal_graph() -> CausalGraph {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("Treatment"));
        g.add_node(Node::new("Outcome"));
        g.add_node(Node::new("Confounder"));
        g.add_edge(Edge::new("Confounder", "Treatment")).unwrap();
        g.add_edge(Edge::new("Confounder", "Outcome")).unwrap();
        g.add_edge(Edge::new("Treatment", "Outcome")).unwrap();
        g
    }

    #[test]
    fn test_intervention_removes_parents() {
        let g = simple_causal_graph();
        let dc = DoCalculus::new(g);
        let intervention = Intervention::do_set("Treatment");
        let modified = dc.intervene(&intervention);
        assert!(modified.parents("Treatment").is_empty());
        // Treatment -> Outcome edge should remain
        assert_eq!(modified.children("Treatment").len(), 1);
    }

    #[test]
    fn test_causal_effect_set() {
        let g = simple_causal_graph();
        let dc = DoCalculus::new(g);
        let effects = dc.causal_effect_set("Treatment");
        assert!(effects.contains("Outcome"));
        assert_eq!(effects.len(), 1);
    }

    #[test]
    fn test_observable_effect_no_confounder() {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("X"));
        g.add_node(Node::new("Y"));
        g.add_edge(Edge::new("X", "Y")).unwrap();
        let dc = DoCalculus::new(g);
        assert!(dc.is_observable_effect("X", "Y"));
    }

    #[test]
    fn test_counterfactual() {
        let g = simple_causal_graph();
        let cf = Counterfactual::new(g);
        assert!(cf.is_valid_query("Treatment"));
        let effect = cf.counterfactual_effect("Treatment", 0.0, 1.0);
        assert!(effect.contains("Outcome"));
    }

    #[test]
    fn test_intervention_preserves_other_edges() {
        let mut g = CausalGraph::new();
        g.add_node(Node::new("A"));
        g.add_node(Node::new("B"));
        g.add_node(Node::new("C"));
        g.add_edge(Edge::new("A", "B")).unwrap();
        g.add_edge(Edge::new("B", "C")).unwrap();
        let dc = DoCalculus::new(g);
        let modified = dc.intervene(&Intervention::do_set("B"));
        // A->B removed, B->C preserved
        assert!(modified.parents("B").is_empty());
        assert_eq!(modified.children("B").len(), 1);
    }
}
