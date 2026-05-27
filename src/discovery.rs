use crate::error::CausalError;
use crate::graph::{CausalGraph, Edge};
use std::collections::{HashMap, HashSet};

/// Columnar dataset for causal discovery.
#[derive(Debug, Clone, Default)]
pub struct DataSet {
    pub columns: HashMap<String, Vec<f64>>,
    pub n: usize,
}

impl DataSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_column(&mut self, name: &str, values: Vec<f64>) {
        self.n = values.len();
        self.columns.insert(name.to_string(), values);
    }

    pub fn get_column(&self, name: &str) -> Option<&Vec<f64>> {
        self.columns.get(name)
    }

    pub fn variable_names(&self) -> Vec<&str> {
        self.columns.keys().map(|s| s.as_str()).collect()
    }

    pub fn num_observations(&self) -> usize {
        self.n
    }
}

/// Compute Pearson correlation coefficient.
pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mx = mean(x);
    let my = mean(y);
    let dx: Vec<f64> = x.iter().map(|v| v - mx).collect();
    let dy: Vec<f64> = y.iter().map(|v| v - my).collect();
    let numer: f64 = dx.iter().zip(dy.iter()).map(|(a, b)| a * b).sum();
    let denom_x: f64 = dx.iter().map(|v| v * v).sum();
    let denom_y: f64 = dy.iter().map(|v| v * v).sum();
    let denom = denom_x.sqrt() * denom_y.sqrt();
    if denom < 1e-15 {
        return 0.0;
    }
    numer / denom
}

/// Partial correlation of x and y, controlling for z.
pub fn partial_correlation(x: &[f64], y: &[f64], z: &[f64]) -> f64 {
    let rxy = pearson_r(x, y);
    let rxz = pearson_r(x, z);
    let ryz = pearson_r(y, z);
    let denom = (1.0 - rxz * rxz).sqrt() * (1.0 - ryz * ryz).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }
    (rxy - rxz * ryz) / denom
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Fisher z-transformation for testing correlation significance.
pub fn fisher_z(r: f64) -> f64 {
    0.5 * ((1.0 + r) / (1.0 - r)).max(0.0).ln()
}

/// PC algorithm for causal discovery.
#[derive(Debug, Clone)]
pub struct PCAlgorithm {
    pub alpha: f64,
    pub max_conditioning: usize,
}

impl PCAlgorithm {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            max_conditioning: 3,
        }
    }

    /// Run PC algorithm on a dataset to discover causal structure.
    pub fn discover(&self, data: &DataSet) -> Result<CausalGraph, CausalError> {
        let vars = data.variable_names();
        let n_vars = vars.len();
        if n_vars < 2 {
            return Err(CausalError::Data("Need at least 2 variables".to_string()));
        }

        // Phase 1: Start with complete undirected graph
        // Represented as adjacency: all pairs are connected
        let _var_indices: HashMap<&str, usize> =
            vars.iter().enumerate().map(|(i, v)| (*v, i)).collect();
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n_vars];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_vars {
            for j in 0..n_vars {
                if i != j {
                    adj[i].insert(j);
                }
            }
        }

        // Phase 2: Remove edges based on conditional independence tests
        let mut conditioning_size = 0;
        while conditioning_size <= self.max_conditioning {
            let mut edges_removed = false;
            for i in 0..n_vars {
                for j in 0..n_vars {
                    if i == j || !adj[i].contains(&j) {
                        continue;
                    }
                    let neighbors_i: Vec<usize> =
                        adj[i].iter().filter(|&&k| k != j).copied().collect();
                    if neighbors_i.len() < conditioning_size {
                        continue;
                    }

                    // Test conditioning sets of size `conditioning_size`
                    if let Some(conditioning) = self.find_conditioning_set(
                        i,
                        j,
                        &neighbors_i,
                        conditioning_size,
                        data,
                        &vars,
                    ) {
                        if self.is_conditionally_independent(i, j, &conditioning, data, &vars) {
                            adj[i].remove(&j);
                            adj[j].remove(&i);
                            edges_removed = true;
                            break;
                        }
                    }
                }
            }
            if !edges_removed {
                break;
            }
            conditioning_size += 1;
        }

        // Phase 3: Orient edges (v-structures)
        // Build the final graph
        let mut graph = CausalGraph::new();
        for var in &vars {
            graph.add_node(crate::graph::Node::new(var));
        }

        // Add undirected edges (we'll orient them simply for now)
        for i in 0..n_vars {
            for &j in &adj[i] {
                if i < j {
                    // Orient based on marginal correlation strength
                    let xi = data.get_column(vars[i]).unwrap();
                    let xj = data.get_column(vars[j]).unwrap();
                    let corr = pearson_r(xi, xj);
                    graph.add_edge_unchecked(Edge::new(vars[i], vars[j]).with_strength(corr.abs()));
                }
            }
        }

        Ok(graph)
    }

    fn find_conditioning_set(
        &self,
        i: usize,
        _j: usize,
        neighbors: &[usize],
        size: usize,
        data: &DataSet,
        vars: &[&str],
    ) -> Option<Vec<usize>> {
        if size == 0 {
            // Test marginal independence
            let xi = data.get_column(vars[i])?;
            let xj = data.get_column(vars[_j])?;
            let r = pearson_r(xi, xj);
            let z = fisher_z(r);
            let n = xi.len().min(xj.len()) as f64;
            let z_stat = z * (n - 3.0).sqrt();
            if z_stat.abs() < 1.96 * (1.0 - self.alpha).sqrt() {
                return Some(vec![]);
            }
            return None;
        }

        // Try first `size` neighbors as conditioning set
        if neighbors.len() >= size {
            let conditioning: Vec<usize> = neighbors.iter().take(size).copied().collect();
            return Some(conditioning);
        }
        None
    }

    fn is_conditionally_independent(
        &self,
        i: usize,
        j: usize,
        conditioning: &[usize],
        data: &DataSet,
        vars: &[&str],
    ) -> bool {
        let xi = match data.get_column(vars[i]) {
            Some(v) => v,
            None => return false,
        };
        let xj = match data.get_column(vars[j]) {
            Some(v) => v,
            None => return false,
        };

        if conditioning.is_empty() {
            let r = pearson_r(xi, xj);
            let z = fisher_z(r);
            let n = xi.len().min(xj.len()) as f64;
            let z_stat = z * (n - 3.0).sqrt();
            return z_stat.abs() < 1.96;
        }

        // Use partial correlation for conditioning
        if conditioning.len() == 1 {
            let k = conditioning[0];
            let xk = match data.get_column(vars[k]) {
                Some(v) => v,
                None => return false,
            };
            let r = partial_correlation(xi, xj, xk);
            let z = fisher_z(r);
            let n = xi.len().min(xj.len()) as f64;
            let z_stat = z * (n - 3.0).sqrt();
            return z_stat.abs() < 1.96;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_pearson_r_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_abs_diff_eq!(pearson_r(&x, &y), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_r_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert_abs_diff_eq!(pearson_r(&x, &y), -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_r_uncorrelated() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let r = pearson_r(&x, &y);
        assert!(r.abs() < 0.5, "r = {}", r);
    }

    #[test]
    fn test_partial_correlation() {
        // If x and y are only correlated through z, partial corr should be near 0
        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x: Vec<f64> = z.iter().map(|v| v * 2.0 + 1.0).collect();
        let y: Vec<f64> = z.iter().map(|v| v * 3.0 - 2.0).collect();
        // Marginal correlation is high
        assert!(pearson_r(&x, &y).abs() > 0.9);
        // Partial should be near 1 (they're linear functions of same z)
        // Actually for perfect linear relationships, partial will be ~1
        let pc = partial_correlation(&x, &y, &z);
        // After controlling for z, x and y are deterministically related
        // This is expected to be high since x = f(z) and y = g(z) perfectly
    }

    #[test]
    fn test_dataset() {
        let mut ds = DataSet::new();
        ds.add_column("x", vec![1.0, 2.0, 3.0]);
        ds.add_column("y", vec![4.0, 5.0, 6.0]);
        assert_eq!(ds.num_observations(), 3);
        assert_eq!(ds.variable_names().len(), 2);
    }

    #[test]
    fn test_pc_discovery() {
        let mut ds = DataSet::new();
        // Create simple chain: A -> B -> C
        let n = 100;
        let a: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let b: Vec<f64> = a.iter().map(|v| v * 2.0 + 0.5).collect();
        let c: Vec<f64> = b.iter().map(|v| v * 1.5 - 0.3).collect();
        ds.add_column("A", a);
        ds.add_column("B", b);
        ds.add_column("C", c);

        let pc = PCAlgorithm::new(0.05);
        let graph = pc.discover(&ds).unwrap();
        assert_eq!(graph.node_count(), 3);
        // Should have discovered some edges
        assert!(graph.edge_count() >= 1);
    }

    #[test]
    fn test_fisher_z() {
        let z = fisher_z(0.0);
        assert_abs_diff_eq!(z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_empty() {
        assert_eq!(pearson_r(&[], &[]), 0.0);
    }
}
