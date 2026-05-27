//! causal-graph: Causal discovery and inference — PC algorithm, do-calculus, counterfactual reasoning.
//!
//! Rust port of the Python [causal-graph](https://github.com/SuperInstance/causal-graph) library.

mod discovery;
mod error;
mod graph;
mod inference;
mod intervention;

pub use discovery::{partial_correlation, pearson_r, DataSet, PCAlgorithm};
pub use error::CausalError;
pub use graph::{CausalGraph, Edge, Node, NodeType};
pub use inference::{d_separated, InferenceEngine};
pub use intervention::{Counterfactual, DoCalculus, Intervention};
