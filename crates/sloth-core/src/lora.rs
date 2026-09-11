use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoraError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Layer '{0}' not found")]
    LayerNotFound(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub target_modules: Vec<String>,
    pub dropout: f32,
    pub learning_rate: f32,
    pub weight_decay: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            dropout: 0.0,
            learning_rate: 2e-4,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraLayer {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    pub alpha: f32,
    pub scale: f32,

    // Weight matrices
    // weight_a: [in_features, rank]
    pub weight_a: Vec<f32>,
    // weight_b: [rank, out_features]
    pub weight_b: Vec<f32>,

    // Gradients (accumulated across micro-batches)
    #[serde(skip)]
    pub grad_a: Vec<f32>,
    #[serde(skip)]
    pub grad_b: Vec<f32>,

    // Optimizer states (AdamW 1st & 2nd moments)
    #[serde(skip)]
    pub m_a: Vec<f32>,
    #[serde(skip)]
    pub v_a: Vec<f32>,
    #[serde(skip)]
    pub m_b: Vec<f32>,
    #[serde(skip)]
    pub v_b: Vec<f32>,
    pub step: u32,
}

impl LoraLayer {
    pub fn new(name: String, in_features: usize, out_features: usize, rank: usize, alpha: f32) -> Self {
        let scale = if rank > 0 { alpha / rank as f32 } else { 1.0 };
        let mut rng = StdRng::seed_from_u64(42 + in_features as u64 + out_features as u64);

        // Kaiming Gaussian initialization for matrix A
        // std = sqrt(2 / in_features)
        let std = (2.0f32 / in_features as f32).sqrt();

        let num_a = in_features * rank;
        let mut weight_a = Vec::with_capacity(num_a);
        for _ in 0..num_a {
            // Box-Muller transform for standard normal sample
            let u1: f32 = rng.gen::<f32>().max(1e-7);
            let u2: f32 = rng.gen::<f32>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            weight_a.push(z * std);
        }

        // Matrix B initialized to zeros
        let num_b = rank * out_features;
        let weight_b = vec![0.0f32; num_b];

        Self {
            name,
            in_features,
            out_features,
            rank,
            alpha,
            scale,
            weight_a,
            weight_b,
            grad_a: vec![0.0f32; num_a],
            grad_b: vec![0.0f32; num_b],
            m_a: vec![0.0f32; num_a],
            v_a: vec![0.0f32; num_a],
            m_b: vec![0.0f32; num_b],
            v_b: vec![0.0f32; num_b],
            step: 0,
        }
    }

    pub fn num_parameters(&self) -> usize {
        self.weight_a.len() + self.weight_b.len()
    }

    pub fn zero_grad(&mut self) {
        self.grad_a.fill(0.0);
        self.grad_b.fill(0.0);
    }

    /// Forward pass: Out = X * W_base + (alpha / rank) * (X * A) * B
    /// X: [batch_seq, in_features]
    /// Returns Out: [batch_seq, out_features]
    pub fn forward(&self, x: &[f32], w_base: Option<&[f32]>, batch_seq: usize) -> Result<Vec<f32>, LoraError> {
        if x.len() != batch_seq * self.in_features {
            return Err(LoraError::DimensionMismatch {
                expected: batch_seq * self.in_features,
                actual: x.len(),
            });
        }

        let mut out = vec![0.0f32; batch_seq * self.out_features];

        // 1. Base weights contribution if provided
        if let Some(w) = w_base {
            for b in 0..batch_seq {
                for o in 0..self.out_features {
                    let mut sum = 0.0f32;
                    for i in 0..self.in_features {
                        sum += x[b * self.in_features + i] * w[i * self.out_features + o];
                    }
                    out[b * self.out_features + o] = sum;
                }
            }
        }

        // 2. LoRA contribution: h = X * A, then Out += scale * h * B
        let mut h = vec![0.0f32; batch_seq * self.rank];
        for b in 0..batch_seq {
            for r in 0..self.rank {
                let mut sum = 0.0f32;
                for i in 0..self.in_features {
                    sum += x[b * self.in_features + i] * self.weight_a[i * self.rank + r];
                }
                h[b * self.rank + r] = sum;
            }
        }

        for b in 0..batch_seq {
            for o in 0..self.out_features {
                let mut sum = 0.0f32;
                for r in 0..self.rank {
                    sum += h[b * self.rank + r] * self.weight_b[r * self.out_features + o];
                }
                out[b * self.out_features + o] += self.scale * sum;
            }
        }

        Ok(out)
    }

    /// Backward pass: computes and accumulates dA and dB gradients directly.
    /// d_out: [batch_seq, out_features]
    pub fn backward(&mut self, x: &[f32], d_out: &[f32], batch_seq: usize) -> Result<(), LoraError> {
        if x.len() != batch_seq * self.in_features || d_out.len() != batch_seq * self.out_features {
            return Err(LoraError::DimensionMismatch {
                expected: batch_seq * self.out_features,
                actual: d_out.len(),
            });
        }

        // 1. Recompute intermediate h = X * A: [batch_seq, rank]
        let mut h = vec![0.0f32; batch_seq * self.rank];
        for b in 0..batch_seq {
            for r in 0..self.rank {
                let mut sum = 0.0f32;
                for i in 0..self.in_features {
                    sum += x[b * self.in_features + i] * self.weight_a[i * self.rank + r];
                }
                h[b * self.rank + r] = sum;
            }
        }

        // 2. Accumulate grad_b: dB = scale * (h^T * dOut)
        // h^T: [rank, batch_seq], dOut: [batch_seq, out_features]
        for r in 0..self.rank {
            for o in 0..self.out_features {
                let mut sum = 0.0f32;
                for b in 0..batch_seq {
                    sum += h[b * self.rank + r] * d_out[b * self.out_features + o];
                }
                self.grad_b[r * self.out_features + o] += self.scale * sum;
            }
        }

        // 3. dh = scale * (dOut * B^T): [batch_seq, rank]
        let mut dh = vec![0.0f32; batch_seq * self.rank];
        for b in 0..batch_seq {
            for r in 0..self.rank {
                let mut sum = 0.0f32;
                for o in 0..self.out_features {
                    sum += d_out[b * self.out_features + o] * self.weight_b[r * self.out_features + o];
                }
                dh[b * self.rank + r] = self.scale * sum;
            }
        }

        // 4. Accumulate grad_a: dA = X^T * dh: [in_features, rank]
        for i in 0..self.in_features {
            for r in 0..self.rank {
                let mut sum = 0.0f32;
                for b in 0..batch_seq {
                    sum += x[b * self.in_features + i] * dh[b * self.rank + r];
                }
                self.grad_a[i * self.rank + r] += sum;
            }
        }

        Ok(())
    }

    /// AdamW optimization update
    pub fn adamw_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.step += 1;
        let bias_corr1 = 1.0 - beta1.powi(self.step as i32);
        let bias_corr2 = 1.0 - beta2.powi(self.step as i32);

        // Update Matrix A
        for i in 0..self.weight_a.len() {
            let g = self.grad_a[i];
            self.weight_a[i] -= lr * weight_decay * self.weight_a[i];

            self.m_a[i] = beta1 * self.m_a[i] + (1.0 - beta1) * g;
            self.v_a[i] = beta2 * self.v_a[i] + (1.0 - beta2) * (g * g);

            let m_hat = self.m_a[i] / bias_corr1;
            let v_hat = self.v_a[i] / bias_corr2;
            self.weight_a[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        // Update Matrix B
        for i in 0..self.weight_b.len() {
            let g = self.grad_b[i];
            self.weight_b[i] -= lr * weight_decay * self.weight_b[i];

            self.m_b[i] = beta1 * self.m_b[i] + (1.0 - beta1) * g;
            self.v_b[i] = beta2 * self.v_b[i] + (1.0 - beta2) * (g * g);

            let m_hat = self.m_b[i] / bias_corr1;
            let v_hat = self.v_b[i] / bias_corr2;
            self.weight_b[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        self.zero_grad();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraModel {
    pub config: LoraConfig,
    pub layers: HashMap<String, LoraLayer>,
}

impl LoraModel {
    pub fn new(config: LoraConfig, layer_dims: Vec<(String, usize, usize)>) -> Self {
        let mut layers = HashMap::with_capacity(layer_dims.len());
        for (name, in_dim, out_dim) in layer_dims {
            let layer = LoraLayer::new(name.clone(), in_dim, out_dim, config.rank, config.alpha);
            layers.insert(name, layer);
        }
        Self { config, layers }
    }

    pub fn total_parameters(&self) -> usize {
        self.layers.values().map(|l| l.num_parameters()).sum()
    }

    pub fn zero_grad(&mut self) {
        for layer in self.layers.values_mut() {
            layer.zero_grad();
        }
    }

    pub fn get_flat_gradients(&self) -> Vec<f32> {
        let mut flat = Vec::new();
        // Sort keys for deterministic gradient payload layout
        let mut keys: Vec<&String> = self.layers.keys().collect();
        keys.sort();
        for k in keys {
            let layer = &self.layers[k];
            flat.extend_from_slice(&layer.grad_a);
            flat.extend_from_slice(&layer.grad_b);
        }
        flat
    }

    pub fn set_flat_gradients(&mut self, grads: &[f32]) -> Result<(), LoraError> {
        let mut keys: Vec<String> = self.layers.keys().cloned().collect();
        keys.sort();

        let mut offset = 0;
        for k in &keys {
            let layer = self.layers.get_mut(k).unwrap();
            let len_a = layer.grad_a.len();
            let len_b = layer.grad_b.len();

            if offset + len_a + len_b > grads.len() {
                return Err(LoraError::DimensionMismatch {
                    expected: offset + len_a + len_b,
                    actual: grads.len(),
                });
            }

            layer.grad_a.copy_from_slice(&grads[offset..offset + len_a]);
            offset += len_a;
            layer.grad_b.copy_from_slice(&grads[offset..offset + len_b]);
            offset += len_b;
        }

        Ok(())
    }

    pub fn adamw_step(&mut self, lr: f32) {
        let wd = self.config.weight_decay;
        for layer in self.layers.values_mut() {
            layer.adamw_step(lr, 0.9, 0.999, 1e-8, wd);
        }
    }

    pub fn save_adapter_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<(), LoraError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn load_adapter_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self, LoraError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model = serde_json::from_reader(reader)?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_initialization_and_forward() {
        let mut layer = LoraLayer::new("q_proj".to_string(), 8, 8, 4, 8.0);
        assert_eq!(layer.scale, 2.0);
        // Initially B is zero, so LoRA delta should be exactly zero
        let x = vec![1.0f32; 8];
        let out = layer.forward(&x, None, 1).unwrap();
        assert_eq!(out.len(), 8);
        for val in out {
            assert_eq!(val, 0.0);
        }

        // Set some dummy weights in B
        layer.weight_b.fill(1.0);
        let out2 = layer.forward(&x, None, 1).unwrap();
        assert!(out2.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_lora_backward_and_adamw() {
        let mut layer = LoraLayer::new("v_proj".to_string(), 4, 4, 2, 4.0);
        layer.weight_b.fill(0.1);

        let x = vec![1.0f32; 8]; // 2 tokens
        let d_out = vec![0.5f32; 8]; // 2 tokens

        layer.backward(&x, &d_out, 2).unwrap();
        assert!(layer.grad_a.iter().any(|&v| v != 0.0));
        assert!(layer.grad_b.iter().any(|&v| v != 0.0));

        let initial_a = layer.weight_a.clone();
        layer.adamw_step(0.01, 0.9, 0.999, 1e-8, 0.01);
        assert_ne!(initial_a, layer.weight_a);
        assert_eq!(layer.step, 1);
    }
}
