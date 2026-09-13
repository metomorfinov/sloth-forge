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

fn check_len(values: &[f32], expected: usize) -> Result<(), LoraError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(LoraError::DimensionMismatch {
            expected,
            actual: values.len(),
        })
    }
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

    // Матрицы в раскладке PyTorch/PEFT, построчно — так же хранит их GPU-операция
    // sloth_vk_forward_lora и так они лежат в файлах адаптеров Hugging Face.
    // weight_a (lora_A): [rank, in_features]
    pub weight_a: Vec<f32>,
    // weight_b (lora_B): [out_features, rank]
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
    pub fn new(
        name: String,
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
    ) -> Self {
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

    /// Готовит слой к обучению после загрузки из файла. Градиенты и моменты AdamW в файл
    /// адаптера не пишутся, поэтому раньше после загрузки они были пустыми и первый же шаг
    /// обучения падал с выходом за границы массива.
    fn restore_training_state(&mut self) -> Result<(), LoraError> {
        let expected_a = self.in_features * self.rank;
        if self.weight_a.len() != expected_a {
            return Err(LoraError::DimensionMismatch {
                expected: expected_a,
                actual: self.weight_a.len(),
            });
        }
        let expected_b = self.rank * self.out_features;
        if self.weight_b.len() != expected_b {
            return Err(LoraError::DimensionMismatch {
                expected: expected_b,
                actual: self.weight_b.len(),
            });
        }
        self.grad_a = vec![0.0; expected_a];
        self.grad_b = vec![0.0; expected_b];
        self.m_a = vec![0.0; expected_a];
        self.v_a = vec![0.0; expected_a];
        self.m_b = vec![0.0; expected_b];
        self.v_b = vec![0.0; expected_b];
        // Моменты начинаются заново, значит и поправка на смещение считается с первого шага
        self.step = 0;
        Ok(())
    }

    /// h[t, r] = (A x_t)[r]: проекция в пространство ранга, [batch_seq, rank].
    fn project_down(&self, x: &[f32], batch_seq: usize) -> Vec<f32> {
        let (in_f, rank) = (self.in_features, self.rank);
        let mut h = vec![0.0f32; batch_seq * rank];
        for t in 0..batch_seq {
            let x_row = &x[t * in_f..(t + 1) * in_f];
            for r in 0..rank {
                let a_row = &self.weight_a[r * in_f..(r + 1) * in_f];
                h[t * rank + r] = x_row.iter().zip(a_row).map(|(x, a)| x * a).sum();
            }
        }
        h
    }

    /// Прямой проход: Out = W x + scale * B (A x), раскладка как в PyTorch/PEFT и на GPU.
    /// X: [batch_seq, in_features], W_base: [out_features, in_features].
    /// Возвращает Out: [batch_seq, out_features].
    pub fn forward(
        &self,
        x: &[f32],
        w_base: Option<&[f32]>,
        batch_seq: usize,
    ) -> Result<Vec<f32>, LoraError> {
        let (in_f, out_f, rank) = (self.in_features, self.out_features, self.rank);
        check_len(x, batch_seq * in_f)?;
        if let Some(w) = w_base {
            check_len(w, out_f * in_f)?;
        }

        let mut out = vec![0.0f32; batch_seq * out_f];
        if let Some(w) = w_base {
            for t in 0..batch_seq {
                let x_row = &x[t * in_f..(t + 1) * in_f];
                for o in 0..out_f {
                    let w_row = &w[o * in_f..(o + 1) * in_f];
                    out[t * out_f + o] = x_row.iter().zip(w_row).map(|(x, w)| x * w).sum();
                }
            }
        }

        let h = self.project_down(x, batch_seq);
        for t in 0..batch_seq {
            let h_row = &h[t * rank..(t + 1) * rank];
            for o in 0..out_f {
                let b_row = &self.weight_b[o * rank..(o + 1) * rank];
                let lora: f32 = h_row.iter().zip(b_row).map(|(h, b)| h * b).sum();
                out[t * out_f + o] += self.scale * lora;
            }
        }
        Ok(out)
    }

    /// Обратный проход: **прибавляет** градиенты к grad_a [rank, in] и grad_b [out, rank],
    /// чтобы копить их между микробатчами (сбрасывает `zero_grad`/`adamw_step`).
    /// Операция на GPU градиенты перезаписывает: накопление — забота вызывающего кода.
    /// d_out: [batch_seq, out_features].
    pub fn backward(
        &mut self,
        x: &[f32],
        d_out: &[f32],
        batch_seq: usize,
    ) -> Result<(), LoraError> {
        let (in_f, out_f, rank) = (self.in_features, self.out_features, self.rank);
        check_len(x, batch_seq * in_f)?;
        check_len(d_out, batch_seq * out_f)?;

        let h = self.project_down(x, batch_seq);

        // dB[o, r] = scale * sum_t dOut[t, o] * h[t, r]
        for o in 0..out_f {
            for r in 0..rank {
                let sum: f32 = (0..batch_seq)
                    .map(|t| d_out[t * out_f + o] * h[t * rank + r])
                    .sum();
                self.grad_b[o * rank + r] += self.scale * sum;
            }
        }

        // dh[t, r] = scale * (B^T dOut_t)[r]
        let mut dh = vec![0.0f32; batch_seq * rank];
        for t in 0..batch_seq {
            for r in 0..rank {
                let sum: f32 = (0..out_f)
                    .map(|o| d_out[t * out_f + o] * self.weight_b[o * rank + r])
                    .sum();
                dh[t * rank + r] = self.scale * sum;
            }
        }

        // dA[r, i] = sum_t dh[t, r] * x[t, i]
        for r in 0..rank {
            for i in 0..in_f {
                let sum: f32 = (0..batch_seq)
                    .map(|t| dh[t * rank + r] * x[t * in_f + i])
                    .sum();
                self.grad_a[r * in_f + i] += sum;
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
        let mut model: Self = serde_json::from_reader(reader)?;
        for layer in model.layers.values_mut() {
            layer.restore_training_state()?;
        }
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

    fn small_config() -> LoraConfig {
        LoraConfig {
            rank: 2,
            alpha: 4.0,
            ..LoraConfig::default()
        }
    }

    #[test]
    fn loaded_checkpoint_can_continue_training() {
        let mut model = LoraModel::new(small_config(), vec![("q_proj".to_string(), 4, 4)]);
        let layer = model.layers.get_mut("q_proj").unwrap();
        layer.weight_b.fill(0.1);
        layer.backward(&[1.0; 8], &[0.5; 8], 2).unwrap();
        model.adamw_step(0.01);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.json");
        model.save_adapter_checkpoint(&path).unwrap();

        let mut loaded = LoraModel::load_adapter_checkpoint(&path).unwrap();
        let layer = loaded.layers.get_mut("q_proj").unwrap();
        assert_eq!(layer.weight_a, model.layers["q_proj"].weight_a);
        assert_eq!(
            layer.step, 0,
            "моменты начинаются заново вместе со счётчиком шага"
        );
        // Раньше здесь была паника: градиенты и моменты после загрузки были пустыми
        layer.backward(&[1.0; 8], &[0.5; 8], 2).unwrap();
        loaded.adamw_step(0.01);
        assert_eq!(loaded.layers["q_proj"].step, 1);
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        const TOLERANCE: f32 = 1e-5;
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= TOLERANCE,
                "элемент {index}: {a} вместо {e}"
            );
        }
    }

    /// Раскладка PEFT на неоднородных данных: раньше слой хранил A и B транспонированными
    /// относительно GPU-операции, а тесты на одинаковых числах этого не замечали.
    #[test]
    fn matches_peft_layout_on_uneven_data() {
        let (in_f, out_f, rank, tokens) = (5, 3, 2, 2);
        let mut layer = LoraLayer::new("o_proj".to_string(), in_f, out_f, rank, 6.0);
        layer.weight_a = (0..rank * in_f).map(|i| (i as f32 * 0.37).sin()).collect();
        layer.weight_b = (0..out_f * rank).map(|i| (i as f32 * 0.91).cos()).collect();
        let x: Vec<f32> = (0..tokens * in_f)
            .map(|i| (i as f32 * 0.53).sin() + 0.1)
            .collect();
        let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32 * 0.29).cos()).collect();
        let d_out: Vec<f32> = (0..tokens * out_f)
            .map(|i| (i as f32 * 0.71).sin())
            .collect();

        // Эталон по формулам PEFT: A [rank, in], B [out, rank], W [out, in]
        let (a, b, scale) = (layer.weight_a.clone(), layer.weight_b.clone(), layer.scale);
        let h = |t: usize, r: usize| -> f32 {
            (0..in_f).map(|i| a[r * in_f + i] * x[t * in_f + i]).sum()
        };
        let mut expected_out = vec![0.0f32; tokens * out_f];
        let mut expected_grad_b = vec![0.0f32; out_f * rank];
        let mut expected_grad_a = vec![0.0f32; rank * in_f];
        for t in 0..tokens {
            for o in 0..out_f {
                let base: f32 = (0..in_f).map(|i| w[o * in_f + i] * x[t * in_f + i]).sum();
                let lora: f32 = (0..rank).map(|r| b[o * rank + r] * h(t, r)).sum();
                expected_out[t * out_f + o] = base + scale * lora;
                for r in 0..rank {
                    expected_grad_b[o * rank + r] += scale * d_out[t * out_f + o] * h(t, r);
                }
            }
            for r in 0..rank {
                let dh: f32 = (0..out_f)
                    .map(|o| d_out[t * out_f + o] * b[o * rank + r])
                    .sum();
                for i in 0..in_f {
                    expected_grad_a[r * in_f + i] += scale * dh * x[t * in_f + i];
                }
            }
        }

        assert_close(&layer.forward(&x, Some(&w), tokens).unwrap(), &expected_out);
        layer.backward(&x, &d_out, tokens).unwrap();
        assert_close(&layer.grad_a, &expected_grad_a);
        assert_close(&layer.grad_b, &expected_grad_b);

        // Второй микробатч прибавляется к градиентам, а не заменяет их
        layer.backward(&x, &d_out, tokens).unwrap();
        let doubled = |values: &[f32]| values.iter().map(|v| v * 2.0).collect::<Vec<_>>();
        assert_close(&layer.grad_a, &doubled(&expected_grad_a));
        assert_close(&layer.grad_b, &doubled(&expected_grad_b));
    }

    #[test]
    fn wrong_base_weight_size_is_rejected() {
        let layer = LoraLayer::new("q_proj".to_string(), 4, 3, 2, 4.0);
        assert!(matches!(
            layer.forward(&[0.5; 4], Some(&[1.0; 11]), 1),
            Err(LoraError::DimensionMismatch {
                expected: 12,
                actual: 11
            })
        ));
    }

    #[test]
    fn checkpoint_with_wrong_dimensions_is_rejected() {
        let mut model = LoraModel::new(small_config(), vec![("q_proj".to_string(), 4, 4)]);
        model.layers.get_mut("q_proj").unwrap().weight_a.pop();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.json");
        model.save_adapter_checkpoint(&path).unwrap();
        assert!(matches!(
            LoraModel::load_adapter_checkpoint(&path),
            Err(LoraError::DimensionMismatch { .. })
        ));
    }
}
