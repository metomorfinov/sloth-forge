//! Кластер из нескольких ПК: регистрация узлов, heartbeat и усреднение градиентов (AllReduce).
//!
//! Раньше ранг узла был «число узлов + 1» (повторная регистрация давала новый ранг),
//! размер кластера всегда был 2, задержка при расхождении часов выдумывалась (0,5 мс),
//! а усреднение перезаписывало градиенты мастера. Распределённое обучение появится
//! на этапе 3; здесь — корректная основа для него.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::{broadcast, Mutex, Notify};

/// Размер одного float32 в двоичном формате градиентов.
const F32_BYTES: usize = 4;
/// Первый ранг узла: ранг 0 — у мастера.
const FIRST_WORKER_RANK: usize = 1;

#[derive(Error, Debug, PartialEq)]
pub enum ClusterError {
    #[error("Gradient length mismatch: master has {master_len}, worker has {worker_len}")]
    GradientLengthMismatch {
        master_len: usize,
        worker_len: usize,
    },
    #[error("Worker not registered: {0}")]
    WorkerNotRegistered(String),
    #[error("Timeout waiting for barrier at step {step}: {name}")]
    BarrierTimeout { step: u32, name: String },
    #[error("Network error: {0}")]
    Network(String),
    #[error("Binary gradient payload of {len} bytes is not a whole number of float32 values")]
    PartialGradientBytes { len: usize },
    #[error("Gradient at index {index} is NaN or infinite")]
    NonFiniteGradient { index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    Standalone,
    Master,
    Worker,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub worker_id: String,
    /// Номер узла: у мастера 0, у узлов — с 1. Назначается координатором и не меняется
    /// при повторной регистрации того же узла.
    pub rank: usize,
    pub ip: String,
    pub gpu_name: String,
    pub vram_mb: u64,
    /// Задержка по последнему heartbeat; `None`, пока её не измерили.
    pub latency_ms: Option<f32>,
    pub last_seen_ms: u64,
}

impl WorkerInfo {
    /// Узел на связи, если heartbeat приходил не раньше чем `timeout_ms` назад.
    pub fn is_online(&self, now_ms: u64, timeout_ms: u64) -> bool {
        now_ms.saturating_sub(self.last_seen_ms) <= timeout_ms
    }
}

/// Градиенты мастера для одного шага обучения (их выставляет движок обучения).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepGradients {
    pub step: u32,
    pub gradients: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum ClusterMessage {
    RegisterWorker {
        worker_id: String,
        ip: String,
        gpu_name: String,
        vram_mb: u64,
    },
    RegisterAck {
        status: String,
        worker_id: String,
        rank: usize,
        world_size: usize,
    },
    Heartbeat {
        timestamp_ms: u64,
    },
    HeartbeatAck {
        client_timestamp_ms: Option<u64>,
        server_timestamp_ms: u64,
        latency_ms: Option<f32>,
    },
    SyncStep {
        step: u32,
        epoch: u32,
        micro_batch_idx: usize,
        total_micro_batches: usize,
    },
    GradientExchange {
        step: u32,
        rank: usize,
        gradients: Vec<f32>,
    },
    AveragedGradients {
        step: u32,
        gradients: Vec<f32>,
    },
    Barrier {
        step: u32,
        name: String,
    },
    BarrierAck {
        step: u32,
        name: String,
    },
}

pub struct AllReduceEngine;

impl AllReduceEngine {
    /// In-place AllReduce averaging across 2 PCs: (grad_master + grad_worker) / 2
    pub fn average_two(master_grad: &mut [f32], worker_grad: &[f32]) -> Result<(), ClusterError> {
        if master_grad.len() != worker_grad.len() {
            return Err(ClusterError::GradientLengthMismatch {
                master_len: master_grad.len(),
                worker_len: worker_grad.len(),
            });
        }

        for (m, &w) in master_grad.iter_mut().zip(worker_grad.iter()) {
            *m = (*m + w) * 0.5;
        }

        Ok(())
    }

    /// Среднее двух наборов без изменения исходных: мастер-копия остаётся нетронутой.
    pub fn averaged(master_grad: &[f32], worker_grad: &[f32]) -> Result<Vec<f32>, ClusterError> {
        if master_grad.len() != worker_grad.len() {
            return Err(ClusterError::GradientLengthMismatch {
                master_len: master_grad.len(),
                worker_len: worker_grad.len(),
            });
        }
        Ok(master_grad
            .iter()
            .zip(worker_grad)
            .map(|(master, worker)| (master + worker) * 0.5)
            .collect())
    }

    /// General N-node AllReduce averaging: (1 / N) * sum(grad_i)
    pub fn average_many(all_grads: &[Vec<f32>]) -> Result<Vec<f32>, ClusterError> {
        let Some(first) = all_grads.first() else {
            return Ok(Vec::new());
        };
        let len = first.len();
        if let Some(mismatched) = all_grads.iter().find(|grads| grads.len() != len) {
            return Err(ClusterError::GradientLengthMismatch {
                master_len: len,
                worker_len: mismatched.len(),
            });
        }

        let mut out = vec![0.0f32; len];
        for grads in all_grads {
            for (sum, value) in out.iter_mut().zip(grads) {
                *sum += value;
            }
        }
        let n = all_grads.len() as f32;
        for sum in &mut out {
            *sum /= n;
        }
        Ok(out)
    }

    /// NaN или бесконечность в одном узле испортили бы среднее для всех.
    pub fn ensure_finite(grads: &[f32]) -> Result<(), ClusterError> {
        match grads.iter().position(|value| !value.is_finite()) {
            Some(index) => Err(ClusterError::NonFiniteGradient { index }),
            None => Ok(()),
        }
    }

    pub fn serialize_gradients(grads: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(grads.len() * F32_BYTES);
        for &val in grads {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Двоичные float32 (little-endian). Раньше лишние байты в конце молча отбрасывались.
    pub fn deserialize_gradients(bytes: &[u8]) -> Result<Vec<f32>, ClusterError> {
        let (chunks, rest) = bytes.as_chunks::<F32_BYTES>();
        if !rest.is_empty() {
            return Err(ClusterError::PartialGradientBytes { len: bytes.len() });
        }
        Ok(chunks
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect())
    }
}

pub struct ClusterCoordinator {
    pub role: NodeRole,
    /// Размер кластера, под который он задуман; фактический считается по зарегистрированным узлам.
    pub world_size: usize,
    pub current_step: AtomicU32,
    workers: Mutex<HashMap<String, WorkerInfo>>,
    step_notify: Arc<Notify>,
    barrier_notify: Arc<Notify>,
    pub msg_tx: broadcast::Sender<ClusterMessage>,
}

impl ClusterCoordinator {
    pub fn new(role: NodeRole, world_size: usize) -> Self {
        let (msg_tx, _) = broadcast::channel(100);
        Self {
            role,
            world_size,
            current_step: AtomicU32::new(0),
            workers: Mutex::new(HashMap::new()),
            step_notify: Arc::new(Notify::new()),
            barrier_notify: Arc::new(Notify::new()),
            msg_tx,
        }
    }

    /// Регистрирует узел. Поле `rank` из `info` игнорируется: повторная регистрация того же
    /// узла сохраняет его ранг, новый узел получает наименьший свободный.
    pub async fn register_worker(&self, mut info: WorkerInfo) -> ClusterMessage {
        let mut workers = self.workers.lock().await;
        info.rank = match workers.get(&info.worker_id) {
            Some(existing) => existing.rank,
            None => (FIRST_WORKER_RANK..)
                .find(|rank| workers.values().all(|worker| worker.rank != *rank))
                .unwrap_or(FIRST_WORKER_RANK),
        };
        workers.insert(info.worker_id.clone(), info.clone());
        tracing::info!(
            "Registered worker '{}' (rank {}) from {}. Total workers: {}",
            info.worker_id,
            info.rank,
            info.ip,
            workers.len()
        );

        ClusterMessage::RegisterAck {
            status: "ok".to_string(),
            worker_id: info.worker_id,
            rank: info.rank,
            // Мастер и все зарегистрированные узлы
            world_size: workers.len() + 1,
        }
    }

    /// Узлы по возрастанию ранга.
    pub async fn list_workers(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.lock().await;
        let mut list: Vec<WorkerInfo> = workers.values().cloned().collect();
        list.sort_by_key(|worker| worker.rank);
        list
    }

    pub fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Отмечает узел на связи. Задержка считается по метке времени узла; если часы узла
    /// спешат (метка из будущего), задержка не выдумывается.
    pub async fn handle_heartbeat(
        &self,
        worker_id: &str,
        client_ts: Option<u64>,
    ) -> Result<ClusterMessage, ClusterError> {
        let server_ts = Self::now_ms();
        let latency_ms = client_ts
            .filter(|client| *client <= server_ts)
            .map(|client| (server_ts - client) as f32);

        let mut workers = self.workers.lock().await;
        let worker = workers
            .get_mut(worker_id)
            .ok_or_else(|| ClusterError::WorkerNotRegistered(worker_id.to_string()))?;
        worker.last_seen_ms = server_ts;
        if latency_ms.is_some() {
            worker.latency_ms = latency_ms;
        }

        Ok(ClusterMessage::HeartbeatAck {
            client_timestamp_ms: client_ts,
            server_timestamp_ms: server_ts,
            latency_ms,
        })
    }

    pub async fn coordinate_step_sync(
        &self,
        step: u32,
        epoch: u32,
        micro_batch_idx: usize,
        total_micro_batches: usize,
    ) {
        self.current_step.store(step, Ordering::SeqCst);
        let _ = self.msg_tx.send(ClusterMessage::SyncStep {
            step,
            epoch,
            micro_batch_idx,
            total_micro_batches,
        });
        self.step_notify.notify_waiters();
    }

    pub async fn trigger_barrier(&self, step: u32, name: &str) {
        let _ = self.msg_tx.send(ClusterMessage::Barrier {
            step,
            name: name.to_string(),
        });
        self.barrier_notify.notify_waiters();
    }

    pub async fn wait_for_barrier(&self) {
        self.barrier_notify.notified().await;
    }

    /// Среднее градиентов мастера и узла; градиенты мастера не меняются.
    pub async fn perform_allreduce(
        &self,
        master_grads: &[f32],
        worker_grads: &[f32],
    ) -> Result<Vec<f32>, ClusterError> {
        AllReduceEngine::averaged(master_grads, worker_grads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: &str) -> WorkerInfo {
        WorkerInfo {
            worker_id: id.to_string(),
            rank: 0,
            ip: "192.168.1.102".to_string(),
            gpu_name: "AMD Radeon RX 570".to_string(),
            vram_mb: 4096,
            latency_ms: None,
            last_seen_ms: ClusterCoordinator::now_ms(),
        }
    }

    fn ack_rank(ack: ClusterMessage) -> (usize, usize) {
        match ack {
            ClusterMessage::RegisterAck {
                rank, world_size, ..
            } => (rank, world_size),
            other => panic!("Expected RegisterAck, got {other:?}"),
        }
    }

    #[test]
    fn test_allreduce_two_pcs() {
        let mut master_grads = vec![1.0f32, 2.0, 3.0, 4.0];
        let worker_grads = vec![3.0f32, 2.0, 5.0, 0.0];

        AllReduceEngine::average_two(&mut master_grads, &worker_grads).unwrap();

        // (1+3)/2=2, (2+2)/2=2, (3+5)/2=4, (4+0)/2=2
        assert_eq!(master_grads, vec![2.0, 2.0, 4.0, 2.0]);
    }

    #[test]
    fn averaged_keeps_master_untouched_and_checks_length() {
        let master = vec![1.0f32, 2.0, 3.0, 4.0];
        let averaged = AllReduceEngine::averaged(&master, &[3.0, 2.0, 5.0, 0.0]).unwrap();
        assert_eq!(averaged, [2.0, 2.0, 4.0, 2.0]);
        assert_eq!(master, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            AllReduceEngine::averaged(&master, &[1.0]),
            Err(ClusterError::GradientLengthMismatch {
                master_len: 4,
                worker_len: 1
            })
        );
    }

    #[test]
    fn average_many_nodes() {
        let grads = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 9.0]];
        assert_eq!(AllReduceEngine::average_many(&grads).unwrap(), [3.0, 5.0]);
        assert!(AllReduceEngine::average_many(&[vec![1.0], vec![1.0, 2.0]]).is_err());
        assert!(AllReduceEngine::average_many(&[]).unwrap().is_empty());
    }

    #[test]
    fn test_gradient_serialization_roundtrip() {
        let original = vec![0.1234f32, -0.5678, 1.0, 0.0, -100.25];
        let bytes = AllReduceEngine::serialize_gradients(&original);
        assert_eq!(bytes.len(), original.len() * 4);

        let recovered = AllReduceEngine::deserialize_gradients(&bytes).unwrap();
        assert_eq!(original, recovered);
        assert_eq!(
            AllReduceEngine::deserialize_gradients(&bytes[..7]),
            Err(ClusterError::PartialGradientBytes { len: 7 })
        );
    }

    #[test]
    fn non_finite_gradients_are_rejected() {
        assert!(AllReduceEngine::ensure_finite(&[1.0, 2.0]).is_ok());
        assert_eq!(
            AllReduceEngine::ensure_finite(&[1.0, f32::NAN]),
            Err(ClusterError::NonFiniteGradient { index: 1 })
        );
        assert!(AllReduceEngine::ensure_finite(&[f32::INFINITY]).is_err());
    }

    #[tokio::test]
    async fn test_cluster_coordinator_worker_registration() {
        let coordinator = ClusterCoordinator::new(NodeRole::Master, 2);

        assert_eq!(
            ack_rank(coordinator.register_worker(worker("a")).await),
            (1, 2)
        );
        // Повторная регистрация сохраняет ранг
        assert_eq!(
            ack_rank(coordinator.register_worker(worker("a")).await),
            (1, 2)
        );
        assert_eq!(
            ack_rank(coordinator.register_worker(worker("b")).await),
            (2, 3)
        );

        let list = coordinator.list_workers().await;
        let ranks: Vec<usize> = list.iter().map(|w| w.rank).collect();
        assert_eq!(ranks, [1, 2]);
        assert_eq!(list[0].worker_id, "a");
    }

    #[tokio::test]
    async fn heartbeat_updates_known_workers_only() {
        let coordinator = ClusterCoordinator::new(NodeRole::Master, 2);
        coordinator.register_worker(worker("a")).await;

        let past = ClusterCoordinator::now_ms() - 5;
        match coordinator.handle_heartbeat("a", Some(past)).await.unwrap() {
            ClusterMessage::HeartbeatAck { latency_ms, .. } => {
                assert!(latency_ms.is_some_and(|ms| ms >= 0.0))
            }
            other => panic!("Expected HeartbeatAck, got {other:?}"),
        }
        let future = ClusterCoordinator::now_ms() + 60_000;
        match coordinator
            .handle_heartbeat("a", Some(future))
            .await
            .unwrap()
        {
            ClusterMessage::HeartbeatAck { latency_ms, .. } => assert_eq!(latency_ms, None),
            other => panic!("Expected HeartbeatAck, got {other:?}"),
        }
        assert_eq!(
            coordinator.handle_heartbeat("ghost", None).await.err(),
            Some(ClusterError::WorkerNotRegistered("ghost".to_string()))
        );
    }

    #[test]
    fn worker_online_status_follows_last_heartbeat() {
        let mut info = worker("a");
        info.last_seen_ms = 1_000;
        assert!(info.is_online(10_000, 15_000));
        assert!(!info.is_online(20_000, 15_000));
    }
}
