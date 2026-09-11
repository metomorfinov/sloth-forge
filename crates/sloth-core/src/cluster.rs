use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::{broadcast, Mutex, Notify};

#[derive(Error, Debug)]
pub enum ClusterError {
    #[error("Gradient length mismatch: master has {master_len}, worker has {worker_len}")]
    GradientLengthMismatch { master_len: usize, worker_len: usize },
    #[error("Worker not registered: {0}")]
    WorkerNotRegistered(String),
    #[error("Timeout waiting for barrier at step {step}: {name}")]
    BarrierTimeout { step: u32, name: String },
    #[error("Network error: {0}")]
    Network(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    Standalone,
    Master,
    Worker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub ip: String,
    pub gpu_name: String,
    pub vram_mb: u64,
    pub latency_ms: f32,
    pub last_seen_ms: u64,
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
        client_timestamp_ms: u64,
        server_timestamp_ms: u64,
        latency_ms: f32,
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

    /// General N-node AllReduce averaging: (1 / N) * sum(grad_i)
    pub fn average_many(all_grads: &[Vec<f32>]) -> Result<Vec<f32>, ClusterError> {
        if all_grads.is_empty() {
            return Ok(Vec::new());
        }
        let len = all_grads[0].len();
        for g in all_grads {
            if g.len() != len {
                return Err(ClusterError::GradientLengthMismatch {
                    master_len: len,
                    worker_len: g.len(),
                });
            }
        }

        let mut out = vec![0.0f32; len];
        let n = all_grads.len() as f32;

        for g in all_grads {
            for i in 0..len {
                out[i] += g[i];
            }
        }

        for i in 0..len {
            out[i] /= n;
        }

        Ok(out)
    }

    pub fn serialize_gradients(grads: &[f32]) -> Vec<u8> {
        let byte_len = grads.len() * 4;
        let mut bytes = Vec::with_capacity(byte_len);
        for &val in grads {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    pub fn deserialize_gradients(bytes: &[u8]) -> Vec<f32> {
        let num_floats = bytes.len() / 4;
        let mut grads = Vec::with_capacity(num_floats);
        for chunk in bytes.chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            grads.push(f32::from_le_bytes(arr));
        }
        grads
    }
}

pub struct ClusterCoordinator {
    pub role: NodeRole,
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

    pub async fn register_worker(&self, info: WorkerInfo) -> ClusterMessage {
        let mut workers = self.workers.lock().await;
        let rank = workers.len() + 1;
        workers.insert(info.worker_id.clone(), info.clone());
        tracing::info!(
            "Registered worker '{}' from {}. Total workers: {}",
            info.worker_id,
            info.ip,
            workers.len()
        );

        ClusterMessage::RegisterAck {
            status: "ok".to_string(),
            worker_id: info.worker_id,
            rank,
            world_size: self.world_size,
        }
    }

    pub async fn list_workers(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.lock().await;
        workers.values().cloned().collect()
    }

    pub fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    pub async fn handle_heartbeat(&self, worker_id: &str, client_ts: u64) -> ClusterMessage {
        let server_ts = Self::now_ms();
        let latency_ms = if server_ts >= client_ts {
            (server_ts - client_ts) as f32
        } else {
            0.5
        };

        let mut workers = self.workers.lock().await;
        if let Some(w) = workers.get_mut(worker_id) {
            w.last_seen_ms = server_ts;
            w.latency_ms = latency_ms;
        }

        ClusterMessage::HeartbeatAck {
            client_timestamp_ms: client_ts,
            server_timestamp_ms: server_ts,
            latency_ms,
        }
    }

    pub async fn coordinate_step_sync(&self, step: u32, epoch: u32, micro_batch_idx: usize, total_micro_batches: usize) {
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

    pub async fn perform_allreduce(
        &self,
        master_grads: &mut [f32],
        worker_grads: &[f32],
    ) -> Result<Vec<f32>, ClusterError> {
        AllReduceEngine::average_two(master_grads, worker_grads)?;
        Ok(master_grads.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allreduce_two_pcs() {
        let mut master_grads = vec![1.0f32, 2.0, 3.0, 4.0];
        let worker_grads = vec![3.0f32, 2.0, 5.0, 0.0];

        AllReduceEngine::average_two(&mut master_grads, &worker_grads).unwrap();

        // (1+3)/2=2, (2+2)/2=2, (3+5)/2=4, (4+0)/2=2
        assert_eq!(master_grads, vec![2.0, 2.0, 4.0, 2.0]);
    }

    #[test]
    fn test_gradient_serialization_roundtrip() {
        let original = vec![0.1234f32, -0.5678, 1.0, 0.0, -100.25];
        let bytes = AllReduceEngine::serialize_gradients(&original);
        assert_eq!(bytes.len(), original.len() * 4);

        let recovered = AllReduceEngine::deserialize_gradients(&bytes);
        assert_eq!(original, recovered);
    }

    #[tokio::test]
    async fn test_cluster_coordinator_worker_registration() {
        let coordinator = ClusterCoordinator::new(NodeRole::Master, 2);
        let worker = WorkerInfo {
            worker_id: "rx570-worker-01".to_string(),
            ip: "192.168.1.102".to_string(),
            gpu_name: "AMD Radeon RX 570".to_string(),
            vram_mb: 4096,
            latency_ms: 1.2,
            last_seen_ms: ClusterCoordinator::now_ms(),
        };

        let ack = coordinator.register_worker(worker).await;
        match ack {
            ClusterMessage::RegisterAck { status, rank, world_size, .. } => {
                assert_eq!(status, "ok");
                assert_eq!(rank, 1);
                assert_eq!(world_size, 2);
            }
            _ => panic!("Expected RegisterAck"),
        }

        let list = coordinator.list_workers().await;
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].worker_id, "rx570-worker-01");
    }
}
