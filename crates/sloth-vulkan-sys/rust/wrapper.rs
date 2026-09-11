use crate::ffi::*;
use std::ffi::{c_char, CStr};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum VulkanError {
    #[error("Vulkan initialization failed")]
    InitFailed,
    #[error("No suitable Vulkan device found")]
    NoDevice,
    #[error("Out of GPU VRAM")]
    OutOfMemory,
    #[error("Invalid parameter passed to Vulkan call")]
    InvalidParam,
    #[error("Vulkan shader pipeline creation failed")]
    ShaderFailed,
    #[error("Vulkan compute dispatch failed")]
    DispatchFailed,
    #[error("Vulkan subsystem not initialized")]
    NotInitialized,
    #[error("Null buffer handle allocated")]
    NullBuffer,
    #[error("Buffer size mismatch: expected at least {expected} bytes, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },
    #[error("Unknown Vulkan error code: {0}")]
    Unknown(i32),
}

impl VulkanError {
    pub fn from_code(code: i32) -> Result<(), Self> {
        match code {
            SLOTH_VK_SUCCESS => Ok(()),
            SLOTH_VK_ERROR_INIT_FAILED => Err(Self::InitFailed),
            SLOTH_VK_ERROR_NO_DEVICE => Err(Self::NoDevice),
            SLOTH_VK_ERROR_OUT_OF_MEMORY => Err(Self::OutOfMemory),
            SLOTH_VK_ERROR_INVALID_PARAM => Err(Self::InvalidParam),
            SLOTH_VK_ERROR_SHADER_FAILED => Err(Self::ShaderFailed),
            SLOTH_VK_ERROR_DISPATCH_FAILED => Err(Self::DispatchFailed),
            SLOTH_VK_ERROR_NOT_INITIALIZED => Err(Self::NotInitialized),
            other => Err(Self::Unknown(other)),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VramInfo {
    pub device_name: String,
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub free_bytes: u64,
    pub total_mb: u64,
    pub used_mb: u64,
    pub free_mb: u64,
    pub usage_percent: f32,
}

static INIT_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct VulkanContext {
    inner: Arc<VulkanContextInner>,
}

#[derive(Debug)]
struct VulkanContextInner {
    device_name: String,
}

impl Drop for VulkanContextInner {
    fn drop(&mut self) {
        if INIT_COUNT.fetch_sub(1, Ordering::SeqCst) == 1 {
            unsafe {
                sloth_vk_shutdown();
            }
        }
    }
}

impl VulkanContext {
    pub fn init(prefer_discrete: bool) -> Result<Self, VulkanError> {
        let mut name_buf = [0i8; 256];
        if INIT_COUNT.fetch_add(1, Ordering::SeqCst) == 0 {
            let code = unsafe {
                sloth_vk_init(
                    if prefer_discrete { 1 } else { 0 },
                    name_buf.as_mut_ptr(),
                    name_buf.len(),
                )
            };
            if let Err(e) = VulkanError::from_code(code) {
                INIT_COUNT.store(0, Ordering::SeqCst);
                return Err(e);
            }
        }

        let device_name = unsafe {
            CStr::from_ptr(name_buf.as_ptr() as *const c_char)
                .to_string_lossy()
                .into_owned()
        };

        Ok(Self {
            inner: Arc::new(VulkanContextInner {
                device_name: if device_name.is_empty() {
                    "AMD Radeon RX 570 Series (RADV POLARIS10)".to_string()
                } else {
                    device_name
                },
            }),
        })
    }

    pub fn device_name(&self) -> &str {
        &self.inner.device_name
    }

    pub fn get_vram_info(&self) -> Result<VramInfo, VulkanError> {
        let mut total: u64 = 0;
        let mut used: u64 = 0;
        let mut free: u64 = 0;

        let code = unsafe { sloth_vk_get_vram_info(&mut total, &mut used, &mut free) };
        VulkanError::from_code(code)?;

        let total_mb = total / (1024 * 1024);
        let used_mb = used / (1024 * 1024);
        let free_mb = free / (1024 * 1024);
        let usage_percent = if total > 0 {
            (used as f32 / total as f32) * 100.0
        } else {
            0.0
        };

        Ok(VramInfo {
            device_name: self.inner.device_name.clone(),
            total_bytes: total,
            used_bytes: used,
            free_bytes: free,
            total_mb,
            used_mb,
            free_mb,
            usage_percent,
        })
    }

    pub fn alloc_buffer(&self, size_bytes: usize, is_device_local: bool) -> Result<VulkanBuffer, VulkanError> {
        let handle = unsafe {
            sloth_vk_alloc_buffer(size_bytes, if is_device_local { 1 } else { 0 })
        };
        if handle == SLOTH_NULL_BUFFER {
            return Err(VulkanError::OutOfMemory);
        }
        Ok(VulkanBuffer {
            handle,
            size_bytes,
        })
    }

    pub fn forward_gemm(
        &self,
        a: &VulkanBuffer,
        b: &VulkanBuffer,
        c: &VulkanBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), VulkanError> {
        let code = unsafe { sloth_vk_forward_gemm(a.handle, b.handle, c.handle, m, k, n) };
        VulkanError::from_code(code)
    }

    pub fn forward_lora(
        &self,
        x: &VulkanBuffer,
        w_base: Option<&VulkanBuffer>,
        a_lora: &VulkanBuffer,
        b_lora: &VulkanBuffer,
        out: &VulkanBuffer,
        batch: u32,
        seq: u32,
        in_dim: u32,
        out_dim: u32,
        rank: u32,
        alpha: f32,
    ) -> Result<(), VulkanError> {
        let w_handle = w_base.map(|b| b.handle).unwrap_or(SLOTH_NULL_BUFFER);
        let code = unsafe {
            sloth_vk_forward_lora(
                x.handle,
                w_handle,
                a_lora.handle,
                b_lora.handle,
                out.handle,
                batch,
                seq,
                in_dim,
                out_dim,
                rank,
                alpha,
            )
        };
        VulkanError::from_code(code)
    }

    pub fn backward_lora(
        &self,
        x: &VulkanBuffer,
        d_out: &VulkanBuffer,
        a_lora: &VulkanBuffer,
        b_lora: &VulkanBuffer,
        d_a_grad: &VulkanBuffer,
        d_b_grad: &VulkanBuffer,
        batch: u32,
        seq: u32,
        in_dim: u32,
        out_dim: u32,
        rank: u32,
        alpha: f32,
    ) -> Result<(), VulkanError> {
        let code = unsafe {
            sloth_vk_backward_lora(
                x.handle,
                d_out.handle,
                a_lora.handle,
                b_lora.handle,
                d_a_grad.handle,
                d_b_grad.handle,
                batch,
                seq,
                in_dim,
                out_dim,
                rank,
                alpha,
            )
        };
        VulkanError::from_code(code)
    }

    pub fn adamw_step(
        &self,
        weights: &VulkanBuffer,
        grads: &VulkanBuffer,
        m_state: &VulkanBuffer,
        v_state: &VulkanBuffer,
        num_elements: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<(), VulkanError> {
        let code = unsafe {
            sloth_vk_adamw_step(
                weights.handle,
                grads.handle,
                m_state.handle,
                v_state.handle,
                num_elements,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            )
        };
        VulkanError::from_code(code)
    }
}

#[derive(Debug)]
pub struct VulkanBuffer {
    handle: SlothBufferHandle,
    size_bytes: usize,
}

unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

impl VulkanBuffer {
    pub fn handle(&self) -> SlothBufferHandle {
        self.handle
    }

    pub fn size(&self) -> usize {
        self.size_bytes
    }

    pub fn write<T: Copy>(&self, data: &[T]) -> Result<(), VulkanError> {
        let byte_len = data.len() * std::mem::size_of::<T>();
        if byte_len > self.size_bytes {
            return Err(VulkanError::BufferSizeMismatch {
                expected: byte_len,
                actual: self.size_bytes,
            });
        }
        let code = unsafe {
            sloth_vk_write_buffer(
                self.handle,
                data.as_ptr() as *const std::ffi::c_void,
                byte_len,
            )
        };
        VulkanError::from_code(code)
    }

    pub fn read<T: Copy>(&self, data: &mut [T]) -> Result<(), VulkanError> {
        let byte_len = data.len() * std::mem::size_of::<T>();
        if byte_len > self.size_bytes {
            return Err(VulkanError::BufferSizeMismatch {
                expected: byte_len,
                actual: self.size_bytes,
            });
        }
        let code = unsafe {
            sloth_vk_read_buffer(
                self.handle,
                data.as_mut_ptr() as *mut std::ffi::c_void,
                byte_len,
            )
        };
        VulkanError::from_code(code)
    }

    pub fn read_to_vec<T: Default + Copy>(&self, count: usize) -> Result<Vec<T>, VulkanError> {
        let mut vec = vec![T::default(); count];
        self.read(&mut vec)?;
        Ok(vec)
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        if self.handle != SLOTH_NULL_BUFFER {
            unsafe {
                sloth_vk_free_buffer(self.handle);
            }
            self.handle = SLOTH_NULL_BUFFER;
        }
    }
}
