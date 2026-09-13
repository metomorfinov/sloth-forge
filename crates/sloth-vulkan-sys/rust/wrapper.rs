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
    #[error("Not supported by the Vulkan driver")]
    NotSupported,
    #[error("Null buffer handle allocated")]
    NullBuffer,
    #[error("Buffer size mismatch: expected at least {expected} bytes, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },
    #[error("{operation}: buffer `{buffer}` needs at least {expected} bytes, got {actual}")]
    ShapeMismatch {
        operation: &'static str,
        buffer: &'static str,
        expected: u64,
        actual: usize,
    },
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
            SLOTH_VK_ERROR_NOT_SUPPORTED => Err(Self::NotSupported),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceType {
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
}

impl DeviceType {
    fn from_raw(raw: u32) -> Self {
        match raw {
            SLOTH_VK_DEVICE_TYPE_INTEGRATED_GPU => Self::IntegratedGpu,
            SLOTH_VK_DEVICE_TYPE_DISCRETE_GPU => Self::DiscreteGpu,
            SLOTH_VK_DEVICE_TYPE_VIRTUAL_GPU => Self::VirtualGpu,
            SLOTH_VK_DEVICE_TYPE_CPU => Self::Cpu,
            _ => Self::Other,
        }
    }
}

/// Сведения о выбранном устройстве (`sloth_vk_get_device_info`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DeviceInfo {
    pub name: String,
    /// Например «radv»; `None`, если драйвер не сообщает.
    pub driver_name: Option<String>,
    /// Например «Mesa 26.2.2».
    pub driver_info: Option<String>,
    pub vendor_id: u32,
    pub device_id: u32,
    /// Версия Vulkan API устройства, например «1.4.328».
    pub api_version: String,
    pub device_type: DeviceType,
    pub subgroup_size: Option<u32>,
    pub max_compute_work_group_count: [u32; 3],
    pub memory_budget_supported: bool,
    pub device_local_bytes: u64,
}

/// Бюджет видеопамяти (`VK_EXT_memory_budget`) по всем DEVICE_LOCAL-кучам.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct MemoryBudget {
    /// Сколько процесс может занять сейчас с учётом других программ.
    pub budget_bytes: u64,
    /// Сколько занимает сам процесс.
    pub usage_bytes: u64,
}

/// VK_MAKE_API_VERSION → «major.minor.patch».
fn format_api_version(version: u32) -> String {
    const MAJOR_SHIFT: u32 = 22;
    const MINOR_SHIFT: u32 = 12;
    const MAJOR_MASK: u32 = 0x7F;
    const MINOR_MASK: u32 = 0x3FF;
    const PATCH_MASK: u32 = 0xFFF;
    format!(
        "{}.{}.{}",
        (version >> MAJOR_SHIFT) & MAJOR_MASK,
        (version >> MINOR_SHIFT) & MINOR_MASK,
        version & PATCH_MASK
    )
}

fn c_string(chars: &[c_char]) -> String {
    let bytes: Vec<u8> = chars
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).trim().to_string()
}

static INIT_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Инициализация и завершение идут по одному, иначе параллельные вызовы расходятся со счётчиком.
static INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
/// Размер буфера под имя устройства (VK_MAX_PHYSICAL_DEVICE_NAME_SIZE).
const DEVICE_NAME_BUFFER_LEN: usize = 256;
const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;
const BYTES_PER_MIB: u64 = 1024 * 1024;

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
        let _guard = INIT_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if INIT_COUNT.fetch_sub(1, Ordering::SeqCst) == 1 {
            unsafe {
                sloth_vk_shutdown();
            }
        }
    }
}

/// Буфер вмещает `count` чисел f32. Раньше обёртка отдавала размерности в C без проверки,
/// и маленький буфер означал чтение и запись за его концом на GPU.
fn expect_floats(
    operation: &'static str,
    buffer_name: &'static str,
    buffer: &VulkanBuffer,
    count: u64,
) -> Result<(), VulkanError> {
    let expected = count
        .checked_mul(F32_BYTES)
        .ok_or(VulkanError::InvalidParam)?;
    if (buffer.size_bytes as u64) < expected {
        return Err(VulkanError::ShapeMismatch {
            operation,
            buffer: buffer_name,
            expected,
            actual: buffer.size_bytes,
        });
    }
    Ok(())
}

impl VulkanContext {
    pub fn init(prefer_discrete: bool) -> Result<Self, VulkanError> {
        let _guard = INIT_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut name_buf = [0 as c_char; DEVICE_NAME_BUFFER_LEN];
        // C-слой вызывается при каждой инициализации: уже готовый контекст просто возвращает
        // имя устройства. Раньше повторный вызов пропускался, имя оставалось пустым и
        // подменялось зашитым «AMD Radeon RX 570».
        let code = unsafe {
            sloth_vk_init(
                if prefer_discrete { 1 } else { 0 },
                name_buf.as_mut_ptr(),
                name_buf.len(),
            )
        };
        VulkanError::from_code(code)?;
        INIT_COUNT.fetch_add(1, Ordering::SeqCst);

        let device_name = unsafe {
            CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        Ok(Self {
            inner: Arc::new(VulkanContextInner { device_name }),
        })
    }

    pub fn device_name(&self) -> &str {
        &self.inner.device_name
    }

    pub fn device_info(&self) -> Result<DeviceInfo, VulkanError> {
        let mut raw = SlothDeviceInfo::default();
        VulkanError::from_code(unsafe { sloth_vk_get_device_info(&mut raw) })?;
        let optional = |chars: &[c_char]| Some(c_string(chars)).filter(|text| !text.is_empty());
        Ok(DeviceInfo {
            name: c_string(&raw.device_name),
            driver_name: optional(&raw.driver_name),
            driver_info: optional(&raw.driver_info),
            vendor_id: raw.vendor_id,
            device_id: raw.device_id,
            api_version: format_api_version(raw.api_version),
            device_type: DeviceType::from_raw(raw.device_type),
            subgroup_size: Some(raw.subgroup_size).filter(|size| *size > 0),
            max_compute_work_group_count: raw.max_compute_work_group_count,
            memory_budget_supported: raw.memory_budget_supported != 0,
            device_local_bytes: raw.device_local_bytes,
        })
    }

    /// `Err(NotSupported)`, если драйвер не умеет `VK_EXT_memory_budget`.
    pub fn memory_budget(&self) -> Result<MemoryBudget, VulkanError> {
        let (mut budget_bytes, mut usage_bytes) = (0u64, 0u64);
        VulkanError::from_code(unsafe {
            sloth_vk_get_memory_budget(&mut budget_bytes, &mut usage_bytes)
        })?;
        Ok(MemoryBudget {
            budget_bytes,
            usage_bytes,
        })
    }

    pub fn get_vram_info(&self) -> Result<VramInfo, VulkanError> {
        let mut total: u64 = 0;
        let mut used: u64 = 0;
        let mut free: u64 = 0;

        let code = unsafe { sloth_vk_get_vram_info(&mut total, &mut used, &mut free) };
        VulkanError::from_code(code)?;

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
            total_mb: total / BYTES_PER_MIB,
            used_mb: used / BYTES_PER_MIB,
            free_mb: free / BYTES_PER_MIB,
            usage_percent,
        })
    }

    pub fn alloc_buffer(
        &self,
        size_bytes: usize,
        is_device_local: bool,
    ) -> Result<VulkanBuffer, VulkanError> {
        let handle =
            unsafe { sloth_vk_alloc_buffer(size_bytes, if is_device_local { 1 } else { 0 }) };
        if handle == SLOTH_NULL_BUFFER {
            return Err(VulkanError::OutOfMemory);
        }
        Ok(VulkanBuffer { handle, size_bytes })
    }

    /// C = A × B; A [m, k], B [k, n], C [m, n].
    pub fn forward_gemm(
        &self,
        a: &VulkanBuffer,
        b: &VulkanBuffer,
        c: &VulkanBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), VulkanError> {
        const OP: &str = "forward_gemm";
        let (m64, k64, n64) = (u64::from(m), u64::from(k), u64::from(n));
        expect_floats(OP, "a", a, m64 * k64)?;
        expect_floats(OP, "b", b, k64 * n64)?;
        expect_floats(OP, "c", c, m64 * n64)?;
        let code = unsafe { sloth_vk_forward_gemm(a.handle, b.handle, c.handle, m, k, n) };
        VulkanError::from_code(code)
    }

    /// Прямой проход LoRA в раскладке PEFT: W [out, in], A [rank, in], B [out, rank].
    /// Аргументы повторяют C API один в один.
    #[allow(clippy::too_many_arguments)]
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
        const OP: &str = "forward_lora";
        let tokens = u64::from(batch) * u64::from(seq);
        let (in64, out64, rank64) = (u64::from(in_dim), u64::from(out_dim), u64::from(rank));
        expect_floats(OP, "x", x, tokens * in64)?;
        if let Some(w) = w_base {
            expect_floats(OP, "w_base", w, out64 * in64)?;
        }
        expect_floats(OP, "a_lora", a_lora, rank64 * in64)?;
        expect_floats(OP, "b_lora", b_lora, out64 * rank64)?;
        expect_floats(OP, "out", out, tokens * out64)?;

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

    /// Обратный проход LoRA: перезаписывает d_a_grad [rank, in] и d_b_grad [out, rank].
    #[allow(clippy::too_many_arguments)]
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
        const OP: &str = "backward_lora";
        let tokens = u64::from(batch) * u64::from(seq);
        let (in64, out64, rank64) = (u64::from(in_dim), u64::from(out_dim), u64::from(rank));
        expect_floats(OP, "x", x, tokens * in64)?;
        expect_floats(OP, "d_out", d_out, tokens * out64)?;
        expect_floats(OP, "a_lora", a_lora, rank64 * in64)?;
        expect_floats(OP, "b_lora", b_lora, out64 * rank64)?;
        expect_floats(OP, "d_a_grad", d_a_grad, rank64 * in64)?;
        expect_floats(OP, "d_b_grad", d_b_grad, out64 * rank64)?;

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

    /// Шаг AdamW на месте. Аргументы повторяют C API один в один.
    #[allow(clippy::too_many_arguments)]
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
        const OP: &str = "adamw_step";
        let count = u64::from(num_elements);
        expect_floats(OP, "weights", weights, count)?;
        expect_floats(OP, "grads", grads, count)?;
        expect_floats(OP, "m_state", m_state, count)?;
        expect_floats(OP, "v_state", v_state, count)?;

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

    /// Out = X / RMS(X) × Gamma; X и Out [batch_seq, dim], Gamma [dim].
    pub fn rmsnorm_forward(
        &self,
        x: &VulkanBuffer,
        gamma: &VulkanBuffer,
        out: &VulkanBuffer,
        batch_seq: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), VulkanError> {
        const OP: &str = "rmsnorm_forward";
        let elements = u64::from(batch_seq) * u64::from(dim);
        expect_floats(OP, "x", x, elements)?;
        expect_floats(OP, "gamma", gamma, u64::from(dim))?;
        expect_floats(OP, "out", out, elements)?;
        let code = unsafe {
            sloth_vk_rmsnorm_forward(x.handle, gamma.handle, out.handle, batch_seq, dim, eps)
        };
        VulkanError::from_code(code)
    }

    /// Градиенты RMSNorm: перезаписывает d_x [batch_seq, dim] и d_gamma [dim].
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_backward(
        &self,
        d_out: &VulkanBuffer,
        x: &VulkanBuffer,
        gamma: &VulkanBuffer,
        d_x: &VulkanBuffer,
        d_gamma: &VulkanBuffer,
        batch_seq: u32,
        dim: u32,
        eps: f32,
    ) -> Result<(), VulkanError> {
        const OP: &str = "rmsnorm_backward";
        let elements = u64::from(batch_seq) * u64::from(dim);
        expect_floats(OP, "d_out", d_out, elements)?;
        expect_floats(OP, "x", x, elements)?;
        expect_floats(OP, "gamma", gamma, u64::from(dim))?;
        expect_floats(OP, "d_x", d_x, elements)?;
        expect_floats(OP, "d_gamma", d_gamma, u64::from(dim))?;
        let code = unsafe {
            sloth_vk_rmsnorm_backward(
                d_out.handle,
                x.handle,
                gamma.handle,
                d_x.handle,
                d_gamma.handle,
                batch_seq,
                dim,
                eps,
            )
        };
        VulkanError::from_code(code)
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for u8 {}
}

/// Типы, которые можно копировать в буфер GPU и обратно байт в байт: без указателей и
/// с любым набором битов в роли допустимого значения. Раньше `read` принимал любой `Copy`
/// (например `bool` или `&T`), и чтение байтов с GPU в такой тип было неопределённым поведением.
pub trait GpuScalar: sealed::Sealed + Copy + Default {}
impl GpuScalar for f32 {}
impl GpuScalar for u32 {}
impl GpuScalar for i32 {}
impl GpuScalar for u8 {}

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

    pub fn write<T: GpuScalar>(&self, data: &[T]) -> Result<(), VulkanError> {
        let byte_len = std::mem::size_of_val(data);
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

    pub fn read<T: GpuScalar>(&self, data: &mut [T]) -> Result<(), VulkanError> {
        let byte_len = std::mem::size_of_val(data);
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

    pub fn read_to_vec<T: GpuScalar>(&self, count: usize) -> Result<Vec<T>, VulkanError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_version_is_decoded() {
        // VK_MAKE_API_VERSION(0, 1, 4, 328)
        let version = (1 << 22) | (4 << 12) | 328;
        assert_eq!(format_api_version(version), "1.4.328");
    }

    #[test]
    fn c_strings_stop_at_nul() {
        let mut raw = [0 as c_char; 8];
        for (slot, byte) in raw.iter_mut().zip(b"radv") {
            *slot = *byte as c_char;
        }
        assert_eq!(c_string(&raw), "radv");
        assert_eq!(c_string(&[0; 4]), "");
    }
}
