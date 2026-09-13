use std::ffi::{c_char, c_int, c_void};

pub type SlothBufferHandle = u64;
pub const SLOTH_NULL_BUFFER: SlothBufferHandle = 0;

pub const SLOTH_VK_SUCCESS: c_int = 0;
pub const SLOTH_VK_ERROR_INIT_FAILED: c_int = -1;
pub const SLOTH_VK_ERROR_NO_DEVICE: c_int = -2;
pub const SLOTH_VK_ERROR_OUT_OF_MEMORY: c_int = -3;
pub const SLOTH_VK_ERROR_INVALID_PARAM: c_int = -4;
pub const SLOTH_VK_ERROR_SHADER_FAILED: c_int = -5;
pub const SLOTH_VK_ERROR_DISPATCH_FAILED: c_int = -6;
pub const SLOTH_VK_ERROR_NOT_INITIALIZED: c_int = -7;
pub const SLOTH_VK_ERROR_NOT_SUPPORTED: c_int = -8;

/// Наибольший ранг LoRA, который поддерживает шейдер прямого прохода.
pub const SLOTH_VK_LORA_MAX_RANK: u32 = 128;

/// Размер строковых полей [`SlothDeviceInfo`].
pub const SLOTH_VK_INFO_STRING_SIZE: usize = 256;

pub const SLOTH_VK_DEVICE_TYPE_OTHER: u32 = 0;
pub const SLOTH_VK_DEVICE_TYPE_INTEGRATED_GPU: u32 = 1;
pub const SLOTH_VK_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;
pub const SLOTH_VK_DEVICE_TYPE_VIRTUAL_GPU: u32 = 3;
pub const SLOTH_VK_DEVICE_TYPE_CPU: u32 = 4;

/// Повторяет `SlothDeviceInfo` из `sloth_vulkan.h` поле в поле.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlothDeviceInfo {
    pub device_name: [c_char; SLOTH_VK_INFO_STRING_SIZE],
    pub driver_name: [c_char; SLOTH_VK_INFO_STRING_SIZE],
    pub driver_info: [c_char; SLOTH_VK_INFO_STRING_SIZE],
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
    pub driver_version: u32,
    pub device_type: u32,
    pub subgroup_size: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub memory_budget_supported: u32,
    pub device_local_bytes: u64,
}

// Несовпадение раскладки с C-структурой испортило бы память: проверяем при компиляции
const _: () =
    assert!(std::mem::size_of::<SlothDeviceInfo>() == 3 * SLOTH_VK_INFO_STRING_SIZE + 10 * 4 + 8);

impl Default for SlothDeviceInfo {
    fn default() -> Self {
        Self {
            device_name: [0; SLOTH_VK_INFO_STRING_SIZE],
            driver_name: [0; SLOTH_VK_INFO_STRING_SIZE],
            driver_info: [0; SLOTH_VK_INFO_STRING_SIZE],
            vendor_id: 0,
            device_id: 0,
            api_version: 0,
            driver_version: 0,
            device_type: SLOTH_VK_DEVICE_TYPE_OTHER,
            subgroup_size: 0,
            max_compute_work_group_count: [0; 3],
            memory_budget_supported: 0,
            device_local_bytes: 0,
        }
    }
}

extern "C" {
    pub fn sloth_vk_init(
        prefer_discrete: c_int,
        out_device_name: *mut c_char,
        max_len: usize,
    ) -> c_int;

    pub fn sloth_vk_shutdown();

    pub fn sloth_vk_get_vram_info(
        total_bytes: *mut u64,
        used_bytes: *mut u64,
        free_bytes: *mut u64,
    ) -> c_int;

    pub fn sloth_vk_get_device_info(out: *mut SlothDeviceInfo) -> c_int;

    pub fn sloth_vk_get_memory_budget(budget_bytes: *mut u64, usage_bytes: *mut u64) -> c_int;

    pub fn sloth_vk_alloc_buffer(size_bytes: usize, is_device_local: c_int) -> SlothBufferHandle;

    pub fn sloth_vk_free_buffer(handle: SlothBufferHandle);

    pub fn sloth_vk_write_buffer(
        handle: SlothBufferHandle,
        src: *const c_void,
        size_bytes: usize,
    ) -> c_int;

    pub fn sloth_vk_read_buffer(
        handle: SlothBufferHandle,
        dst: *mut c_void,
        size_bytes: usize,
    ) -> c_int;

    pub fn sloth_vk_forward_gemm(
        a: SlothBufferHandle,
        b: SlothBufferHandle,
        c: SlothBufferHandle,
        m: u32,
        k: u32,
        n: u32,
    ) -> c_int;

    pub fn sloth_vk_forward_lora(
        x: SlothBufferHandle,
        w_base: SlothBufferHandle,
        a_lora: SlothBufferHandle,
        b_lora: SlothBufferHandle,
        out: SlothBufferHandle,
        batch: u32,
        seq: u32,
        in_dim: u32,
        out_dim: u32,
        rank: u32,
        alpha: f32,
    ) -> c_int;

    pub fn sloth_vk_backward_lora(
        x: SlothBufferHandle,
        d_out: SlothBufferHandle,
        a_lora: SlothBufferHandle,
        b_lora: SlothBufferHandle,
        d_a_grad: SlothBufferHandle,
        d_b_grad: SlothBufferHandle,
        batch: u32,
        seq: u32,
        in_dim: u32,
        out_dim: u32,
        rank: u32,
        alpha: f32,
    ) -> c_int;

    pub fn sloth_vk_adamw_step(
        weights: SlothBufferHandle,
        grads: SlothBufferHandle,
        m_state: SlothBufferHandle,
        v_state: SlothBufferHandle,
        num_elements: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> c_int;

    pub fn sloth_vk_rmsnorm_forward(
        x: SlothBufferHandle,
        gamma: SlothBufferHandle,
        out: SlothBufferHandle,
        batch_seq: u32,
        dim: u32,
        eps: f32,
    ) -> c_int;

    pub fn sloth_vk_rmsnorm_backward(
        d_out: SlothBufferHandle,
        x: SlothBufferHandle,
        gamma: SlothBufferHandle,
        d_x: SlothBufferHandle,
        d_gamma: SlothBufferHandle,
        batch_seq: u32,
        dim: u32,
        eps: f32,
    ) -> c_int;
}
