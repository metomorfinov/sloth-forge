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
