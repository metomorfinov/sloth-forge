pub mod ffi;
pub mod wrapper;

pub use ffi::*;
pub use wrapper::*;

/// С чем собран крейт: `vulkan` (настоящий Vulkan-слой) или `stub` (CPU-заглушка без
/// видеокарты). Выбирает build.rs, см. переменную `SLOTH_VULKAN`.
pub const BACKEND: &str = env!("SLOTH_VK_BACKEND");
/// Значение [`BACKEND`] для CPU-заглушки.
pub const STUB_BACKEND: &str = "stub";

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_vulkan_init_and_vram() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let ctx = VulkanContext::init(true).expect("Vulkan init should succeed");
        assert!(!ctx.device_name().is_empty());
        println!("Device: {}", ctx.device_name());

        let vram = ctx.get_vram_info().expect("VRAM info query should succeed");
        if BACKEND == STUB_BACKEND {
            // Заглушка не выдаёт себя за видеокарту: видеопамяти у неё нет
            assert_eq!(vram.total_bytes, 0);
        } else {
            assert!(vram.total_bytes > 0);
        }
        println!("VRAM: {} MB used / {} MB total", vram.used_mb, vram.total_mb);
    }

    #[test]
    fn test_buffer_allocation_and_transfer() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let ctx = VulkanContext::init(true).unwrap();
        let buf = ctx.alloc_buffer(1024 * 4, false).expect("Alloc buffer");
        assert_eq!(buf.size(), 4096);

        let input_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        buf.write(&input_data).expect("Write buffer");

        let read_data: Vec<f32> = buf.read_to_vec(1024).expect("Read buffer");
        assert_eq!(input_data, read_data);
    }

    #[test]
    fn test_gemm_forward() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let ctx = VulkanContext::init(true).unwrap();
        let m = 2u32;
        let k = 3u32;
        let n = 2u32;

        let a = ctx.alloc_buffer((m * k) as usize * 4, true).unwrap();
        let b = ctx.alloc_buffer((k * n) as usize * 4, true).unwrap();
        let c = ctx.alloc_buffer((m * n) as usize * 4, true).unwrap();

        // A = [[1, 2, 3], [4, 5, 6]]
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // B = [[7, 8], [9, 1], [2, 3]]
        let b_data = vec![7.0f32, 8.0, 9.0, 1.0, 2.0, 3.0];
        a.write(&a_data).unwrap();
        b.write(&b_data).unwrap();

        ctx.forward_gemm(&a, &b, &c, m, k, n).unwrap();

        let c_data: Vec<f32> = c.read_to_vec(4).unwrap();
        // C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
        // C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
        // C[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
        // C[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
        assert!((c_data[0] - 31.0).abs() < 1e-4);
        assert!((c_data[1] - 19.0).abs() < 1e-4);
        assert!((c_data[2] - 85.0).abs() < 1e-4);
        assert!((c_data[3] - 55.0).abs() < 1e-4);
    }

    #[test]
    fn test_lora_forward_backward_adamw() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let ctx = VulkanContext::init(true).unwrap();
        let batch = 1u32;
        let seq = 2u32;
        let in_dim = 4u32;
        let out_dim = 4u32;
        let rank = 2u32;
        let alpha = 4.0f32;

        let x = ctx.alloc_buffer((batch * seq * in_dim) as usize * 4, true).unwrap();
        let w_base = ctx.alloc_buffer((in_dim * out_dim) as usize * 4, true).unwrap();
        let a_lora = ctx.alloc_buffer((in_dim * rank) as usize * 4, true).unwrap();
        let b_lora = ctx.alloc_buffer((rank * out_dim) as usize * 4, true).unwrap();
        let out = ctx.alloc_buffer((batch * seq * out_dim) as usize * 4, true).unwrap();

        let x_data = vec![1.0f32; (batch * seq * in_dim) as usize];
        let a_data = vec![0.5f32; (in_dim * rank) as usize];
        let b_data = vec![0.5f32; (rank * out_dim) as usize];
        x.write(&x_data).unwrap();
        a_lora.write(&a_data).unwrap();
        b_lora.write(&b_data).unwrap();

        ctx.forward_lora(&x, Some(&w_base), &a_lora, &b_lora, &out, batch, seq, in_dim, out_dim, rank, alpha).unwrap();
        let out_res: Vec<f32> = out.read_to_vec((batch * seq * out_dim) as usize).unwrap();
        assert!(out_res.iter().all(|&v| v > 0.0));

        // Backward
        let d_out = ctx.alloc_buffer((batch * seq * out_dim) as usize * 4, true).unwrap();
        d_out.write(&vec![1.0f32; (batch * seq * out_dim) as usize]).unwrap();
        let da = ctx.alloc_buffer((in_dim * rank) as usize * 4, true).unwrap();
        let db = ctx.alloc_buffer((rank * out_dim) as usize * 4, true).unwrap();

        ctx.backward_lora(&x, &d_out, &a_lora, &b_lora, &da, &db, batch, seq, in_dim, out_dim, rank, alpha).unwrap();

        // AdamW
        let m_state = ctx.alloc_buffer((in_dim * rank) as usize * 4, true).unwrap();
        let v_state = ctx.alloc_buffer((in_dim * rank) as usize * 4, true).unwrap();
        ctx.adamw_step(&a_lora, &da, &m_state, &v_state, in_dim * rank, 0.001, 0.9, 0.999, 1e-8, 0.01, 1).unwrap();
    }
}
