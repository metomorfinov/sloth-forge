//! Проверки Vulkan-слоя вне математики операций: сведения об устройстве, бюджет памяти,
//! отказ на неверных размерах, лимиты dispatch и одновременная работа из нескольких потоков.
//! Работают и с настоящим Vulkan, и с CPU-заглушкой (`SLOTH_VULKAN=stub`).

use sloth_vulkan_sys::{
    sloth_vk_forward_gemm, DeviceType, VulkanContext, VulkanError, BACKEND, STUB_BACKEND,
};

const F32_BYTES: u64 = 4;
/// Сколько токенов шейдер LoRA вперёд обрабатывает в одной рабочей группе по оси Y.
const LORA_TOKENS_PER_GROUP: u64 = 4;
/// Тест лимита не выделяет буферы больше этого размера.
const MAX_TEST_BUFFER_BYTES: u64 = 64 * 1024 * 1024;

fn is_stub() -> bool {
    BACKEND == STUB_BACKEND
}

fn context() -> VulkanContext {
    VulkanContext::init(true).expect("Vulkan-контекст создан")
}

#[test]
fn device_info_describes_selected_device() {
    let ctx = context();
    let info = ctx.device_info().expect("сведения об устройстве");
    assert_eq!(info.name, ctx.device_name());
    if is_stub() {
        assert_eq!(info.device_type, DeviceType::Cpu);
        assert_eq!(info.device_local_bytes, 0);
        return;
    }
    eprintln!("{info:#?}");
    assert!(info.device_local_bytes > 0);
    assert!(info.api_version.starts_with("1."), "{}", info.api_version);
    assert!(info
        .max_compute_work_group_count
        .iter()
        .all(|&count| count > 0));
    assert!(!info.name.contains("CPU fallback"));
}

#[test]
fn memory_budget_is_reported_or_honestly_unsupported() {
    let ctx = context();
    let info = ctx.device_info().unwrap();
    match ctx.memory_budget() {
        Ok(budget) => {
            assert!(info.memory_budget_supported);
            assert!(budget.budget_bytes > 0);
            assert!(budget.budget_bytes <= info.device_local_bytes);
            eprintln!("бюджет {budget:?}");
        }
        Err(err) => {
            assert_eq!(err, VulkanError::NotSupported);
            assert!(!info.memory_budget_supported);
        }
    }
}

#[test]
fn undersized_buffers_are_rejected_before_the_gpu() {
    let ctx = context();
    let (m, k, n) = (4u32, 3u32, 5u32);
    let a = ctx.alloc_buffer((m * k) as usize * 4, true).unwrap();
    let b = ctx.alloc_buffer((k * n) as usize * 4, true).unwrap();
    // Для C нужно m*n = 20 чисел, выделено 19
    let c = ctx.alloc_buffer((m * n - 1) as usize * 4, true).unwrap();

    let err = ctx.forward_gemm(&a, &b, &c, m, k, n).unwrap_err();
    assert!(
        matches!(err, VulkanError::ShapeMismatch { buffer: "c", .. }),
        "{err:?}"
    );

    // C API проверяет то же самое, даже если обёртку обойти
    let code = unsafe { sloth_vk_forward_gemm(a.handle(), b.handle(), c.handle(), m, k, n) };
    assert_eq!(VulkanError::from_code(code), Err(VulkanError::InvalidParam));
}

#[test]
fn dispatch_over_device_limit_is_an_error() {
    let ctx = context();
    let info = ctx.device_info().unwrap();
    if is_stub() {
        return;
    }
    // LoRA вперёд кладёт по 4 токена в рабочую группу по оси Y (на RX 570 лимит оси — 65535;
    // по оси X RADV разрешает 2^32-1, поэтому проверяем Y): токенов на группу больше лимита
    let max_groups_y = u64::from(info.max_compute_work_group_count[1]);
    let tokens = (max_groups_y + 1) * LORA_TOKENS_PER_GROUP;
    if tokens * F32_BYTES > MAX_TEST_BUFFER_BYTES {
        eprintln!("пропуск: лимит {max_groups_y} слишком велик для проверки");
        return;
    }
    let buffer = |elements: u64| {
        ctx.alloc_buffer((elements * F32_BYTES) as usize, true)
            .unwrap()
    };
    let (x, a, b, out) = (buffer(tokens), buffer(1), buffer(1), buffer(tokens));
    assert_eq!(
        ctx.forward_lora(&x, None, &a, &b, &out, 1, tokens as u32, 1, 1, 1, 1.0),
        Err(VulkanError::InvalidParam)
    );

    // На единицу меньше лимита — работает
    let tokens = max_groups_y * LORA_TOKENS_PER_GROUP;
    let (x, out) = (buffer(tokens), buffer(tokens));
    assert_eq!(
        ctx.forward_lora(&x, None, &a, &b, &out, 1, tokens as u32, 1, 1, 1, 1.0),
        Ok(())
    );
}

#[test]
fn concurrent_dispatches_from_threads_stay_correct() {
    const THREADS: usize = 4;
    const ITERATIONS: usize = 15;
    let ctx = context();

    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let ctx = ctx.clone();
            scope.spawn(move || {
                let (m, k, n) = (9usize, 17usize, 13usize);
                for iteration in 0..ITERATIONS {
                    // У каждого потока и шага свои данные: перепутанные привязки буферов
                    // между потоками дали бы чужой результат
                    let seed = (thread * 31 + iteration) as f32;
                    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13 + seed).sin()).collect();
                    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.29 - seed).cos()).collect();
                    let expected: Vec<f32> = (0..m * n)
                        .map(|idx| {
                            let (row, col) = (idx / n, idx % n);
                            (0..k).map(|p| a[row * k + p] * b[p * n + col]).sum()
                        })
                        .collect();

                    let ga = ctx.alloc_buffer(a.len() * 4, true).unwrap();
                    let gb = ctx.alloc_buffer(b.len() * 4, true).unwrap();
                    let gc = ctx.alloc_buffer(m * n * 4, true).unwrap();
                    ga.write(&a).unwrap();
                    gb.write(&b).unwrap();
                    ctx.forward_gemm(&ga, &gb, &gc, m as u32, k as u32, n as u32)
                        .unwrap();
                    let actual: Vec<f32> = gc.read_to_vec(m * n).unwrap();
                    for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
                        assert!(
                            (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                            "поток {thread}, шаг {iteration}, элемент {index}: {got} вместо {want}"
                        );
                    }
                }
            });
        }
    });
}

#[test]
fn integer_and_byte_buffers_round_trip() {
    let ctx = context();
    let bytes: Vec<u8> = (0..=255).collect();
    let buf = ctx.alloc_buffer(bytes.len(), true).unwrap();
    buf.write(&bytes).unwrap();
    assert_eq!(buf.read_to_vec::<u8>(bytes.len()).unwrap(), bytes);

    let ints = [-7i32, 0, i32::MAX, i32::MIN];
    let buf = ctx.alloc_buffer(ints.len() * 4, false).unwrap();
    buf.write(&ints).unwrap();
    assert_eq!(buf.read_to_vec::<i32>(ints.len()).unwrap(), ints);

    assert!(matches!(
        buf.write(&[0u32; 5]),
        Err(VulkanError::BufferSizeMismatch {
            expected: 20,
            actual: 16
        })
    ));
}
