//! Сверка GPU-операций с эталоном на CPU на неоднородных данных «неудобных» размеров
//! (не кратных размерам рабочих групп шейдеров). Раньше тесты проверяли только
//! «результат больше нуля» на массивах из одинаковых чисел, и ошибки индексации
//! в шейдерах оставались незамеченными.

use sloth_vulkan_sys::{
    sloth_vk_rmsnorm_backward, sloth_vk_rmsnorm_forward, VulkanBuffer, VulkanContext, VulkanError,
    SLOTH_VK_LORA_MAX_RANK,
};

/// Допустимое относительное расхождение GPU и CPU (float32 против float64-эталона).
const RELATIVE_TOLERANCE: f64 = 1e-3;
/// Сколько раз повторять прогон: ошибки гонок между потоками GPU проявляются не всегда.
const REPEATS: usize = 3;
const RMS_EPS: f32 = 1e-5;

/// Детерминированный генератор чисел в [-1, 1): тесты не зависят от внешних крейтов.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vec(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next()).collect()
    }
}

fn upload(ctx: &VulkanContext, data: &[f32]) -> VulkanBuffer {
    let buf = ctx
        .alloc_buffer(std::mem::size_of_val(data), true)
        .expect("буфер выделен");
    buf.write(data).expect("данные записаны");
    buf
}

fn zeros(ctx: &VulkanContext, len: usize) -> VulkanBuffer {
    upload(ctx, &vec![0.0; len])
}

/// Наибольшее относительное расхождение двух массивов.
fn max_relative_error(gpu: &[f32], cpu: &[f64]) -> f64 {
    assert_eq!(gpu.len(), cpu.len());
    gpu.iter()
        .zip(cpu)
        .map(|(&g, &c)| (g as f64 - c).abs() / c.abs().max(1.0))
        .fold(0.0, f64::max)
}

struct Report {
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, name: &str, gpu: Result<Vec<f32>, VulkanError>, cpu: &[f64]) {
        let gpu = match gpu {
            Ok(values) => values,
            Err(err) => {
                self.failures.push(format!("{name}: ошибка {err}"));
                return;
            }
        };
        let error = max_relative_error(&gpu, cpu);
        eprintln!("{name}: максимальное расхождение {error:.2e}");
        if !error.is_finite() || error > RELATIVE_TOLERANCE {
            self.failures
                .push(format!("{name}: расхождение {error:.2e}"));
        }
    }
}

fn check_gemm(
    ctx: &VulkanContext,
    rng: &mut Lcg,
    report: &mut Report,
    (m, k, n): (usize, usize, usize),
) {
    let a = rng.vec(m * k);
    let b = rng.vec(k * n);
    let mut expected = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = (0..k)
                .map(|p| a[i * k + p] as f64 * b[p * n + j] as f64)
                .sum();
        }
    }
    let (ga, gb, gc) = (upload(ctx, &a), upload(ctx, &b), zeros(ctx, m * n));
    let result = ctx
        .forward_gemm(&ga, &gb, &gc, m as u32, k as u32, n as u32)
        .and_then(|_| gc.read_to_vec(m * n));
    report.check(&format!("gemm {m}x{k}x{n}"), result, &expected);
}

/// LoRA в раскладке PEFT: W [out,in], A [rank,in], B [out,rank].
fn check_lora(
    ctx: &VulkanContext,
    rng: &mut Lcg,
    report: &mut Report,
    (t, d_in, d_out, rank): (usize, usize, usize, usize),
) {
    let alpha = 7.0f32;
    let scale = alpha as f64 / rank as f64;
    let x = rng.vec(t * d_in);
    let w = rng.vec(d_out * d_in);
    let la = rng.vec(rank * d_in);
    let lb = rng.vec(d_out * rank);
    let d_output = rng.vec(t * d_out);

    let mut h = vec![0.0f64; t * rank];
    for tok in 0..t {
        for r in 0..rank {
            h[tok * rank + r] = (0..d_in)
                .map(|p| x[tok * d_in + p] as f64 * la[r * d_in + p] as f64)
                .sum();
        }
    }
    let mut out_ref = vec![0.0f64; t * d_out];
    for tok in 0..t {
        for o in 0..d_out {
            let base: f64 = (0..d_in)
                .map(|p| x[tok * d_in + p] as f64 * w[o * d_in + p] as f64)
                .sum();
            let lora: f64 = (0..rank)
                .map(|r| lb[o * rank + r] as f64 * h[tok * rank + r])
                .sum();
            out_ref[tok * d_out + o] = base + scale * lora;
        }
    }
    let mut db_ref = vec![0.0f64; d_out * rank];
    for o in 0..d_out {
        for r in 0..rank {
            db_ref[o * rank + r] = scale
                * (0..t)
                    .map(|tok| d_output[tok * d_out + o] as f64 * h[tok * rank + r])
                    .sum::<f64>();
        }
    }
    let mut da_ref = vec![0.0f64; rank * d_in];
    for r in 0..rank {
        for p in 0..d_in {
            da_ref[r * d_in + p] = scale
                * (0..t)
                    .map(|tok| {
                        let dh: f64 = (0..d_out)
                            .map(|o| d_output[tok * d_out + o] as f64 * lb[o * rank + r] as f64)
                            .sum();
                        dh * x[tok * d_in + p] as f64
                    })
                    .sum::<f64>();
        }
    }

    let label = format!("T={t} in={d_in} out={d_out} rank={rank}");
    let (gx, gw, gla, glb) = (
        upload(ctx, &x),
        upload(ctx, &w),
        upload(ctx, &la),
        upload(ctx, &lb),
    );
    let gout = zeros(ctx, t * d_out);
    let (t32, in32, out32, rank32) = (t as u32, d_in as u32, d_out as u32, rank as u32);
    let forward = ctx.forward_lora(
        &gx,
        Some(&gw),
        &gla,
        &glb,
        &gout,
        1,
        t32,
        in32,
        out32,
        rank32,
        alpha,
    );
    if rank32 > SLOTH_VK_LORA_MAX_RANK {
        // Раньше ранг молча обрезался до 128 и результат был неверным
        if forward != Err(VulkanError::InvalidParam) {
            report.failures.push(format!(
                "lora_forward {label}: ожидалась ошибка InvalidParam, получено {forward:?}"
            ));
        }
    } else {
        report.check(
            &format!("lora_forward {label}"),
            forward.and_then(|_| gout.read_to_vec(t * d_out)),
            &out_ref,
        );
    }

    let gdout = upload(ctx, &d_output);
    let (gda, gdb) = (zeros(ctx, rank * d_in), zeros(ctx, d_out * rank));
    let backward = ctx.backward_lora(
        &gx, &gdout, &gla, &glb, &gda, &gdb, 1, t32, in32, out32, rank32, alpha,
    );
    report.check(
        &format!("lora_backward dA {label}"),
        backward.clone().and_then(|_| gda.read_to_vec(rank * d_in)),
        &da_ref,
    );
    report.check(
        &format!("lora_backward dB {label}"),
        backward.and_then(|_| gdb.read_to_vec(d_out * rank)),
        &db_ref,
    );
}

fn check_rmsnorm(
    ctx: &VulkanContext,
    rng: &mut Lcg,
    report: &mut Report,
    (t, dim): (usize, usize),
) {
    let x = rng.vec(t * dim);
    let gamma = rng.vec(dim);
    let d_output = rng.vec(t * dim);
    let eps = RMS_EPS as f64;

    let mut out_ref = vec![0.0f64; t * dim];
    let mut dx_ref = vec![0.0f64; t * dim];
    let mut dgamma_ref = vec![0.0f64; dim];
    for row in 0..t {
        let xs = &x[row * dim..(row + 1) * dim];
        let mean_sq = xs.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / dim as f64 + eps;
        let inv_rms = 1.0 / mean_sq.sqrt();
        let dot: f64 = (0..dim)
            .map(|d| d_output[row * dim + d] as f64 * gamma[d] as f64 * xs[d] as f64)
            .sum();
        for d in 0..dim {
            let xi = xs[d] as f64;
            out_ref[row * dim + d] = xi * inv_rms * gamma[d] as f64;
            dx_ref[row * dim + d] = d_output[row * dim + d] as f64 * gamma[d] as f64 * inv_rms
                - xi * inv_rms.powi(3) * dot / dim as f64;
            dgamma_ref[d] += d_output[row * dim + d] as f64 * xi * inv_rms;
        }
    }

    let label = format!("T={t} dim={dim}");
    let (gx, ggamma, gout) = (upload(ctx, &x), upload(ctx, &gamma), zeros(ctx, t * dim));
    let code = unsafe {
        sloth_vk_rmsnorm_forward(
            gx.handle(),
            ggamma.handle(),
            gout.handle(),
            t as u32,
            dim as u32,
            RMS_EPS,
        )
    };
    report.check(
        &format!("rmsnorm_forward {label}"),
        VulkanError::from_code(code).and_then(|_| gout.read_to_vec(t * dim)),
        &out_ref,
    );

    let gdout = upload(ctx, &d_output);
    let (gdx, gdgamma) = (zeros(ctx, t * dim), zeros(ctx, dim));
    let code = unsafe {
        sloth_vk_rmsnorm_backward(
            gdout.handle(),
            gx.handle(),
            ggamma.handle(),
            gdx.handle(),
            gdgamma.handle(),
            t as u32,
            dim as u32,
            RMS_EPS,
        )
    };
    let backward = VulkanError::from_code(code);
    report.check(
        &format!("rmsnorm_backward dX {label}"),
        backward.clone().and_then(|_| gdx.read_to_vec(t * dim)),
        &dx_ref,
    );
    report.check(
        &format!("rmsnorm_backward dGamma {label}"),
        backward.and_then(|_| gdgamma.read_to_vec(dim)),
        &dgamma_ref,
    );
}

fn check_adamw(ctx: &VulkanContext, rng: &mut Lcg, report: &mut Report, count: usize) {
    let (lr, beta1, beta2, eps, wd, step) = (1e-2f32, 0.9f32, 0.999f32, 1e-8f32, 0.01f32, 3u32);
    let weights = rng.vec(count);
    let grads = rng.vec(count);
    let m_state: Vec<f32> = rng.vec(count).iter().map(|v| v * 0.1).collect();
    let v_state: Vec<f32> = rng.vec(count).iter().map(|v| v.abs() * 0.01).collect();
    let bc1 = 1.0 - (beta1 as f64).powi(step as i32);
    let bc2 = 1.0 - (beta2 as f64).powi(step as i32);
    let expected: Vec<f64> = (0..count)
        .map(|i| {
            let mut wi = weights[i] as f64;
            wi -= lr as f64 * wd as f64 * wi;
            let mi = beta1 as f64 * m_state[i] as f64 + (1.0 - beta1 as f64) * grads[i] as f64;
            let vi =
                beta2 as f64 * v_state[i] as f64 + (1.0 - beta2 as f64) * (grads[i] as f64).powi(2);
            wi - lr as f64 * (mi / bc1) / ((vi / bc2).sqrt() + eps as f64)
        })
        .collect();
    let (gw, gg, gm, gv) = (
        upload(ctx, &weights),
        upload(ctx, &grads),
        upload(ctx, &m_state),
        upload(ctx, &v_state),
    );
    let result = ctx
        .adamw_step(
            &gw,
            &gg,
            &gm,
            &gv,
            count as u32,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            step,
        )
        .and_then(|_| gw.read_to_vec(count));
    report.check(&format!("adamw n={count}"), result, &expected);
}

#[test]
fn gpu_kernels_match_cpu_reference() {
    let Ok(ctx) = VulkanContext::init(true) else {
        eprintln!("пропуск: Vulkan недоступен");
        return;
    };
    let mut rng = Lcg(0x5107_4F09_2026_0913);
    let mut report = Report {
        failures: Vec::new(),
    };

    for _ in 0..REPEATS {
        check_gemm(&ctx, &mut rng, &mut report, (37, 45, 29));
        check_gemm(&ctx, &mut rng, &mut report, (70, 130, 90));
        check_lora(&ctx, &mut rng, &mut report, (7, 50, 33, 19));
        check_lora(
            &ctx,
            &mut rng,
            &mut report,
            (3, 64, 20, SLOTH_VK_LORA_MAX_RANK as usize),
        );
        check_lora(&ctx, &mut rng, &mut report, (5, 70, 40, 200));
        check_rmsnorm(&ctx, &mut rng, &mut report, (5, 50));
        check_rmsnorm(&ctx, &mut rng, &mut report, (4, 300));
        check_adamw(&ctx, &mut rng, &mut report, 203);
    }

    assert!(
        report.failures.is_empty(),
        "расхождения GPU и CPU: {:#?}",
        report.failures
    );
}
