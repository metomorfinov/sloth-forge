//! Телеметрия видеокарты для `/api/health`, `/api/system`, `/api/hardware`, `/api/train/hardware`.
//!
//! Раньше эти эндпоинты отдавали константы: «AMD Radeon RX 570», 4 ГБ, «Mesa 24.0.0»,
//! занятые 1420 МБ, температуру «68 + номер шага % 4» °C. Здесь:
//! - имя и объём видеопамяти берутся у выбранного Vulkan-устройства;
//! - на Linux с драйвером amdgpu из sysfs читаются занятая VRAM, загрузка GPU, температура,
//!   мощность, её предел и обороты вентилятора;
//! - чего узнать нельзя, остаётся `None` (в JSON — `null`), а не придуманное число.
//!
//! Где sysfs нет (Windows, видеокарты не AMD), занятость видеопамяти оценивается по бюджету
//! `VK_EXT_memory_budget`, а драйвер и версия Vulkan берутся у Vulkan-устройства.

use crate::state::AppState;
use serde_json::{json, Value};
use sloth_vulkan_sys::{MemoryBudget, VulkanContext, VulkanError};
use std::fs;
use std::path::{Path, PathBuf};

/// Где Linux показывает видеокарты.
const DRM_CLASS_DIR: &str = "/sys/class/drm";
const AMD_VENDOR_ID: &str = "0x1002";
const AMDGPU_HWMON_NAME: &str = "amdgpu";
/// Имя бэкенда, которое фронтенд понимает для Vulkan-устройств.
const VULKAN_BACKEND: &str = "vulkan";
const BYTES_PER_MIB: u64 = 1024 * 1024;
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MILLIDEGREES_PER_DEGREE: f64 = 1000.0;
const MICROWATTS_PER_WATT: f64 = 1_000_000.0;
const PERCENT: f64 = 100.0;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuTelemetry {
    pub name: Option<String>,
    pub vram_total_bytes: Option<u64>,
    /// Занято всеми программами: рабочим столом, браузером и SlothForge.
    pub vram_used_bytes: Option<u64>,
    pub utilization_pct: Option<f64>,
    pub temperature_c: Option<f64>,
    pub power_draw_w: Option<f64>,
    pub power_limit_w: Option<f64>,
    pub fan_rpm: Option<u64>,
    /// Драйвер ядра (например, «amdgpu»).
    pub kernel_driver: Option<String>,
    /// Vulkan-драйвер и его версия (например, «radv» и «Mesa 26.2.2»).
    pub driver_name: Option<String>,
    pub driver_info: Option<String>,
    /// Версия Vulkan API устройства (например, «1.4.328»).
    pub vulkan_api_version: Option<String>,
}

impl GpuTelemetry {
    /// Видеокарта найдена (через Vulkan или драйвер).
    pub fn available(&self) -> bool {
        self.name.is_some() || self.vram_total_bytes.is_some()
    }

    pub fn vram_free_bytes(&self) -> Option<u64> {
        Some(self.vram_total_bytes?.saturating_sub(self.vram_used_bytes?))
    }

    pub fn vram_utilization_pct(&self) -> Option<f64> {
        let total = self.vram_total_bytes.filter(|total| *total > 0)?;
        Some(round(
            self.vram_used_bytes? as f64 / total as f64 * PERCENT,
            1,
        ))
    }

    pub fn power_utilization_pct(&self) -> Option<f64> {
        let limit = self.power_limit_w.filter(|limit| *limit > 0.0)?;
        Some(round(self.power_draw_w? / limit * PERCENT, 1))
    }
}

pub fn round(value: f64, digits: i32) -> f64 {
    let factor = 10f64.powi(digits);
    (value * factor).round() / factor
}

pub fn bytes_to_mib(bytes: u64) -> u64 {
    bytes / BYTES_PER_MIB
}

pub fn bytes_to_gib(bytes: u64) -> f64 {
    round(bytes as f64 / BYTES_PER_GIB, 2)
}

/// Платформа сервера в словаре фронтенда (`DeviceType`: mac / windows / linux).
pub fn host_platform() -> &'static str {
    match std::env::consts::OS {
        "macos" => "mac",
        other => other,
    }
}

/// `vulkan`, если видеокарта найдена, иначе `cpu`.
pub fn device_backend(gpu: &GpuTelemetry) -> &'static str {
    if gpu.available() {
        VULKAN_BACKEND
    } else {
        "cpu"
    }
}

/// Снимок телеметрии без блокировки асинхронного сервера.
pub async fn collect_for(state: &AppState) -> GpuTelemetry {
    let vk = state.vk_ctx.clone();
    match tokio::task::spawn_blocking(move || collect(vk.as_ref(), Path::new(DRM_CLASS_DIR))).await
    {
        Ok(telemetry) => telemetry,
        Err(err) => {
            tracing::error!("Сбор телеметрии видеокарты прерван: {err}");
            GpuTelemetry::default()
        }
    }
}

/// Синхронная: читает sysfs, вызывать через `spawn_blocking`.
pub fn collect(vk: Option<&VulkanContext>, drm_dir: &Path) -> GpuTelemetry {
    let mut telemetry = GpuTelemetry::default();
    if let Some(ctx) = vk {
        telemetry.name = Some(ctx.device_name().trim().to_string()).filter(|name| !name.is_empty());
        match ctx.device_info() {
            Ok(info) => {
                telemetry.vram_total_bytes =
                    Some(info.device_local_bytes).filter(|total| *total > 0);
                telemetry.driver_name = info.driver_name;
                telemetry.driver_info = info.driver_info;
                telemetry.vulkan_api_version = Some(info.api_version);
            }
            Err(err) => tracing::warn!("Vulkan не сообщил сведения об устройстве: {err}"),
        }
    }
    if sysfs_matches_vulkan_name(telemetry.name.as_deref()) {
        let cards = amd_cards(drm_dir);
        if let Some(card) = pick_card(&cards, telemetry.vram_total_bytes) {
            read_card(&mut telemetry, card);
        }
    }
    // Без sysfs (Windows, не AMD) занятость оцениваем по бюджету Vulkan
    if telemetry.vram_used_bytes.is_none() {
        if let (Some(ctx), Some(total)) = (vk, telemetry.vram_total_bytes) {
            match ctx.memory_budget() {
                Ok(budget) => telemetry.vram_used_bytes = Some(used_from_budget(total, budget)),
                Err(VulkanError::NotSupported) => {}
                Err(err) => tracing::debug!("Бюджет видеопамяти недоступен: {err}"),
            }
        }
    }
    telemetry
}

/// Занято всеми программами ≈ объём − доступное процессу + занятое самим процессом.
/// Драйвер считает бюджет как свободную память плюс уже занятую процессом, поэтому это
/// оценка, а не точное число, как в sysfs.
fn used_from_budget(total: u64, budget: MemoryBudget) -> u64 {
    total
        .saturating_sub(budget.budget_bytes)
        .saturating_add(budget.usage_bytes)
        .min(total)
}

/// Данные amdgpu относятся к Vulkan-устройству, только если это видеокарта AMD:
/// иначе при NVIDIA + встроенной AMD показались бы чужие числа.
fn sysfs_matches_vulkan_name(vulkan_name: Option<&str>) -> bool {
    vulkan_name.is_none_or(|name| name.contains("AMD") || name.contains("Radeon"))
}

#[derive(Debug, Clone, PartialEq)]
struct Card {
    device_dir: PathBuf,
    vram_total_bytes: u64,
}

fn read_trimmed(path: &Path) -> Option<String> {
    match fs::read_to_string(path) {
        Ok(text) => Some(text.trim().to_string()),
        // Отсутствующий файл — обычное дело (другой драйвер или старое ядро)
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => {
            tracing::debug!("Не удалось прочитать {}: {err}", path.display());
            None
        }
    }
}

fn read_u64(path: &Path) -> Option<u64> {
    read_trimmed(path)?.parse().ok()
}

/// Видеокарты AMD с драйвером amdgpu (у них есть `mem_info_vram_total`).
fn amd_cards(drm_dir: &Path) -> Vec<Card> {
    let Ok(entries) = fs::read_dir(drm_dir) else {
        return Vec::new();
    };
    let mut cards: Vec<Card> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            // card0, card1…; записи вида card1-DP-1 — это разъёмы, а не устройства
            let is_card = name.strip_prefix("card").is_some_and(|number| {
                !number.is_empty() && number.chars().all(|c| c.is_ascii_digit())
            });
            if !is_card {
                return None;
            }
            let device_dir = entry.path().join("device");
            if read_trimmed(&device_dir.join("vendor")).as_deref() != Some(AMD_VENDOR_ID) {
                return None;
            }
            let vram_total_bytes = read_u64(&device_dir.join("mem_info_vram_total"))?;
            Some(Card {
                device_dir,
                vram_total_bytes,
            })
        })
        .collect();
    cards.sort_by(|a, b| a.device_dir.cmp(&b.device_dir));
    cards
}

/// Карта Vulkan-устройства: при нескольких AMD — с ближайшим объёмом VRAM.
fn pick_card(cards: &[Card], vulkan_total: Option<u64>) -> Option<&Card> {
    match vulkan_total {
        Some(total) => cards
            .iter()
            .min_by_key(|card| card.vram_total_bytes.abs_diff(total)),
        None => cards.iter().max_by_key(|card| card.vram_total_bytes),
    }
}

fn amdgpu_hwmon(device_dir: &Path) -> Option<PathBuf> {
    fs::read_dir(device_dir.join("hwmon"))
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| read_trimmed(&path.join("name")).as_deref() == Some(AMDGPU_HWMON_NAME))
}

fn read_card(telemetry: &mut GpuTelemetry, card: &Card) {
    telemetry.vram_total_bytes = Some(card.vram_total_bytes);
    telemetry.vram_used_bytes = read_u64(&card.device_dir.join("mem_info_vram_used"));
    telemetry.utilization_pct =
        read_u64(&card.device_dir.join("gpu_busy_percent")).map(|percent| percent as f64);
    telemetry.kernel_driver = fs::read_link(card.device_dir.join("driver"))
        .ok()
        .and_then(|link| {
            link.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        });
    if let Some(hwmon) = amdgpu_hwmon(&card.device_dir) {
        telemetry.temperature_c = read_u64(&hwmon.join("temp1_input"))
            .map(|millidegrees| round(millidegrees as f64 / MILLIDEGREES_PER_DEGREE, 1));
        // power1_average есть на старых ядрах, power1_input — на новых
        telemetry.power_draw_w = read_u64(&hwmon.join("power1_average"))
            .or_else(|| read_u64(&hwmon.join("power1_input")))
            .map(|microwatts| round(microwatts as f64 / MICROWATTS_PER_WATT, 1));
        telemetry.power_limit_w = read_u64(&hwmon.join("power1_cap"))
            .map(|microwatts| round(microwatts as f64 / MICROWATTS_PER_WATT, 1));
        telemetry.fan_rpm = read_u64(&hwmon.join("fan1_input"));
    }
}

/// Блок `gpu` / `inference_gpu` для `/api/system` (`SystemGpuInfo` во фронтенде).
pub fn system_gpu_json(gpu: &GpuTelemetry) -> Value {
    if !gpu.available() {
        // Бэкенд «cpu», а не «vulkan»: иначе интерфейс каждые 3 секунды переспрашивал бы,
        // не появилась ли Vulkan-видеокарта
        return json!({ "available": false, "backend": "cpu", "devices": [] });
    }
    let total_gb = gpu.vram_total_bytes.map(bytes_to_gib);
    json!({
        "available": true,
        "backend": VULKAN_BACKEND,
        "index_kind": "vulkan",
        // Загрузка модели на выбранные по номеру GPU пока не поддерживается
        "gguf_gpu_ids_supported": false,
        "devices": [{
            "index": 0,
            "visible_ordinal": 0,
            "index_kind": "vulkan",
            "name": gpu.name,
            "gpu_name": gpu.name,
            "memory_total_gb": total_gb,
            "vram_total_gb": total_gb,
            "vram_used_gb": gpu.vram_used_bytes.map(bytes_to_gib),
            "vram_free_gb": gpu.vram_free_bytes().map(bytes_to_gib),
            "vram_utilization_pct": gpu.vram_utilization_pct(),
            "backend": VULKAN_BACKEND,
            "shared_memory": false,
            "unified_memory": false
        }]
    })
}

/// `/api/train/hardware` (`GpuUtilization` во фронтенде): живые показатели для панели обучения.
pub fn utilization_json(gpu: &GpuTelemetry) -> Value {
    json!({
        "available": gpu.available(),
        "backend": gpu.available().then_some(VULKAN_BACKEND),
        "index": 0,
        "visible_ordinal": 0,
        "gpu_utilization_pct": gpu.utilization_pct,
        "temperature_c": gpu.temperature_c,
        "vram_used_gb": gpu.vram_used_bytes.map(bytes_to_gib),
        "vram_total_gb": gpu.vram_total_bytes.map(bytes_to_gib),
        "vram_utilization_pct": gpu.vram_utilization_pct(),
        "power_draw_w": gpu.power_draw_w,
        "power_limit_w": gpu.power_limit_w,
        "power_utilization_pct": gpu.power_utilization_pct(),
        "fan_rpm": gpu.fan_rpm
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    /// Дерево sysfs как у RX 570: встроенная карта Intel, AMD с hwmon и разъём.
    fn fake_sysfs() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let drm = dir.path();
        write(&drm.join("card0/device/vendor"), "0x8086\n");
        let amd = drm.join("card1/device");
        write(&amd.join("vendor"), "0x1002\n");
        write(&amd.join("mem_info_vram_total"), "4294967296\n");
        write(&amd.join("mem_info_vram_used"), "1073741824\n");
        write(&amd.join("gpu_busy_percent"), "37\n");
        let hwmon = amd.join("hwmon/hwmon2");
        write(&hwmon.join("name"), "amdgpu\n");
        write(&hwmon.join("temp1_input"), "48000\n");
        write(&hwmon.join("power1_input"), "21180000\n");
        write(&hwmon.join("power1_cap"), "90000000\n");
        write(&hwmon.join("fan1_input"), "1015\n");
        fs::create_dir_all(drm.join("card1-DP-1")).unwrap();
        dir
    }

    #[test]
    fn reads_amdgpu_sysfs() {
        let sysfs = fake_sysfs();
        let gpu = collect(None, sysfs.path());
        assert!(gpu.available());
        assert_eq!(gpu.vram_total_bytes, Some(4 * 1024 * 1024 * 1024));
        assert_eq!(gpu.vram_used_bytes, Some(1024 * 1024 * 1024));
        assert_eq!(gpu.vram_free_bytes(), Some(3 * 1024 * 1024 * 1024));
        assert_eq!(gpu.vram_utilization_pct(), Some(25.0));
        assert_eq!(gpu.utilization_pct, Some(37.0));
        assert_eq!(gpu.temperature_c, Some(48.0));
        assert_eq!(gpu.power_draw_w, Some(21.2));
        assert_eq!(gpu.power_limit_w, Some(90.0));
        assert_eq!(gpu.power_utilization_pct(), Some(23.6));
        assert_eq!(gpu.fan_rpm, Some(1015));

        let util = utilization_json(&gpu);
        assert_eq!(util["vram_total_gb"], 4.0);
        assert_eq!(util["temperature_c"], 48.0);
        let system = system_gpu_json(&gpu);
        assert_eq!(system["backend"], "vulkan");
        assert_eq!(system["devices"][0]["vram_used_gb"], 1.0);
    }

    #[test]
    fn nothing_found_means_nulls_not_numbers() {
        let empty = tempfile::tempdir().unwrap();
        let gpu = collect(None, empty.path());
        assert_eq!(gpu, GpuTelemetry::default());
        assert!(!gpu.available());
        let util = utilization_json(&gpu);
        assert_eq!(util["available"], false);
        assert!(util["temperature_c"].is_null());
        assert!(util["vram_total_gb"].is_null());
        let system = system_gpu_json(&gpu);
        assert_eq!(system["backend"], "cpu");
        assert_eq!(system["devices"].as_array().map(Vec::len), Some(0));
        assert_eq!(device_backend(&gpu), "cpu");
    }

    #[test]
    fn picks_card_matching_vulkan_memory() {
        let small = Card {
            device_dir: PathBuf::from("/card0"),
            vram_total_bytes: 4 << 30,
        };
        let large = Card {
            device_dir: PathBuf::from("/card1"),
            vram_total_bytes: 8 << 30,
        };
        let cards = [small.clone(), large.clone()];
        assert_eq!(pick_card(&cards, Some(8 << 30)), Some(&large));
        assert_eq!(pick_card(&cards, Some(4 << 30)), Some(&small));
        assert_eq!(pick_card(&cards, None), Some(&large));
        assert_eq!(pick_card(&[], None), None);
    }

    #[test]
    fn amd_sysfs_is_not_used_for_other_vendors() {
        assert!(sysfs_matches_vulkan_name(Some(
            "AMD Radeon RX 570 Series (RADV POLARIS10)"
        )));
        assert!(sysfs_matches_vulkan_name(None));
        assert!(!sysfs_matches_vulkan_name(Some("NVIDIA GeForce RTX 3060")));
    }

    #[test]
    fn budget_gives_estimate_of_used_memory() {
        let total: u64 = 4 << 30;
        let budget = MemoryBudget {
            budget_bytes: 1_300 << 20,
            usage_bytes: 100 << 20,
        };
        assert_eq!(
            used_from_budget(total, budget),
            total - (1_300 << 20) + (100 << 20)
        );
        // Странный ответ драйвера не даёт «занято больше, чем есть»
        let broken = MemoryBudget {
            budget_bytes: 0,
            usage_bytes: 8 << 30,
        };
        assert_eq!(used_from_budget(total, broken), total);
    }

    #[test]
    fn platform_uses_frontend_names() {
        let platform = host_platform();
        assert_ne!(platform, "macos");
        assert!(!platform.is_empty());
    }
}
