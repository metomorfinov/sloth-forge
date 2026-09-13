//! Парсер формата GGUF — файлов весов моделей (формат экосистемы ggml).
//!
//! Устройство файла:
//! 1. заголовок: магическое число `GGUF`, версия, число тензоров и число ключей метаданных;
//! 2. метаданные «ключ → значение» (архитектура, длина контекста, токенизатор…);
//! 3. описания тензоров: имя, размерности, тип квантования, смещение данных;
//! 4. секция данных, выровненная по `general.alignment` (обычно 32 байта).
//!
//! Разбор не доверяет числам из файла: каждая длина строки, массива и каждое число
//! записей сверяется с оставшимися байтами **до** выделения памяти, а данные каждого
//! тензора обязаны лежать внутри файла. Раньше испорченный файл ронял весь сервер,
//! а размеры большинства типов квантования считались неверно («2 байта на вес»).

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

/// Магическое число «GGUF» в порядке байтов little-endian.
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// Выравнивание секции данных, если в метаданных не указано другое.
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// Максимум измерений у тензора в ggml (`GGML_MAX_DIMS`).
pub const MAX_DIMENSIONS: usize = 4;
/// Предел вложенности массивов в метаданных: защита от переполнения стека.
const MAX_VALUE_NESTING: usize = 8;
/// Минимальный размер записи метаданных: длина ключа (8) + тип (4) + значение (≥1).
const MIN_KV_BYTES: usize = 13;
/// Минимальный размер описания тензора: длина имени (8) + число измерений (4) + тип (4) + смещение (8).
const MIN_TENSOR_INFO_BYTES: usize = 24;

#[derive(Error, Debug)]
pub enum GGUFError {
    #[error("Ошибка ввода-вывода: {0}")]
    Io(#[from] std::io::Error),
    #[error("Это не GGUF-файл: магическое число 0x{0:08x} вместо 0x{GGUF_MAGIC:08x}")]
    InvalidMagic(u32),
    #[error("Версия GGUF {0} не поддерживается (нужна 2 или 3)")]
    UnsupportedVersion(u32),
    #[error("Неизвестный тип значения метаданных: {0}")]
    InvalidValueType(u32),
    #[error("Строка в файле не в кодировке UTF-8: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),
    #[error("Файл обрывается при чтении: {0}")]
    Truncated(&'static str),
    #[error("Слишком большое число записей ({count}) для {what}: файл повреждён")]
    CountTooLarge { what: &'static str, count: u64 },
    #[error("Слишком глубокая вложенность массивов в метаданных")]
    NestingTooDeep,
    #[error("Недопустимое выравнивание {0}: должно быть степенью двойки")]
    InvalidAlignment(u64),
    #[error("Ключ метаданных «{0}» встречается дважды")]
    DuplicateKey(String),
    #[error("Тензор «{0}» встречается дважды")]
    DuplicateTensor(String),
    #[error("У тензора «{name}» {dims} измерений (максимум {MAX_DIMENSIONS})")]
    TooManyDimensions { name: String, dims: u32 },
    #[error("У тензора «{0}» переполняется число элементов")]
    ElementCountOverflow(String),
    #[error(
        "Число элементов тензора «{name}» ({elements}) не кратно размеру блока типа {tensor_type}"
    )]
    PartialBlock {
        name: String,
        elements: u64,
        tensor_type: &'static str,
    },
    #[error("Данные тензора «{0}» выходят за пределы файла")]
    TensorOutOfBounds(String),
    #[error("Тензор «{0}» не найден")]
    TensorNotFound(String),
}

/// Устройство блока квантования: сколько весов в блоке и сколько байт он занимает.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLayout {
    pub elements_per_block: u64,
    pub bytes_per_block: u64,
}

/// Тип хранения тензора. Номера совпадают с `enum ggml_type`; размеры блоков
/// сверены с эталонной библиотекой ggml (`ggml_blck_size` / `ggml_type_size`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
    NVFP4,
    Q1_0,
    /// Номер, которого нет в таблице: тип устарел или появился позже этого кода.
    Unknown(u32),
}

/// Таблица типов: (номер ggml, тип, имя, весов в блоке, байт на блок).
const TYPE_TABLE: &[(u32, GGMLType, &str, u64, u64)] = &[
    (0, GGMLType::F32, "F32", 1, 4),
    (1, GGMLType::F16, "F16", 1, 2),
    (2, GGMLType::Q4_0, "Q4_0", 32, 18),
    (3, GGMLType::Q4_1, "Q4_1", 32, 20),
    (6, GGMLType::Q5_0, "Q5_0", 32, 22),
    (7, GGMLType::Q5_1, "Q5_1", 32, 24),
    (8, GGMLType::Q8_0, "Q8_0", 32, 34),
    (9, GGMLType::Q8_1, "Q8_1", 32, 36),
    (10, GGMLType::Q2_K, "Q2_K", 256, 84),
    (11, GGMLType::Q3_K, "Q3_K", 256, 110),
    (12, GGMLType::Q4_K, "Q4_K", 256, 144),
    (13, GGMLType::Q5_K, "Q5_K", 256, 176),
    (14, GGMLType::Q6_K, "Q6_K", 256, 210),
    (15, GGMLType::Q8_K, "Q8_K", 256, 292),
    (16, GGMLType::IQ2_XXS, "IQ2_XXS", 256, 66),
    (17, GGMLType::IQ2_XS, "IQ2_XS", 256, 74),
    (18, GGMLType::IQ3_XXS, "IQ3_XXS", 256, 98),
    (19, GGMLType::IQ1_S, "IQ1_S", 256, 50),
    (20, GGMLType::IQ4_NL, "IQ4_NL", 32, 18),
    (21, GGMLType::IQ3_S, "IQ3_S", 256, 110),
    (22, GGMLType::IQ2_S, "IQ2_S", 256, 82),
    (23, GGMLType::IQ4_XS, "IQ4_XS", 256, 136),
    (24, GGMLType::I8, "I8", 1, 1),
    (25, GGMLType::I16, "I16", 1, 2),
    (26, GGMLType::I32, "I32", 1, 4),
    (27, GGMLType::I64, "I64", 1, 8),
    (28, GGMLType::F64, "F64", 1, 8),
    (29, GGMLType::IQ1_M, "IQ1_M", 256, 56),
    (30, GGMLType::BF16, "BF16", 1, 2),
    (34, GGMLType::TQ1_0, "TQ1_0", 256, 54),
    (35, GGMLType::TQ2_0, "TQ2_0", 256, 66),
    (39, GGMLType::MXFP4, "MXFP4", 32, 17),
    (40, GGMLType::NVFP4, "NVFP4", 64, 36),
    (41, GGMLType::Q1_0, "Q1_0", 128, 18),
];

impl From<u32> for GGMLType {
    fn from(id: u32) -> Self {
        TYPE_TABLE
            .iter()
            .find(|(table_id, ..)| *table_id == id)
            .map(|(_, ty, ..)| *ty)
            .unwrap_or(GGMLType::Unknown(id))
    }
}

impl GGMLType {
    fn table_row(self) -> Option<&'static (u32, GGMLType, &'static str, u64, u64)> {
        TYPE_TABLE.iter().find(|(_, ty, ..)| *ty == self)
    }

    /// Номер типа в ggml.
    pub fn id(self) -> u32 {
        match self {
            GGMLType::Unknown(id) => id,
            known => known.table_row().map(|row| row.0).unwrap_or(u32::MAX),
        }
    }

    /// Имя типа, как его пишут в названиях квантований (`Q4_K`, `TQ2_0`).
    pub fn name(self) -> &'static str {
        self.table_row().map(|row| row.2).unwrap_or("unknown")
    }

    /// Устройство блока или `None` для неизвестного типа.
    pub fn block_layout(self) -> Option<BlockLayout> {
        self.table_row().map(|row| BlockLayout {
            elements_per_block: row.3,
            bytes_per_block: row.4,
        })
    }

    /// Размер данных в байтах для `elements` весов. `None`, если тип неизвестен,
    /// число весов не кратно размеру блока или размер не помещается в u64.
    pub fn data_size(self, elements: u64) -> Option<u64> {
        let layout = self.block_layout()?;
        if !elements.is_multiple_of(layout.elements_per_block) {
            return None;
        }
        (elements / layout.elements_per_block).checked_mul(layout.bytes_per_block)
    }
}

impl std::fmt::Display for GGMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GGMLType::Unknown(id) => write!(f, "unknown({id})"),
            known => f.write_str(known.name()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GGUFValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GGUFValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Неотрицательное целое любого размера.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            GGUFValue::Uint8(v) => Some(u64::from(v)),
            GGUFValue::Uint16(v) => Some(u64::from(v)),
            GGUFValue::Uint32(v) => Some(u64::from(v)),
            GGUFValue::Uint64(v) => Some(v),
            GGUFValue::Int8(v) => u64::try_from(v).ok(),
            GGUFValue::Int16(v) => u64::try_from(v).ok(),
            GGUFValue::Int32(v) => u64::try_from(v).ok(),
            GGUFValue::Int64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            GGUFValue::Float32(v) => Some(v),
            GGUFValue::Float64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            GGUFValue::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GGUFValue]> {
        match self {
            GGUFValue::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GGMLType,
    /// Смещение данных от начала секции данных.
    pub offset: u64,
    /// Число весов (произведение размерностей).
    pub n_elements: u64,
    /// Размер данных в байтах; `None` для неизвестного типа.
    pub size_bytes: Option<u64>,
}

impl GGUFTensorInfo {
    pub fn total_elements(&self) -> u64 {
        self.n_elements
    }
}

/// Откуда берутся байты файла.
enum Backing {
    Mmap(Mmap),
    Bytes(Vec<u8>),
}

impl Backing {
    fn bytes(&self) -> &[u8] {
        match self {
            Backing::Mmap(mmap) => mmap,
            Backing::Bytes(bytes) => bytes,
        }
    }
}

pub struct GGUFFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: HashMap<String, GGUFValue>,
    pub tensors: Vec<GGUFTensorInfo>,
    pub tensor_map: HashMap<String, usize>,
    /// Начало секции данных от начала файла.
    pub data_offset: u64,
    /// Выравнивание секции данных.
    pub alignment: u64,
    backing: Backing,
}

/// Результат разбора заголовка — до того, как файл перемещается внутрь `GGUFFile`.
struct Parsed {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    metadata: HashMap<String, GGUFValue>,
    tensors: Vec<GGUFTensorInfo>,
    tensor_map: HashMap<String, usize>,
    data_offset: u64,
    alignment: u64,
}

impl GGUFFile {
    /// Открывает файл через отображение в память (mmap): веса не копируются в RAM целиком.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GGUFError> {
        let file = File::open(path)?;
        // SAFETY: отображение только для чтения. Если другой процесс изменит файл, пока он
        // открыт, прочитанные байты могут измениться — это ограничение mmap, а не памяти Rust:
        // все обращения к данным идут через проверенные границы среза.
        let mmap = unsafe { Mmap::map(&file)? };
        let parsed = parse(&mmap)?;
        Ok(Self::from_parsed(parsed, Backing::Mmap(mmap)))
    }

    /// Разбирает файл, целиком загруженный в память (удобно для тестов).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, GGUFError> {
        let parsed = parse(&bytes)?;
        Ok(Self::from_parsed(parsed, Backing::Bytes(bytes)))
    }

    fn from_parsed(parsed: Parsed, backing: Backing) -> Self {
        Self {
            version: parsed.version,
            tensor_count: parsed.tensor_count,
            metadata_kv_count: parsed.metadata_kv_count,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            tensor_map: parsed.tensor_map,
            data_offset: parsed.data_offset,
            alignment: parsed.alignment,
            backing,
        }
    }

    /// Размер файла в байтах.
    pub fn file_size(&self) -> u64 {
        self.backing.bytes().len() as u64
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    /// Значение метаданных архитектуры: `<arch>.<suffix>`, например `llama.block_count`.
    pub fn arch_value(&self, suffix: &str) -> Option<&GGUFValue> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{arch}.{suffix}"))
    }

    /// Целое значение метаданных архитектуры (массивы по слоям дают `None`).
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        self.arch_value(suffix).and_then(GGUFValue::as_u64)
    }

    /// Длина контекста, на которую обучена модель.
    pub fn context_length(&self) -> Option<u64> {
        self.arch_u64("context_length")
    }

    pub fn embedding_length(&self) -> Option<u64> {
        self.arch_u64("embedding_length")
    }

    pub fn block_count(&self) -> Option<u64> {
        self.arch_u64("block_count")
    }

    pub fn head_count(&self) -> Option<u64> {
        self.arch_u64("attention.head_count")
    }

    pub fn head_count_kv(&self) -> Option<u64> {
        self.arch_u64("attention.head_count_kv")
    }

    /// Общее число весов модели (сумма по всем тензорам).
    pub fn parameter_count(&self) -> u64 {
        self.tensors
            .iter()
            .fold(0u64, |sum, tensor| sum.saturating_add(tensor.n_elements))
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<&GGUFTensorInfo> {
        let idx = self.tensor_map.get(name)?;
        self.tensors.get(*idx)
    }

    /// Байты тензора. Границы проверены при разборе; `None` — тензора нет или тип неизвестен.
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.get_tensor_info(name)?;
        let start = usize::try_from(self.data_offset.checked_add(info.offset)?).ok()?;
        let len = usize::try_from(info.size_bytes?).ok()?;
        self.backing.bytes().get(start..start.checked_add(len)?)
    }
}

// ---------- низкоуровневое чтение с проверкой границ ----------

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, len: usize, what: &'static str) -> Result<&'a [u8], GGUFError> {
        if len > self.remaining() {
            return Err(GGUFError::Truncated(what));
        }
        let bytes = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(bytes)
    }

    fn array<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N], GGUFError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N, what)?);
        Ok(out)
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, GGUFError> {
        Ok(u32::from_le_bytes(self.array(what)?))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, GGUFError> {
        Ok(u64::from_le_bytes(self.array(what)?))
    }

    /// Число записей, для которых в файле хватает хотя бы `min_item_bytes` на каждую.
    fn count(&mut self, min_item_bytes: usize, what: &'static str) -> Result<usize, GGUFError> {
        let count = self.u64(what)?;
        usize::try_from(count)
            .ok()
            .filter(|n| {
                n.checked_mul(min_item_bytes)
                    .is_some_and(|bytes| bytes <= self.remaining())
            })
            .ok_or(GGUFError::CountTooLarge { what, count })
    }

    fn string(&mut self, what: &'static str) -> Result<String, GGUFError> {
        let len = self.u64(what)?;
        let len = usize::try_from(len).map_err(|_| GGUFError::Truncated(what))?;
        Ok(String::from_utf8(self.take(len, what)?.to_vec())?)
    }
}

/// Минимальный размер значения метаданных данного типа (для проверки длины массивов).
fn min_value_bytes(value_type: u32) -> Option<usize> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        8 | 10..=12 => Some(8),
        9 => Some(12),
        _ => None,
    }
}

fn read_value(reader: &mut Reader, value_type: u32, depth: usize) -> Result<GGUFValue, GGUFError> {
    const WHAT: &str = "значение метаданных";
    Ok(match value_type {
        0 => GGUFValue::Uint8(u8::from_le_bytes(reader.array(WHAT)?)),
        1 => GGUFValue::Int8(i8::from_le_bytes(reader.array(WHAT)?)),
        2 => GGUFValue::Uint16(u16::from_le_bytes(reader.array(WHAT)?)),
        3 => GGUFValue::Int16(i16::from_le_bytes(reader.array(WHAT)?)),
        4 => GGUFValue::Uint32(reader.u32(WHAT)?),
        5 => GGUFValue::Int32(i32::from_le_bytes(reader.array(WHAT)?)),
        6 => GGUFValue::Float32(f32::from_le_bytes(reader.array(WHAT)?)),
        7 => GGUFValue::Bool(u8::from_le_bytes(reader.array(WHAT)?) != 0),
        8 => GGUFValue::String(reader.string(WHAT)?),
        9 => {
            if depth >= MAX_VALUE_NESTING {
                return Err(GGUFError::NestingTooDeep);
            }
            let element_type = reader.u32("тип элементов массива")?;
            let min_bytes =
                min_value_bytes(element_type).ok_or(GGUFError::InvalidValueType(element_type))?;
            let len = reader.count(min_bytes, "массив метаданных")?;
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(read_value(reader, element_type, depth + 1)?);
            }
            GGUFValue::Array(items)
        }
        10 => GGUFValue::Uint64(reader.u64(WHAT)?),
        11 => GGUFValue::Int64(i64::from_le_bytes(reader.array(WHAT)?)),
        12 => GGUFValue::Float64(f64::from_le_bytes(reader.array(WHAT)?)),
        other => return Err(GGUFError::InvalidValueType(other)),
    })
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
}

fn parse(data: &[u8]) -> Result<Parsed, GGUFError> {
    let mut reader = Reader { data, pos: 0 };

    let magic = reader.u32("магическое число")?;
    if magic != GGUF_MAGIC {
        return Err(GGUFError::InvalidMagic(magic));
    }
    let version = reader.u32("версия")?;
    if !(2..=3).contains(&version) {
        return Err(GGUFError::UnsupportedVersion(version));
    }

    let raw_tensor_count = reader.u64("число тензоров")?;
    let raw_kv_count = reader.u64("число ключей метаданных")?;

    // Проверка «хватит ли байтов хотя бы на минимальные записи» — до выделения памяти
    let kv_count = usize::try_from(raw_kv_count)
        .ok()
        .filter(|n| {
            n.checked_mul(MIN_KV_BYTES)
                .is_some_and(|b| b <= reader.remaining())
        })
        .ok_or(GGUFError::CountTooLarge {
            what: "метаданных",
            count: raw_kv_count,
        })?;
    let mut metadata = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = reader.string("ключ метаданных")?;
        let value_type = reader.u32("тип значения")?;
        let value = read_value(&mut reader, value_type, 0)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(GGUFError::DuplicateKey(key));
        }
    }

    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(value) => value
            .as_u64()
            .filter(|a| a.is_power_of_two())
            .ok_or_else(|| GGUFError::InvalidAlignment(value.as_u64().unwrap_or(0)))?,
    };

    let tensor_count = usize::try_from(raw_tensor_count)
        .ok()
        .filter(|n| {
            n.checked_mul(MIN_TENSOR_INFO_BYTES)
                .is_some_and(|b| b <= reader.remaining())
        })
        .ok_or(GGUFError::CountTooLarge {
            what: "тензоров",
            count: raw_tensor_count,
        })?;
    let mut tensors = Vec::with_capacity(tensor_count);
    let mut tensor_map = HashMap::with_capacity(tensor_count);
    for index in 0..tensor_count {
        let name = reader.string("имя тензора")?;
        let dims = reader.u32("число измерений")?;
        if dims as usize > MAX_DIMENSIONS {
            return Err(GGUFError::TooManyDimensions { name, dims });
        }
        let mut dimensions = Vec::with_capacity(dims as usize);
        for _ in 0..dims {
            dimensions.push(reader.u64("размерность тензора")?);
        }
        let tensor_type = GGMLType::from(reader.u32("тип тензора")?);
        let offset = reader.u64("смещение тензора")?;

        let n_elements = dimensions
            .iter()
            .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| GGUFError::ElementCountOverflow(name.clone()))?;
        let size_bytes =
            match tensor_type.block_layout() {
                None => None,
                Some(_) => Some(tensor_type.data_size(n_elements).ok_or_else(|| {
                    GGUFError::PartialBlock {
                        name: name.clone(),
                        elements: n_elements,
                        tensor_type: tensor_type.name(),
                    }
                })?),
            };

        if tensor_map.insert(name.clone(), index).is_some() {
            return Err(GGUFError::DuplicateTensor(name));
        }
        tensors.push(GGUFTensorInfo {
            name,
            dimensions,
            tensor_type,
            offset,
            n_elements,
            size_bytes,
        });
    }

    let data_offset = align_up(reader.pos as u64, alignment)
        .ok_or(GGUFError::Truncated("начало секции данных"))?;
    let file_len = data.len() as u64;
    for tensor in &tensors {
        // Для неизвестного типа размер не проверить, но начало данных всё равно должно быть в файле
        let end = data_offset
            .checked_add(tensor.offset)
            .and_then(|start| start.checked_add(tensor.size_bytes.unwrap_or(0)));
        if end.is_none_or(|end| end > file_len) {
            return Err(GGUFError::TensorOutOfBounds(tensor.name.clone()));
        }
    }

    Ok(Parsed {
        version,
        tensor_count: raw_tensor_count,
        metadata_kv_count: raw_kv_count,
        metadata,
        tensors,
        tensor_map,
        data_offset,
        alignment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    fn write_string(buf: &mut Vec<u8>, text: &str) {
        buf.write_u64::<LittleEndian>(text.len() as u64).unwrap();
        buf.write_all(text.as_bytes()).unwrap();
    }

    /// Минимальный корректный файл: 3 ключа и тензор F32 4×4.
    pub fn create_mock_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap();
        buf.write_u64::<LittleEndian>(3).unwrap();

        write_string(&mut buf, "general.architecture");
        buf.write_u32::<LittleEndian>(8).unwrap();
        write_string(&mut buf, "llama");

        write_string(&mut buf, "llama.context_length");
        buf.write_u32::<LittleEndian>(4).unwrap();
        buf.write_u32::<LittleEndian>(4096).unwrap();

        write_string(&mut buf, "general.alignment");
        buf.write_u32::<LittleEndian>(4).unwrap();
        buf.write_u32::<LittleEndian>(32).unwrap();

        write_string(&mut buf, "token_embd.weight");
        buf.write_u32::<LittleEndian>(2).unwrap();
        buf.write_u64::<LittleEndian>(4).unwrap();
        buf.write_u64::<LittleEndian>(4).unwrap();
        buf.write_u32::<LittleEndian>(0).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap();

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for i in 0..16 {
            buf.write_f32::<LittleEndian>(i as f32).unwrap();
        }
        buf
    }

    #[test]
    fn test_parse_mock_gguf() {
        let gguf = GGUFFile::from_bytes(create_mock_gguf()).expect("файл должен разобраться");

        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensor_count, 1);
        assert_eq!(gguf.architecture(), Some("llama"));
        assert_eq!(gguf.context_length(), Some(4096));
        assert_eq!(
            gguf.block_count(),
            None,
            "отсутствующее значение не выдумывается"
        );
        assert_eq!(gguf.parameter_count(), 16);

        let t = gguf
            .get_tensor_info("token_embd.weight")
            .expect("тензор найден");
        assert_eq!(t.dimensions, vec![4, 4]);
        assert_eq!(t.tensor_type, GGMLType::F32);
        assert_eq!(t.size_bytes, Some(64));

        let data = gguf
            .get_tensor_data("token_embd.weight")
            .expect("данные тензора");
        assert_eq!(data.len(), 64);
        assert_eq!(f32::from_le_bytes(data[4..8].try_into().unwrap()), 1.0);
    }

    #[test]
    fn type_ids_and_block_sizes_match_ggml() {
        // Значения из эталонной libggml-base (ggml_type_name / ggml_blck_size / ggml_type_size)
        assert_eq!(GGMLType::from(29), GGMLType::IQ1_M);
        assert_eq!(GGMLType::from(30), GGMLType::BF16);
        assert_eq!(GGMLType::from(35), GGMLType::TQ2_0);
        assert_eq!(GGMLType::from(4), GGMLType::Unknown(4));
        assert_eq!(GGMLType::TQ2_0.id(), 35);
        assert_eq!(GGMLType::Q6_K.data_size(256 * 3), Some(210 * 3));
        assert_eq!(GGMLType::Q4_0.data_size(32 * 10), Some(180));
        assert_eq!(GGMLType::IQ4_XS.data_size(256), Some(136));
        assert_eq!(
            GGMLType::Q4_K.data_size(100),
            None,
            "неполный блок недопустим"
        );
        assert_eq!(GGMLType::Unknown(99).data_size(256), None);
        for (id, ty, name, ..) in TYPE_TABLE {
            assert_eq!(GGMLType::from(*id), *ty);
            assert_eq!(ty.id(), *id);
            assert_eq!(ty.name(), *name);
        }
    }

    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        let full = create_mock_gguf();
        for len in 0..full.len() {
            assert!(
                GGUFFile::from_bytes(full[..len].to_vec()).is_err(),
                "обрезанный до {len} байт файл должен давать ошибку"
            );
        }
    }

    /// Заголовок с заданными числами тензоров и ключей, без самих записей.
    fn header(tensors: u64, kvs: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(tensors).unwrap();
        buf.write_u64::<LittleEndian>(kvs).unwrap();
        buf
    }

    #[test]
    fn rejects_huge_counts_and_lengths_without_allocating() {
        assert!(matches!(
            GGUFFile::from_bytes(header(0, u64::MAX)),
            Err(GGUFError::CountTooLarge { .. })
        ));
        assert!(matches!(
            GGUFFile::from_bytes(header(u64::MAX, 0)),
            Err(GGUFError::CountTooLarge { .. })
        ));

        let mut huge_string = header(0, 1);
        huge_string.write_u64::<LittleEndian>(u64::MAX).unwrap();
        // Байтов хватает на одну запись метаданных, поэтому ошибку даёт именно длина строки
        huge_string.extend_from_slice(&[0u8; 32]);
        assert!(matches!(
            GGUFFile::from_bytes(huge_string),
            Err(GGUFError::Truncated(_))
        ));

        let mut huge_array = header(0, 1);
        write_string(&mut huge_array, "tokenizer.ggml.tokens");
        huge_array.write_u32::<LittleEndian>(9).unwrap();
        huge_array.write_u32::<LittleEndian>(8).unwrap();
        huge_array.write_u64::<LittleEndian>(1 << 40).unwrap();
        assert!(matches!(
            GGUFFile::from_bytes(huge_array),
            Err(GGUFError::CountTooLarge { .. })
        ));
    }

    #[test]
    fn rejects_invalid_alignment_and_structure() {
        for alignment in [0u32, 3, 48] {
            let mut buf = header(0, 1);
            write_string(&mut buf, "general.alignment");
            buf.write_u32::<LittleEndian>(4).unwrap();
            buf.write_u32::<LittleEndian>(alignment).unwrap();
            assert!(
                matches!(
                    GGUFFile::from_bytes(buf),
                    Err(GGUFError::InvalidAlignment(_))
                ),
                "выравнивание {alignment} должно отклоняться"
            );
        }

        let mut duplicate = header(0, 2);
        for _ in 0..2 {
            write_string(&mut duplicate, "general.name");
            duplicate.write_u32::<LittleEndian>(8).unwrap();
            write_string(&mut duplicate, "x");
        }
        assert!(matches!(
            GGUFFile::from_bytes(duplicate),
            Err(GGUFError::DuplicateKey(_))
        ));

        let mut too_many_dims = header(1, 0);
        write_string(&mut too_many_dims, "t");
        too_many_dims.write_u32::<LittleEndian>(5).unwrap();
        too_many_dims.extend_from_slice(&[0u8; 64]);
        assert!(matches!(
            GGUFFile::from_bytes(too_many_dims),
            Err(GGUFError::TooManyDimensions { .. })
        ));
    }

    #[test]
    fn rejects_tensor_data_outside_file() {
        let mut buf = header(1, 0);
        write_string(&mut buf, "big.weight");
        buf.write_u32::<LittleEndian>(1).unwrap();
        buf.write_u64::<LittleEndian>(1_000_000).unwrap();
        buf.write_u32::<LittleEndian>(0).unwrap(); // F32 → 4 МБ данных, которых нет
        buf.write_u64::<LittleEndian>(0).unwrap();
        assert!(matches!(
            GGUFFile::from_bytes(buf),
            Err(GGUFError::TensorOutOfBounds(_))
        ));
    }

    #[test]
    fn rejects_deeply_nested_arrays() {
        let mut buf = header(0, 1);
        write_string(&mut buf, "nested");
        buf.write_u32::<LittleEndian>(9).unwrap();
        for _ in 0..(MAX_VALUE_NESTING + 2) {
            buf.write_u32::<LittleEndian>(9).unwrap(); // элементы — снова массивы
            buf.write_u64::<LittleEndian>(1).unwrap();
        }
        buf.extend_from_slice(&[0u8; 64]);
        assert!(GGUFFile::from_bytes(buf).is_err());
    }

    /// Проверяет реальный файл из папки models/ (в CI его нет — тогда тест пропускается).
    fn check_real_model(
        file_name: &str,
        expected_tensors: u64,
        expected_arch: &str,
    ) -> Option<GGUFFile> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models")
            .join(file_name);
        if !path.is_file() {
            eprintln!("пропуск: {} не найден", path.display());
            return None;
        }
        let gguf = GGUFFile::open(&path).expect("реальная модель должна разобраться");
        assert_eq!(gguf.tensor_count, expected_tensors);
        assert_eq!(gguf.architecture(), Some(expected_arch));

        // Тензоры идут подряд: каждый следующий начинается сразу после выровненного конца предыдущего
        let mut by_offset: Vec<&GGUFTensorInfo> = gguf.tensors.iter().collect();
        by_offset.sort_by_key(|t| t.offset);
        let mut expected_offset = 0u64;
        for tensor in &by_offset {
            assert_eq!(
                tensor.offset, expected_offset,
                "дыра или перекрытие перед {}",
                tensor.name
            );
            let size = tensor
                .size_bytes
                .expect("все типы в реальных моделях известны");
            assert_eq!(
                gguf.get_tensor_data(&tensor.name).map(<[u8]>::len),
                Some(size as usize)
            );
            expected_offset = align_up(tensor.offset + size, gguf.alignment).unwrap();
        }
        let data_len = gguf.file_size() - gguf.data_offset;
        assert!(data_len >= expected_offset - gguf.alignment && data_len <= expected_offset);
        Some(gguf)
    }

    #[test]
    fn real_llama_3_2_1b_q4_k_m() {
        let Some(gguf) = check_real_model("Llama-3.2-1B-Instruct-Q4_K_M.gguf", 147, "llama") else {
            return;
        };
        assert_eq!(gguf.block_count(), Some(16));
        assert_eq!(gguf.head_count_kv(), Some(8));
        assert_eq!(gguf.context_length(), Some(131_072));
        let count = |ty| gguf.tensors.iter().filter(|t| t.tensor_type == ty).count();
        assert_eq!(count(GGMLType::Q4_K), 96);
        assert_eq!(count(GGMLType::Q6_K), 17);
        assert_eq!(count(GGMLType::F32), 34);
        // Около 1,24 млрд весов: число из метаданных, а не угаданное по размеру файла
        assert!((1_200_000_000..1_300_000_000).contains(&gguf.parameter_count()));
    }

    #[test]
    fn real_gemma_4_e2b_with_ternary_tq2_0() {
        let Some(gguf) = check_real_model("gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf", 541, "gemma4")
        else {
            return;
        };
        let count = |ty| gguf.tensors.iter().filter(|t| t.tensor_type == ty).count();
        assert_eq!(count(GGMLType::TQ2_0), 61);
        assert_eq!(count(GGMLType::Q4_0), 146);
        assert_eq!(count(GGMLType::Q8_0), 70);
        assert_eq!(gguf.block_count(), Some(35));
    }
}
