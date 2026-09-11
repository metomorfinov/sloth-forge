use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek};
use std::path::Path;
use thiserror::Error;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little endian

#[derive(Error, Debug)]
pub enum GGUFError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GGUF magic: expected 0x{GGUF_MAGIC:08x}, got 0x{0:08x}")]
    InvalidMagic(u32),
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("Invalid metadata value type: {0}")]
    InvalidValueType(u32),
    #[error("Invalid UTF-8 string: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),
    #[error("Unexpected EOF while parsing GGUF")]
    UnexpectedEof,
    #[error("Tensor '{0}' not found")]
    TensorNotFound(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 29,
    Unknown(u32),
}

impl From<u32> for GGMLType {
    fn from(val: u32) -> Self {
        match val {
            0 => GGMLType::F32,
            1 => GGMLType::F16,
            2 => GGMLType::Q4_0,
            3 => GGMLType::Q4_1,
            6 => GGMLType::Q5_0,
            7 => GGMLType::Q5_1,
            8 => GGMLType::Q8_0,
            9 => GGMLType::Q8_1,
            10 => GGMLType::Q2_K,
            11 => GGMLType::Q3_K,
            12 => GGMLType::Q4_K,
            13 => GGMLType::Q5_K,
            14 => GGMLType::Q6_K,
            15 => GGMLType::Q8_K,
            24 => GGMLType::I8,
            25 => GGMLType::I16,
            26 => GGMLType::I32,
            27 => GGMLType::I64,
            28 => GGMLType::F64,
            29 => GGMLType::BF16,
            other => GGMLType::Unknown(other),
        }
    }
}

impl GGMLType {
    pub fn name(&self) -> &'static str {
        match self {
            GGMLType::F32 => "F32",
            GGMLType::F16 => "F16",
            GGMLType::BF16 => "BF16",
            GGMLType::Q4_0 => "Q4_0",
            GGMLType::Q4_1 => "Q4_1",
            GGMLType::Q5_0 => "Q5_0",
            GGMLType::Q5_1 => "Q5_1",
            GGMLType::Q8_0 => "Q8_0",
            GGMLType::Q4_K => "Q4_K",
            GGMLType::Q5_K => "Q5_K",
            GGMLType::Q6_K => "Q6_K",
            GGMLType::Q8_K => "Q8_K",
            _ => "Other",
        }
    }

    pub fn type_size_in_bytes(&self, num_elements: usize) -> usize {
        match self {
            GGMLType::F32 => num_elements * 4,
            GGMLType::F16 | GGMLType::BF16 => num_elements * 2,
            GGMLType::Q8_0 => {
                // block size 32: 2 bytes f16 scale + 32 bytes i8 = 34 bytes per 32 elements
                let blocks = (num_elements + 31) / 32;
                blocks * 34
            }
            GGMLType::Q4_K => {
                // block size 256: 144 bytes per 256 elements
                let blocks = (num_elements + 255) / 256;
                blocks * 144
            }
            _ => num_elements * 2,
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

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GGUFValue::Uint8(v) => Some(*v as u64),
            GGUFValue::Uint16(v) => Some(*v as u64),
            GGUFValue::Uint32(v) => Some(*v as u64),
            GGUFValue::Uint64(v) => Some(*v),
            GGUFValue::Int32(v) if *v >= 0 => Some(*v as u64),
            GGUFValue::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GGUFValue::Float32(v) => Some(*v),
            GGUFValue::Float64(v) => Some(*v as f32),
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
    pub offset: u64,
}

impl GGUFTensorInfo {
    pub fn total_elements(&self) -> usize {
        self.dimensions.iter().product::<u64>() as usize
    }
}

pub struct GGUFFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: HashMap<String, GGUFValue>,
    pub tensors: Vec<GGUFTensorInfo>,
    pub tensor_map: HashMap<String, usize>,
    pub data_offset: usize,
    mmap: Option<Mmap>,
    raw_buffer: Option<Vec<u8>>,
}

impl GGUFFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GGUFError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, GGUFError> {
        let mut cursor = Cursor::new(&bytes);
        let mut parsed = Self::parse_headers_and_metadata(&mut cursor)?;
        let data_offset = cursor.position() as usize;
        let alignment = parsed
            .metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let aligned_data_offset = (data_offset + alignment - 1) & !(alignment - 1);

        parsed.data_offset = aligned_data_offset;
        parsed.raw_buffer = Some(bytes);
        Ok(parsed)
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self, GGUFError> {
        let mut cursor = Cursor::new(&mmap[..]);
        let mut parsed = Self::parse_headers_and_metadata(&mut cursor)?;
        let data_offset = cursor.position() as usize;
        let alignment = parsed
            .metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let aligned_data_offset = (data_offset + alignment - 1) & !(alignment - 1);

        parsed.data_offset = aligned_data_offset;
        parsed.mmap = Some(mmap);
        Ok(parsed)
    }

    fn parse_headers_and_metadata<R: Read + Seek>(reader: &mut R) -> Result<Self, GGUFError> {
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(GGUFError::InvalidMagic(magic));
        }

        let version = reader.read_u32::<LittleEndian>()?;
        if !(2..=3).contains(&version) {
            return Err(GGUFError::UnsupportedVersion(version));
        }

        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let metadata_kv_count = reader.read_u64::<LittleEndian>()?;

        let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(reader)?;
            let val_type = reader.read_u32::<LittleEndian>()?;
            let val = read_gguf_value(reader, val_type)?;
            metadata.insert(key, val);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut tensor_map = HashMap::with_capacity(tensor_count as usize);

        for i in 0..tensor_count {
            let name = read_gguf_string(reader)?;
            let n_dims = reader.read_u32::<LittleEndian>()?;
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(reader.read_u64::<LittleEndian>()?);
            }
            let raw_type = reader.read_u32::<LittleEndian>()?;
            let tensor_type = GGMLType::from(raw_type);
            let offset = reader.read_u64::<LittleEndian>()?;

            tensor_map.insert(name.clone(), i as usize);
            tensors.push(GGUFTensorInfo {
                name,
                dimensions,
                tensor_type,
                offset,
            });
        }

        Ok(Self {
            version,
            tensor_count,
            metadata_kv_count,
            metadata,
            tensors,
            tensor_map,
            data_offset: 0,
            mmap: None,
            raw_buffer: None,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| v.as_str())
    }

    pub fn context_length(&self) -> u64 {
        let arch = self.architecture().unwrap_or("llama");
        let key = format!("{}.context_length", arch);
        self.metadata
            .get(&key)
            .or_else(|| self.metadata.get("general.context_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2048)
    }

    pub fn embedding_length(&self) -> u64 {
        let arch = self.architecture().unwrap_or("llama");
        let key = format!("{}.embedding_length", arch);
        self.metadata
            .get(&key)
            .and_then(|v| v.as_u64())
            .unwrap_or(4096)
    }

    pub fn block_count(&self) -> u64 {
        let arch = self.architecture().unwrap_or("llama");
        let key = format!("{}.block_count", arch);
        self.metadata
            .get(&key)
            .and_then(|v| v.as_u64())
            .unwrap_or(32)
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<&GGUFTensorInfo> {
        let idx = self.tensor_map.get(name)?;
        self.tensors.get(*idx)
    }

    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.get_tensor_info(name)?;
        let start = self.data_offset + info.offset as usize;
        let size = info.tensor_type.type_size_in_bytes(info.total_elements());

        if let Some(mmap) = &self.mmap {
            if start + size <= mmap.len() {
                return Some(&mmap[start..start + size]);
            }
        } else if let Some(buf) = &self.raw_buffer {
            if start + size <= buf.len() {
                return Some(&buf[start..start + size]);
            }
        }
        None
    }
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String, GGUFError> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_gguf_value<R: Read + Seek>(reader: &mut R, val_type: u32) -> Result<GGUFValue, GGUFError> {
    match val_type {
        0 => Ok(GGUFValue::Uint8(reader.read_u8()?)),
        1 => Ok(GGUFValue::Int8(reader.read_i8()?)),
        2 => Ok(GGUFValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        3 => Ok(GGUFValue::Int16(reader.read_i16::<LittleEndian>()?)),
        4 => Ok(GGUFValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        5 => Ok(GGUFValue::Int32(reader.read_i32::<LittleEndian>()?)),
        6 => Ok(GGUFValue::Float32(reader.read_f32::<LittleEndian>()?)),
        7 => Ok(GGUFValue::Bool(reader.read_u8()? != 0)),
        8 => Ok(GGUFValue::String(read_gguf_string(reader)?)),
        9 => {
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let len = reader.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_gguf_value(reader, elem_type)?);
            }
            Ok(GGUFValue::Array(arr))
        }
        10 => Ok(GGUFValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(GGUFValue::Int64(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(GGUFValue::Float64(reader.read_f64::<LittleEndian>()?)),
        other => Err(GGUFError::InvalidValueType(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Write;

    pub fn create_mock_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap();
        buf.write_u64::<LittleEndian>(3).unwrap();

        let k1 = "general.architecture";
        buf.write_u64::<LittleEndian>(k1.len() as u64).unwrap();
        buf.write_all(k1.as_bytes()).unwrap();
        buf.write_u32::<LittleEndian>(8).unwrap();
        let v1 = "llama";
        buf.write_u64::<LittleEndian>(v1.len() as u64).unwrap();
        buf.write_all(v1.as_bytes()).unwrap();

        let k2 = "llama.context_length";
        buf.write_u64::<LittleEndian>(k2.len() as u64).unwrap();
        buf.write_all(k2.as_bytes()).unwrap();
        buf.write_u32::<LittleEndian>(4).unwrap();
        buf.write_u32::<LittleEndian>(4096).unwrap();

        let k3 = "general.alignment";
        buf.write_u64::<LittleEndian>(k3.len() as u64).unwrap();
        buf.write_all(k3.as_bytes()).unwrap();
        buf.write_u32::<LittleEndian>(4).unwrap();
        buf.write_u32::<LittleEndian>(32).unwrap();

        let t1_name = "token_embd.weight";
        buf.write_u64::<LittleEndian>(t1_name.len() as u64).unwrap();
        buf.write_all(t1_name.as_bytes()).unwrap();
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
        let bytes = create_mock_gguf();
        let gguf = GGUFFile::from_bytes(bytes).expect("Should parse GGUF successfully");

        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensor_count, 1);
        assert_eq!(gguf.architecture(), Some("llama"));
        assert_eq!(gguf.context_length(), 4096);

        let t = gguf.get_tensor_info("token_embd.weight").expect("Tensor found");
        assert_eq!(t.dimensions, vec![4, 4]);
        assert_eq!(t.tensor_type, GGMLType::F32);

        let data = gguf.get_tensor_data("token_embd.weight").expect("Tensor data found");
        assert_eq!(data.len(), 64);
    }
}
