use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

pub const PAD_TOKEN_ID: u32 = 0;
pub const BOS_TOKEN_ID: u32 = 1;
pub const EOS_TOKEN_ID: u32 = 2;
pub const IM_START_TOKEN_ID: u32 = 100001;
pub const IM_END_TOKEN_ID: u32 = 100002;
pub const IGNORE_LABEL_ID: i64 = -100;

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Unknown or unsupported dataset format")]
    UnknownFormat,
    #[error("Empty dataset provided")]
    EmptyDataset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetFormat {
    Alpaca,
    ChatML,
    RawJsonl,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlpacaEntry {
    pub instruction: String,
    #[serde(default)]
    pub input: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEntry {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub completion: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SampleItem {
    pub prompt: String,
    pub completion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedSequence {
    pub input_ids: Vec<u32>,
    pub labels: Vec<i64>,
    pub position_ids: Vec<u32>,
    pub segment_ids: Vec<u32>,
    pub seq_lengths: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Special tokens
        let specials = [
            ("<unk>", PAD_TOKEN_ID),
            ("<s>", BOS_TOKEN_ID),
            ("</s>", EOS_TOKEN_ID),
            ("<|im_start|>", IM_START_TOKEN_ID),
            ("<|im_end|>", IM_END_TOKEN_ID),
        ];

        for &(s, id) in &specials {
            vocab.insert(s.to_string(), id);
            id_to_token.insert(id, s.to_string());
        }

        // Byte-level tokens (3..259)
        for b in 0u8..=255u8 {
            let s = format!("<0x{:02X}>", b);
            let id = 3 + b as u32;
            vocab.insert(s.clone(), id);
            id_to_token.insert(id, s);
        }

        Self { vocab, id_to_token }
    }

    pub fn from_tokens(tokens: &[String]) -> Self {
        let mut tokenizer = Self::new();
        for (i, t) in tokens.iter().enumerate() {
            let id = (i + 1000) as u32;
            tokenizer.vocab.insert(t.clone(), id);
            tokenizer.id_to_token.insert(id, t.clone());
        }
        tokenizer
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Check for special tags first
            let mut matched_special = false;
            for (special, &id) in &self.vocab {
                if id >= IM_START_TOKEN_ID && remaining.starts_with(special) {
                    tokens.push(id);
                    remaining = &remaining[special.len()..];
                    matched_special = true;
                    break;
                }
            }

            if matched_special {
                continue;
            }

            // Byte-level fallback
            let b = remaining.as_bytes()[0];
            tokens.push(3 + b as u32);
            remaining = &remaining[1..];
        }

        tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            if (3..=258).contains(&t) {
                bytes.push((t - 3) as u8);
            } else if let Some(tok) = self.id_to_token.get(&t) {
                bytes.extend_from_slice(tok.as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

pub struct DatasetLoader;

impl DatasetLoader {
    pub fn parse_samples(content: &str, format: DatasetFormat) -> Result<Vec<SampleItem>, DatasetError> {
        let mut samples = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let fmt = match format {
                DatasetFormat::Auto => {
                    if line.contains("\"messages\"") {
                        DatasetFormat::ChatML
                    } else if line.contains("\"instruction\"") {
                        DatasetFormat::Alpaca
                    } else {
                        DatasetFormat::RawJsonl
                    }
                }
                other => other,
            };

            match fmt {
                DatasetFormat::Alpaca => {
                    let entry: AlpacaEntry = serde_json::from_str(line)?;
                    let prompt = if entry.input.is_empty() {
                        format!("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n", entry.instruction)
                    } else {
                        format!("Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n", entry.instruction, entry.input)
                    };
                    samples.push(SampleItem {
                        prompt,
                        completion: entry.output,
                    });
                }
                DatasetFormat::ChatML => {
                    #[derive(Deserialize)]
                    struct ChatWrapper {
                        messages: Vec<ChatMessage>,
                    }
                    let wrapper: ChatWrapper = serde_json::from_str(line)?;
                    let mut prompt = String::new();
                    let mut completion = String::new();

                    for msg in wrapper.messages {
                        if msg.role == "assistant" {
                            completion = format!("<|im_start|>assistant\n{}<|im_end|>\n", msg.content);
                        } else {
                            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
                        }
                    }
                    if !completion.is_empty() {
                        prompt.push_str("<|im_start|>assistant\n");
                    }
                    samples.push(SampleItem { prompt, completion });
                }
                DatasetFormat::RawJsonl | DatasetFormat::Auto => {
                    let entry: RawEntry = serde_json::from_str(line)?;
                    if let (Some(p), Some(c)) = (entry.prompt, entry.completion) {
                        samples.push(SampleItem {
                            prompt: p,
                            completion: c,
                        });
                    } else if let Some(txt) = entry.text {
                        samples.push(SampleItem {
                            prompt: String::new(),
                            completion: txt,
                        });
                    }
                }
            }
        }

        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }

        Ok(samples)
    }

    pub fn load_file<P: AsRef<Path>>(
        path: P,
        format: DatasetFormat,
        tokenizer: &SimpleTokenizer,
        max_seq_len: usize,
        pack_sequences: bool,
    ) -> Result<Vec<PackedSequence>, DatasetError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut content = String::new();
        for line in reader.lines() {
            let l = line?;
            content.push_str(&l);
            content.push('\n');
        }
        Self::load_from_str(&content, format, tokenizer, max_seq_len, pack_sequences)
    }

    pub fn load_from_str(
        content: &str,
        format: DatasetFormat,
        tokenizer: &SimpleTokenizer,
        max_seq_len: usize,
        pack_sequences: bool,
    ) -> Result<Vec<PackedSequence>, DatasetError> {
        let samples = Self::parse_samples(content, format)?;
        let mut tokenized_samples = Vec::new();

        for s in samples {
            let prompt_tokens = tokenizer.encode(&s.prompt);
            let comp_tokens = tokenizer.encode(&s.completion);

            let mut input_ids = Vec::new();
            let mut labels = Vec::new();

            // Prompt: masked out with IGNORE_LABEL_ID (-100)
            for &t in &prompt_tokens {
                input_ids.push(t);
                labels.push(IGNORE_LABEL_ID);
            }

            // Completion: target label
            for &t in &comp_tokens {
                input_ids.push(t);
                labels.push(t as i64);
            }

            // Append EOS token
            input_ids.push(EOS_TOKEN_ID);
            labels.push(EOS_TOKEN_ID as i64);

            tokenized_samples.push((input_ids, labels));
        }

        if !pack_sequences {
            // Standard individual sequence padding / truncation
            let mut batches = Vec::new();
            for (mut inp, mut lbl) in tokenized_samples {
                if inp.len() > max_seq_len {
                    inp.truncate(max_seq_len);
                    lbl.truncate(max_seq_len);
                }
                let len = inp.len();
                let position_ids = (0..len as u32).collect();
                let segment_ids = vec![0u32; len];

                // Pad up to max_seq_len
                while inp.len() < max_seq_len {
                    inp.push(PAD_TOKEN_ID);
                    lbl.push(IGNORE_LABEL_ID);
                }

                batches.push(PackedSequence {
                    input_ids: inp,
                    labels: lbl,
                    position_ids,
                    segment_ids,
                    seq_lengths: vec![len],
                });
            }
            return Ok(batches);
        }

        // Sequence Packing implementation:
        // Pack multiple sequences into fixed windows of max_seq_len
        let mut packed_batches = Vec::new();
        let mut cur_inp = Vec::with_capacity(max_seq_len);
        let mut cur_lbl = Vec::with_capacity(max_seq_len);
        let mut cur_pos = Vec::with_capacity(max_seq_len);
        let mut cur_seg = Vec::with_capacity(max_seq_len);
        let mut cur_lengths = Vec::new();
        let mut current_segment = 0u32;

        for (inp, lbl) in tokenized_samples {
            let seq_len = inp.len();
            if seq_len > max_seq_len {
                // If sequence alone exceeds max_seq_len, truncate it
                let mut truncated_inp = inp;
                let mut truncated_lbl = lbl;
                truncated_inp.truncate(max_seq_len);
                truncated_lbl.truncate(max_seq_len);
                let t_len = truncated_inp.len();
                let pos = (0..t_len as u32).collect();
                let seg = vec![0u32; t_len];
                packed_batches.push(PackedSequence {
                    input_ids: truncated_inp,
                    labels: truncated_lbl,
                    position_ids: pos,
                    segment_ids: seg,
                    seq_lengths: vec![t_len],
                });
                continue;
            }

            if cur_inp.len() + seq_len > max_seq_len {
                // Flush current packed block
                while cur_inp.len() < max_seq_len {
                    cur_inp.push(PAD_TOKEN_ID);
                    cur_lbl.push(IGNORE_LABEL_ID);
                    cur_pos.push(0);
                    cur_seg.push(current_segment);
                }
                packed_batches.push(PackedSequence {
                    input_ids: std::mem::take(&mut cur_inp),
                    labels: std::mem::take(&mut cur_lbl),
                    position_ids: std::mem::take(&mut cur_pos),
                    segment_ids: std::mem::take(&mut cur_seg),
                    seq_lengths: std::mem::take(&mut cur_lengths),
                });
                cur_inp = Vec::with_capacity(max_seq_len);
                cur_lbl = Vec::with_capacity(max_seq_len);
                cur_pos = Vec::with_capacity(max_seq_len);
                cur_seg = Vec::with_capacity(max_seq_len);
                current_segment = 0;
            }

            // Pack this sequence
            for (idx, (&t, &l)) in inp.iter().zip(lbl.iter()).enumerate() {
                cur_inp.push(t);
                cur_lbl.push(l);
                cur_pos.push(idx as u32); // Reset position ID for each packed sequence
                cur_seg.push(current_segment);
            }
            cur_lengths.push(seq_len);
            current_segment += 1;
        }

        if !cur_inp.is_empty() {
            while cur_inp.len() < max_seq_len {
                cur_inp.push(PAD_TOKEN_ID);
                cur_lbl.push(IGNORE_LABEL_ID);
                cur_pos.push(0);
                cur_seg.push(current_segment);
            }
            packed_batches.push(PackedSequence {
                input_ids: cur_inp,
                labels: cur_lbl,
                position_ids: cur_pos,
                segment_ids: cur_seg,
                seq_lengths: cur_lengths,
            });
        }

        Ok(packed_batches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encode_decode() {
        let tokenizer = SimpleTokenizer::new();
        let text = "Hello world! <|im_start|>user\ntest<|im_end|>";
        let tokens = tokenizer.encode(text);
        assert!(!tokens.is_empty());
        assert!(tokens.contains(&IM_START_TOKEN_ID));
        assert!(tokens.contains(&IM_END_TOKEN_ID));

        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_alpaca_parsing_and_sequence_packing() {
        let alpaca_json = r#"{"instruction": "Tell me a joke", "input": "", "output": "Sloth is fast!"}
{"instruction": "Calculate 2+2", "input": "", "output": "4"}"#;

        let tokenizer = SimpleTokenizer::new();
        // 512 tokens window easily holds both short sequences
        let packed = DatasetLoader::load_from_str(alpaca_json, DatasetFormat::Alpaca, &tokenizer, 512, true).unwrap();

        assert!(!packed.is_empty());
        let first = &packed[0];
        assert_eq!(first.input_ids.len(), 512);
        assert_eq!(first.seq_lengths.len(), 2); // Both sequences packed into single window!
        assert_eq!(first.segment_ids[0], 0);
        // Prompt tokens should have -100 label
        assert_eq!(first.labels[0], IGNORE_LABEL_ID);
    }

    #[test]
    fn test_chatml_parsing() {
        let chatml_json = r#"{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}"#;
        let samples = DatasetLoader::parse_samples(chatml_json, DatasetFormat::ChatML).unwrap();
        assert_eq!(samples.len(), 1);
        assert!(samples[0].prompt.contains("You are helpful"));
        assert!(samples[0].completion.contains("Hello!"));
    }
}
