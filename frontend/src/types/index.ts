export type TrainingMode = 'beginner' | 'pro';

export interface TelemetryData {
  loss: number;
  step: number;
  totalSteps: number;
  learningRate: number;
  tokensPerSec: number;
  elapsedSeconds: number;
  etaSeconds: number;
  vramUsedMb: number;
  vramTotalMb: number;
  gpuTempC: number;
  gpuPowerW: number;
  gpuUtilPercent: number;
  status: 'idle' | 'training' | 'paused' | 'completed' | 'error';
  activeBackend: string;
  clusterMode: 'local' | 'master' | 'worker';
  clusterNodesCount: number;
  lossHistory: Array<{ step: number; loss: number; lr: number }>;
}

export interface HardwareStatus {
  gpuName: string;
  driver: string;
  vramTotalMb: number;
  vramUsedMb: number;
  backend: string;
  isCluster: boolean;
  clusterTotalVramMb?: number;
}

export interface BeginnerPreset {
  id: 'style_chat' | 'knowledge_facts' | 'coding_patterns' | 'memory_save';
  title: string;
  subtitle: string;
  icon: string;
  description: string;
  loraRank: number;
  loraAlpha: number;
  epochs: number;
  learningRate: number;
  vramProfile: string;
}

export interface ProHyperparameters {
  loraRank: number;
  loraAlpha: number;
  loraDropout: number;
  targetModules: string[];
  microBatchSize: number;
  gradientAccumulation: number;
  maxSeqLength: number;
  sequencePacking: boolean;
  optimizer: 'adamw_8bit' | 'adamw_fp32';
  learningRate: number;
  weightDecay: number;
  lrSchedule: 'cosine' | 'linear' | 'constant';
  warmupRatio: number;
  gradientCheckpointing: boolean;
  baseModel: string;
}

export interface ClusterWorker {
  id: string;
  name: string;
  ip: string;
  gpu: string;
  vramMb: number;
  status: 'online' | 'syncing' | 'idle' | 'offline';
  latencyMs: number;
  throughputRatio: number;
}

export interface ClusterConfig {
  mode: 'local' | 'master' | 'worker';
  masterIp: string;
  masterPort: number;
  authToken: string;
  workers: ClusterWorker[];
  workerTargetIp: string;
  workerTargetToken: string;
  pingResult: {
    latencyMs: number;
    status: 'ok' | 'warning' | 'error';
    allReduceReady: boolean;
    timestamp: string;
  } | null;
}

export interface DatasetEntry {
  id: number;
  instruction: string;
  input: string;
  output: string;
  tokenCount: number;
}

export interface GenerationParams {
  model: string;
  systemPrompt: string;
  userPrompt: string;
  temperature: number;
  topP: number;
  maxTokens: number;
  repetitionPenalty: number;
}
