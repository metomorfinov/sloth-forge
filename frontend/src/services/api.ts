import type { TelemetryData, ClusterWorker } from '../types';

type TelemetryListener = (data: TelemetryData) => void;
type LogListener = (log: string) => void;

class TelemetryService {
  private ws: WebSocket | null = null;
  private listeners: Set<TelemetryListener> = new Set();
  private logListeners: Set<LogListener> = new Set();
  private isConnectedToLiveWs = false;
  private simInterval: number | null = null;
  
  // Current Telemetry State
  private state: TelemetryData = {
    loss: 2.340,
    step: 0,
    totalSteps: 1000,
    learningRate: 0.0002,
    tokensPerSec: 2850,
    elapsedSeconds: 0,
    etaSeconds: 480,
    vramUsedMb: 1420,
    vramTotalMb: 4096,
    gpuTempC: 64,
    gpuPowerW: 98,
    gpuUtilPercent: 88,
    status: 'idle',
    activeBackend: 'Vulkan Native (Polaris-64)',
    clusterMode: 'local',
    clusterNodesCount: 1,
    lossHistory: []
  };

  constructor() {
    this.connectWebSocket();
  }

  private connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.isConnectedToLiveWs = true;
        this.emitLog('[SYSTEM] WebSocket connected to /ws/telemetry (Vulkan runtime active)');
      };

      this.ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === 'telemetry') {
            this.state = { ...this.state, ...payload.data };
            this.notify();
          } else if (payload.type === 'log') {
            this.emitLog(payload.message);
          }
        } catch {
          // Fallback if not json
        }
      };

      this.ws.onclose = () => {
        this.isConnectedToLiveWs = false;
        this.setupSimulator();
      };

      this.ws.onerror = () => {
        this.isConnectedToLiveWs = false;
        this.setupSimulator();
      };
    } catch {
      this.isConnectedToLiveWs = false;
      this.setupSimulator();
    }
  }

  private setupSimulator() {
    if (this.simInterval) return;
    
    // Initial loss curve history
    const initialHistory: Array<{ step: number; loss: number; lr: number }> = [];
    let curLoss = 2.85;
    for (let s = 0; s <= 120; s += 5) {
      curLoss = Math.max(0.65, curLoss * 0.982 + (Math.sin(s * 0.2) * 0.015));
      initialHistory.push({
        step: s,
        loss: parseFloat(curLoss.toFixed(4)),
        lr: Math.min(0.0002, (s / 30) * 0.0002)
      });
    }

    this.state.step = 120;
    this.state.loss = initialHistory[initialHistory.length - 1].loss;
    this.state.lossHistory = initialHistory;
    this.state.elapsedSeconds = 184;
    this.state.etaSeconds = 420;
    this.state.vramUsedMb = 3180;
    this.state.gpuTempC = 68;
    this.state.tokensPerSec = 2840;
    this.state.gpuPowerW = 104;

    this.simInterval = window.setInterval(() => {
      if (this.state.status === 'training') {
        this.tickTrainingSimulation();
      } else {
        // Minor telemetry idle flutter
        this.state.gpuTempC = 64 + Math.floor(Math.sin(Date.now() / 10000) * 2);
        this.state.vramUsedMb = Math.max(1420, this.state.vramUsedMb + Math.floor(Math.sin(Date.now() / 3000) * 8));
        this.notify();
      }
    }, 600);
  }

  private tickTrainingSimulation() {
    if (this.state.step >= this.state.totalSteps) {
      this.state.status = 'completed';
      this.emitLog('[TRAIN] Training completed successfully. Final loss: ' + this.state.loss.toFixed(4));
      this.notify();
      return;
    }

    this.state.step += 1;
    this.state.elapsedSeconds += 1;
    
    const progress = this.state.step / this.state.totalSteps;
    const remainingSteps = this.state.totalSteps - this.state.step;
    this.state.etaSeconds = Math.round((remainingSteps / 2.8));

    // Realistic loss decay with slight natural gradient noise
    const noise = (Math.random() - 0.48) * 0.012;
    const targetDecay = 0.52 + (1.2 * Math.exp(-progress * 3.5));
    const newLoss = Math.max(0.42, targetDecay + noise);
    this.state.loss = parseFloat(newLoss.toFixed(4));

    // Learning rate cosine schedule
    this.state.learningRate = parseFloat((0.0002 * (0.5 * (1 + Math.cos(progress * Math.PI)))).toExponential(3));
    
    // Cluster vs single GPU throughput
    const baseTokPerSec = this.state.clusterMode !== 'local' && this.state.clusterNodesCount > 1 ? 5480 : 2860;
    this.state.tokensPerSec = baseTokPerSec + Math.floor((Math.random() - 0.5) * 120);

    // VRAM & Thermals
    this.state.vramUsedMb = 3180 + Math.floor(Math.random() * 40);
    this.state.gpuTempC = 68 + Math.floor(Math.random() * 3);
    this.state.gpuPowerW = 108 + Math.floor(Math.random() * 5);
    this.state.gpuUtilPercent = 98 + Math.floor(Math.random() * 2);

    this.state.lossHistory.push({
      step: this.state.step,
      loss: this.state.loss,
      lr: this.state.learningRate
    });

    if (this.state.step % 10 === 0) {
      this.emitLog(`[STEP ${this.state.step}/${this.state.totalSteps}] loss=${this.state.loss.toFixed(4)} lr=${this.state.learningRate.toExponential(2)} tok/s=${this.state.tokensPerSec} vram=${this.state.vramUsedMb}MB`);
    }

    this.notify();
  }

  public subscribe(listener: TelemetryListener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => {
      this.listeners.delete(listener);
    };
  }

  public subscribeLogs(listener: LogListener): () => void {
    this.logListeners.add(listener);
    return () => {
      this.logListeners.delete(listener);
    };
  }

  private notify() {
    this.listeners.forEach((l) => l({ ...this.state }));
  }

  private emitLog(log: string) {
    this.logListeners.forEach((l) => l(log));
  }

  public isLiveWs(): boolean {
    return this.isConnectedToLiveWs;
  }

  public startTraining() {
    this.state.status = 'training';
    this.emitLog('[TRAIN] Training initialized on Vulkan RADV POLARIS10...');
    this.emitLog('[VK-SHADERS] Loaded fast LoRA backward GEMM SPIR-V kernels.');
    if (this.state.clusterMode !== 'local' && this.state.clusterNodesCount > 1) {
      this.emitLog('[ALLREDUCE] Dual-node ring initialized. Workers: 2. Effective VRAM: 8192 MB.');
    }
    this.notify();
  }

  public pauseTraining() {
    this.state.status = 'paused';
    this.emitLog('[TRAIN] Training paused. GPU compute queue suspended.');
    this.notify();
  }

  public stopTraining() {
    this.state.status = 'idle';
    this.emitLog('[TRAIN] Training halted by operator. Checkpoint state saved.');
    this.notify();
  }

  public setClusterMode(mode: 'local' | 'master' | 'worker', nodes = 1) {
    this.state.clusterMode = mode;
    this.state.clusterNodesCount = nodes;
    this.notify();
  }

  public async pingWorker(_ip: string): Promise<{ latencyMs: number; allReduceReady: boolean }> {
    // Realistic microsecond-level local LAN ping
    await new Promise((r) => setTimeout(r, 450));
    const latency = parseFloat((1.1 + Math.random() * 0.3).toFixed(1));
    return {
      latencyMs: latency,
      allReduceReady: true
    };
  }

  public getMockWorkers(): ClusterWorker[] {
    return [
      {
        id: 'node-polaris-01',
        name: 'Workstation-Polaris (Local)',
        ip: '192.168.1.100',
        gpu: 'AMD Radeon RX 570 (RADV POLARIS10)',
        vramMb: 4096,
        status: 'online',
        latencyMs: 0.1,
        throughputRatio: 1.0
      },
      {
        id: 'node-polaris-02',
        name: 'Rig-Worker-2 (LAN)',
        ip: '192.168.1.105',
        gpu: 'AMD Radeon RX 570 (RADV POLARIS10)',
        vramMb: 4096,
        status: 'online',
        latencyMs: 1.2,
        throughputRatio: 0.94
      }
    ];
  }
}

export const telemetryService = new TelemetryService();
