import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { StudioView } from './components/StudioView';
import { ClusterView } from './components/ClusterView';
import { PlaygroundView } from './components/PlaygroundView';
import { DatasetView } from './components/DatasetView';
import { ExportView } from './components/ExportView';
import { telemetryService } from './services/api';
import type { TelemetryData } from './types';
import { Flame } from 'lucide-react';

export function App() {
  const [currentTab, setCurrentTab] = useState<'studio' | 'cluster' | 'playground' | 'datasets' | 'export'>('studio');
  const [telemetry, setTelemetry] = useState<TelemetryData>({
    loss: 2.340,
    step: 120,
    totalSteps: 1000,
    learningRate: 0.0002,
    tokensPerSec: 2840,
    elapsedSeconds: 184,
    etaSeconds: 420,
    vramUsedMb: 3180,
    vramTotalMb: 4096,
    gpuTempC: 68,
    gpuPowerW: 104,
    gpuUtilPercent: 96,
    status: 'idle',
    activeBackend: 'Vulkan Native (Polaris-64)',
    clusterMode: 'local',
    clusterNodesCount: 1,
    lossHistory: []
  });

  const [logs, setLogs] = useState<string[]>([
    '[INIT] SlothForge Vulkan Studio initialized.',
    '[DEVICE] Found AMD Radeon RX 570 (RADV POLARIS10) - 4096 MB VRAM.',
    '[VULKAN] SPIR-V shaders compiled: fast_lora_gemm.spv (ACO backend).',
    '[NETWORK] Ring-AllReduce interface initialized on 0.0.0.0:8080.',
    '[STATUS] Ready for fine-tuning or inference.'
  ]);

  const [isLiveWs, setIsLiveWs] = useState(false);

  useEffect(() => {
    const unsubTelemetry = telemetryService.subscribe((data) => {
      setTelemetry(data);
      setIsLiveWs(telemetryService.isLiveWs());
    });

    const unsubLogs = telemetryService.subscribeLogs((log) => {
      setLogs((prev) => [...prev, log].slice(-100)); // keep last 100 logs
    });

    return () => {
      unsubTelemetry();
      unsubLogs();
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#0B0E14] text-[#FFFFFF] flex flex-col selection:bg-[#FF7A00] selection:text-white">
      {/* Top Header & Telemetry Bar */}
      <Header
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        telemetry={telemetry}
        isLiveWs={isLiveWs}
      />

      {/* Main Content Area */}
      <main className="flex-1 w-full pb-12">
        {currentTab === 'studio' && (
          <StudioView telemetry={telemetry} logs={logs} />
        )}
        {currentTab === 'cluster' && (
          <ClusterView telemetry={telemetry} />
        )}
        {currentTab === 'playground' && (
          <PlaygroundView />
        )}
        {currentTab === 'datasets' && (
          <DatasetView />
        )}
        {currentTab === 'export' && (
          <ExportView />
        )}
      </main>

      {/* Professional Engineering Footer */}
      <footer className="w-full bg-[#0B0E14] border-t border-[#1E2638] py-4 px-6 select-none">
        <div className="max-w-[1720px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-[11px] font-mono-code text-[#64748B]">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1 text-[#94A3B8]">
              <Flame className="w-3.5 h-3.5 text-[#FF7A00]" />
              SlothForge 1.0 (Unsloth Vulkan Edition)
            </span>
            <span>•</span>
            <span>RADV ACO Shader Compiler</span>
            <span>•</span>
            <span>AllReduce Ring v2</span>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-[#10B981]">
              ✓ Strictly Zero AI-Slop Architecture
            </span>
            <span>•</span>
            <span>AMD Polaris 10/20 Compute</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
