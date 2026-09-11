import React, { useState } from 'react';
import { 
  Network, Server, Cpu, HardDrive, Zap, 
  Copy, Check, RefreshCw 
} from 'lucide-react';
import type { TelemetryData, ClusterWorker } from '../types';
import { telemetryService } from '../services/api';

interface ClusterViewProps {
  telemetry: TelemetryData;
}

export const ClusterView: React.FC<ClusterViewProps> = ({ telemetry }) => {
  const [clusterMode, setClusterMode] = useState<'local' | 'master' | 'worker'>('master');
  const [copiedToken, setCopiedToken] = useState(false);
  const [copiedCmd, setCopiedCmd] = useState(false);

  // Master State
  const masterIp = '192.168.1.100';
  const masterPort = 8080;
  const authToken = 'sf_cluster_polaris_a9f82d1c9e4b';
  const [workers] = useState<ClusterWorker[]>(telemetryService.getMockWorkers());

  // Worker State
  const [targetMasterIp, setTargetMasterIp] = useState('192.168.1.100:8080');
  const [workerToken, setWorkerToken] = useState('sf_cluster_polaris_a9f82d1c9e4b');
  const [isConnecting, setIsConnecting] = useState(false);
  const [pingResult, setPingResult] = useState<{
    latencyMs: number;
    status: 'ok' | 'warning' | 'error';
    allReduceReady: boolean;
  } | null>({
    latencyMs: 1.2,
    status: 'ok',
    allReduceReady: true
  });

  const handleCopyToken = () => {
    navigator.clipboard.writeText(authToken);
    setCopiedToken(true);
    setTimeout(() => setCopiedToken(false), 2000);
  };

  const handleCopyCmd = () => {
    const cmd = `slothforge-worker --connect ${masterIp}:${masterPort} --token ${authToken} --device vulkan:0`;
    navigator.clipboard.writeText(cmd);
    setCopiedCmd(true);
    setTimeout(() => setCopiedCmd(false), 2000);
  };

  const handlePingTest = async () => {
    setIsConnecting(true);
    const res = await telemetryService.pingWorker(targetMasterIp);
    setIsConnecting(false);
    setPingResult({
      latencyMs: res.latencyMs,
      status: res.latencyMs < 5.0 ? 'ok' : 'warning',
      allReduceReady: res.allReduceReady
    });
    telemetryService.setClusterMode('master', 2);
  };

  const isClusterActive = clusterMode !== 'local' && workers.length >= 2;

  return (
    <div className="w-full max-w-[1720px] mx-auto px-4 lg:px-6 py-6 flex flex-col gap-6">
      
      {/* Cluster Header Card */}
      <div className="w-full bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#181F2E] border border-[#273248] flex items-center justify-center text-[#FF7A00]">
            <Network className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              Распределённое обучение на 2 компьютерах
              <span className="text-xs px-2 py-0.5 rounded-full bg-[#10B981]/15 text-[#10B981] font-mono-code font-medium border border-[#10B981]/30">
                2x RX 570 4GB (8 GB VRAM Ring)
              </span>
            </h1>
            <p className="text-xs text-[#94A3B8] mt-0.5">
              Синхронизация градиентов AllReduce через локальную сеть (LAN 1 Gbps / Wi-Fi 6) без накладных расходов
            </p>
          </div>
        </div>

        {/* Cluster Mode Selector: [ Локально (1 ПК) | Координатор (Master) | Воркер (Worker) ] */}
        <div className="flex items-center bg-[#0B0E14] p-1 rounded-full border border-[#1E2638] select-none">
          <button
            onClick={() => {
              setClusterMode('local');
              telemetryService.setClusterMode('local', 1);
            }}
            className={`px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all ${
              clusterMode === 'local'
                ? 'bg-[#181F2E] text-white border border-[#273248]'
                : 'text-[#94A3B8] hover:text-white border border-transparent'
            }`}
          >
            Локально (1 ПК)
          </button>
          
          <button
            onClick={() => {
              setClusterMode('master');
              telemetryService.setClusterMode('master', 2);
            }}
            className={`px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all ${
              clusterMode === 'master'
                ? 'bg-[#181F2E] text-[#FF7A00] border border-[#273248]'
                : 'text-[#94A3B8] hover:text-white border border-transparent'
            }`}
          >
            Координатор (Master)
          </button>

          <button
            onClick={() => {
              setClusterMode('worker');
              telemetryService.setClusterMode('worker', 2);
            }}
            className={`px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all ${
              clusterMode === 'worker'
                ? 'bg-[#181F2E] text-white border border-[#273248]'
                : 'text-[#94A3B8] hover:text-white border border-transparent'
            }`}
          >
            Воркер (Worker)
          </button>
        </div>
      </div>

      {/* Cluster Overview Stats Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-[#121722] border border-[#1E2638] rounded-xl p-4 flex items-center justify-between">
          <div className="flex flex-col">
            <span className="text-[11px] text-[#94A3B8] font-mono-code uppercase">Совокупная VRAM</span>
            <span className="text-xl font-bold font-mono-code text-white mt-0.5">
              {isClusterActive ? '8,192 MB' : `${telemetry.vramTotalMb} MB`}
            </span>
            <span className="text-[10px] text-[#10B981] font-mono-code">2x видеокарты Polaris-10</span>
          </div>
          <HardDrive className="w-8 h-8 text-[#FF7A00]/40" />
        </div>

        <div className="bg-[#121722] border border-[#1E2638] rounded-xl p-4 flex items-center justify-between">
          <div className="flex flex-col">
            <span className="text-[11px] text-[#94A3B8] font-mono-code uppercase">Ускорение батча</span>
            <span className="text-xl font-bold font-mono-code text-[#FF7A00] mt-0.5">
              {isClusterActive ? '1.94x' : '1.0x'}
            </span>
            <span className="text-[10px] text-[#94A3B8] font-mono-code">5,480 tok/s AllReduce</span>
          </div>
          <Zap className="w-8 h-8 text-[#FF7A00]/40" />
        </div>

        <div className="bg-[#121722] border border-[#1E2638] rounded-xl p-4 flex items-center justify-between">
          <div className="flex flex-col">
            <span className="text-[11px] text-[#94A3B8] font-mono-code uppercase">Задержка кольца (Ring)</span>
            <span className="text-xl font-bold font-mono-code text-white mt-0.5">
              {pingResult ? `${pingResult.latencyMs} мс` : '—'}
            </span>
            <span className="text-[10px] text-[#10B981] font-mono-code">Потери пакетов: 0.0%</span>
          </div>
          <Network className="w-8 h-8 text-[#10B981]/40" />
        </div>

        <div className="bg-[#121722] border border-[#1E2638] rounded-xl p-4 flex items-center justify-between">
          <div className="flex flex-col">
            <span className="text-[11px] text-[#94A3B8] font-mono-code uppercase">Статус синхронизации</span>
            <span className="text-base font-bold font-mono-code text-[#10B981] mt-1 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#10B981]" />
              Готов к AllReduce
            </span>
            <span className="text-[10px] text-[#94A3B8] font-mono-code">Ring Topo: 0 -&gt; 1 -&gt; 0</span>
          </div>
          <Server className="w-8 h-8 text-[#10B981]/40" />
        </div>
      </div>

      {/* Dynamic View based on Mode: Master vs Worker vs Local */}
      {clusterMode === 'master' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left: Master Connection Info (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col gap-4">
              <div className="flex items-center justify-between border-b border-[#1E2638] pb-3">
                <span className="text-xs font-bold text-white uppercase tracking-wider">
                  Параметры координатора (Master)
                </span>
                <span className="text-xs font-mono-code text-[#10B981]">ONLINE</span>
              </div>

              <div className="flex flex-col gap-3 font-mono-code text-xs">
                <div>
                  <label className="text-[11px] text-[#94A3B8] uppercase">Локальный IP хоста</label>
                  <div className="mt-1 p-2.5 rounded-lg bg-[#181F2E] border border-[#273248] text-white flex justify-between items-center">
                    <span>{masterIp}</span>
                    <span className="text-[10px] text-[#94A3B8]">eth0 (1000BASE-T)</span>
                  </div>
                </div>

                <div>
                  <label className="text-[11px] text-[#94A3B8] uppercase">Порт координатора AllReduce</label>
                  <div className="mt-1 p-2.5 rounded-lg bg-[#181F2E] border border-[#273248] text-white flex justify-between items-center">
                    <span>{masterPort}</span>
                    <span className="text-[10px] text-[#10B981]">Слушает входящие сокеты</span>
                  </div>
                </div>

                <div>
                  <label className="text-[11px] text-[#94A3B8] uppercase">Токен авторизации ноды</label>
                  <div className="mt-1 p-2.5 rounded-lg bg-[#181F2E] border border-[#273248] text-white flex justify-between items-center">
                    <span className="truncate pr-2">{authToken}</span>
                    <button
                      onClick={handleCopyToken}
                      className="p-1 rounded hover:bg-[#273248] text-[#94A3B8] hover:text-white transition-colors"
                      title="Скопировать токен"
                    >
                      {copiedToken ? <Check className="w-3.5 h-3.5 text-[#10B981]" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                  </div>
                </div>
              </div>

              {/* One-click Worker Launch Command */}
              <div className="mt-2 p-3 bg-[#0B0E14] border border-[#1E2638] rounded-xl flex flex-col gap-2">
                <div className="flex items-center justify-between text-[11px] font-mono-code">
                  <span className="text-[#94A3B8]">Команда для второго ПК:</span>
                  <button
                    onClick={handleCopyCmd}
                    className="text-[#FF7A00] hover:underline flex items-center gap-1 text-[10px]"
                  >
                    {copiedCmd ? <Check className="w-3 h-3 text-[#10B981]" /> : <Copy className="w-3 h-3" />}
                    Скопировать команду
                  </button>
                </div>
                <code className="text-[10px] font-mono-code text-[#FFFFFF] bg-[#121722] p-2 rounded border border-[#273248] break-all select-all">
                  slothforge-worker --connect {masterIp}:{masterPort} --token {authToken}
                </code>
              </div>
            </div>
          </div>

          {/* Right: Connected Workers List & Topology (7 cols) */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col gap-4">
              <div className="flex items-center justify-between border-b border-[#1E2638] pb-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-white uppercase tracking-wider">
                    Подключенные ноды кластера (2x ПК)
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-[#181F2E] text-white text-[10px] font-mono-code border border-[#273248]">
                    {workers.length} / 2 нод активно
                  </span>
                </div>
                <button
                  onClick={handlePingTest}
                  disabled={isConnecting}
                  className="btn-pill-secondary gap-1.5 py-1 px-3 text-xs"
                >
                  <RefreshCw className={`w-3 h-3 ${isConnecting ? 'animate-spin' : ''}`} />
                  <span>Проверить пинг</span>
                </button>
              </div>

              {/* Workers Table / Cards */}
              <div className="flex flex-col gap-3">
                {workers.map((worker, index) => (
                  <div
                    key={worker.id}
                    className="p-4 rounded-xl bg-[#181F2E] border border-[#273248] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-lg bg-[#121722] border border-[#1E2638] flex items-center justify-center font-mono-code text-xs font-bold text-[#FF7A00]">
                        #{index}
                      </div>
                      <div className="flex flex-col">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-bold text-white">{worker.name}</span>
                          <span className="px-1.5 py-0.2 rounded text-[9px] font-mono-code bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/40">
                            {worker.status.toUpperCase()}
                          </span>
                        </div>
                        <span className="text-[11px] font-mono-code text-[#94A3B8]">
                          {worker.ip} • {worker.gpu}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center gap-6 font-mono-code text-xs">
                      <div className="flex flex-col text-right">
                        <span className="text-[10px] text-[#64748B]">VRAM</span>
                        <span className="text-white font-bold">{worker.vramMb} MB</span>
                      </div>
                      <div className="flex flex-col text-right">
                        <span className="text-[10px] text-[#64748B]">Пинг</span>
                        <span className="text-[#10B981] font-bold">{worker.latencyMs} мс</span>
                      </div>
                      <div className="flex flex-col text-right">
                        <span className="text-[10px] text-[#64748B]">Throughput</span>
                        <span className="text-[#FF7A00] font-bold">{worker.throughputRatio}x</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Ring Topology Visual Card */}
              <div className="p-4 bg-[#0B0E14] border border-[#1E2638] rounded-xl flex flex-col gap-2">
                <span className="text-[11px] font-mono-code text-[#94A3B8] uppercase">
                  Схема кольцевого AllReduce (Ring Topology):
                </span>
                <div className="flex items-center justify-center gap-4 py-3 font-mono-code text-xs">
                  <div className="px-3 py-2 rounded-lg bg-[#181F2E] border border-[#FF7A00] text-center">
                    <div className="text-white font-bold">Node #0 (Master)</div>
                    <div className="text-[10px] text-[#94A3B8]">RX 570 4GB [Rank 0]</div>
                  </div>
                  <div className="flex flex-col items-center text-[#FF7A00]">
                    <span className="text-[10px]">1.2 ms AllReduce</span>
                    <div className="flex items-center gap-1">
                      <span>⇄</span>
                    </div>
                  </div>
                  <div className="px-3 py-2 rounded-lg bg-[#181F2E] border border-[#10B981] text-center">
                    <div className="text-white font-bold">Node #1 (Worker)</div>
                    <div className="text-[10px] text-[#94A3B8]">RX 570 4GB [Rank 1]</div>
                  </div>
                </div>
                <div className="text-[10px] text-[#64748B] text-center font-mono-code">
                  Синхронизация только матриц адаптера LoRA: объем передаваемых данных всего ~12 MB за шаг (0.09 сек на 1Gbps)
                </div>
              </div>
            </div>
          </div>

        </div>
      )}

      {clusterMode === 'worker' && (
        <div className="w-full max-w-2xl mx-auto bg-[#121722] border border-[#1E2638] rounded-2xl p-6 flex flex-col gap-5">
          <div className="border-b border-[#1E2638] pb-4">
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Server className="w-5 h-5 text-[#FF7A00]" />
              Подключение воркера к основному координатору
            </h2>
            <p className="text-xs text-[#94A3B8] mt-1">
              Этот компьютер будет обрабатывать половину микро-батчей и синхронизировать градиенты со вторым ПК
            </p>
          </div>

          <div className="flex flex-col gap-4 font-mono-code text-xs">
            <div>
              <label className="text-[11px] text-[#94A3B8] uppercase">IP и порт координатора (Master IP)</label>
              <input
                type="text"
                value={targetMasterIp}
                onChange={(e) => setTargetMasterIp(e.target.value)}
                placeholder="192.168.1.100:8080"
                className="mt-1 w-full bg-[#181F2E] border border-[#273248] rounded-lg px-3 py-2 text-white focus:outline-none focus:border-[#FF7A00]"
              />
            </div>

            <div>
              <label className="text-[11px] text-[#94A3B8] uppercase">Токен авторизации (Cluster Token)</label>
              <input
                type="password"
                value={workerToken}
                onChange={(e) => setWorkerToken(e.target.value)}
                className="mt-1 w-full bg-[#181F2E] border border-[#273248] rounded-lg px-3 py-2 text-white focus:outline-none focus:border-[#FF7A00]"
              />
            </div>

            {/* Ping Result Banner */}
            {pingResult && (
              <div className="p-3 bg-[#181F2E] border border-[#273248] rounded-xl flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-[#10B981]" />
                  <span className="text-white font-medium">
                    🟢 Пинг: <strong>{pingResult.latencyMs} мс</strong> — Готов к AllReduce
                  </span>
                </div>
                <span className="text-[10px] text-[#10B981] font-bold">1 Gbps LAN OK</span>
              </div>
            )}

            <button
              onClick={handlePingTest}
              disabled={isConnecting}
              className="btn-pill-primary w-full py-3 text-sm gap-2 mt-2"
            >
              <Zap className="w-4 h-4 fill-white" />
              <span>{isConnecting ? 'Проверка соединения...' : 'Подключиться для 2x ускорения'}</span>
            </button>
          </div>
        </div>
      )}

      {clusterMode === 'local' && (
        <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-6 flex flex-col items-center justify-center text-center gap-3">
          <Cpu className="w-12 h-12 text-[#FF7A00]/60" />
          <h2 className="text-base font-bold text-white">Локальный режим (1 ПК)</h2>
          <p className="text-xs text-[#94A3B8] max-w-lg">
            Обучение выполняется исключительно на локальной видеокарте AMD Radeon RX 570 4GB. Сетевая синхронизация отключена, весь VRAM бюджет ограничен 4096 MB.
          </p>
          <button
            onClick={() => setClusterMode('master')}
            className="btn-pill-secondary mt-2 text-xs"
          >
            Переключить в режим 2-PC кластера
          </button>
        </div>
      )}

    </div>
  );
};
