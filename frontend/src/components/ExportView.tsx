import React, { useState } from 'react';
import { 
  Download, Check, Copy, HardDrive, 
  Zap, RefreshCw, CheckCircle2 
} from 'lucide-react';

export const ExportView: React.FC = () => {
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('checkpoint-step-1000 (Loss 0.642)');
  const [baseModel] = useState('Llama-3.2-3B-Instruct');
  const [quantization, setQuantization] = useState<'Q4_K_M' | 'Q8_0' | 'F16'>('Q4_K_M');
  const [mergeMode, setMergeMode] = useState<'fused_gguf' | 'lora_adapter'>('fused_gguf');

  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [currentStepText, setCurrentStepText] = useState('');
  const [isCompleted, setIsCompleted] = useState(false);
  const [copiedCmd, setCopiedCmd] = useState(false);

  const handleStartExport = async () => {
    setIsExporting(true);
    setIsCompleted(false);
    setExportProgress(5);
    setCurrentStepText('Загрузка исходных тензоров базовой модели...');

    await new Promise(r => setTimeout(r, 600));
    setExportProgress(25);
    setCurrentStepText('Вычисление весов адаптера LoRA W = W0 + (alpha/r)*(B @ A)...');

    await new Promise(r => setTimeout(r, 700));
    setExportProgress(55);
    setCurrentStepText(`Применение квантования k-quants (${quantization}) на Vulkan compute...`);

    await new Promise(r => setTimeout(r, 800));
    setExportProgress(85);
    setCurrentStepText('Запись GGUF метаданных, токенизатора и заголовков...');

    await new Promise(r => setTimeout(r, 500));
    setExportProgress(100);
    setCurrentStepText('Экспорт успешно завершен! Файл сохранен в ./models/');
    setIsExporting(false);
    setIsCompleted(true);
  };

  const outputFileName = `slothforge-${baseModel.toLowerCase()}-${quantization.toLowerCase()}.gguf`;
  const runCmd = `ollama create slothforge -f ./Modelfile && ollama run slothforge`;

  const handleCopyCmd = () => {
    navigator.clipboard.writeText(runCmd);
    setCopiedCmd(true);
    setTimeout(() => setCopiedCmd(false), 2000);
  };

  return (
    <div className="w-full max-w-[1720px] mx-auto px-4 lg:px-6 py-6 flex flex-col gap-6">
      
      {/* Header */}
      <div className="w-full bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#181F2E] border border-[#273248] flex items-center justify-center text-[#FF7A00]">
            <Download className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              Слияние LoRA и экспорт в GGUF
              <span className="text-xs px-2 py-0.5 rounded-full bg-[#FF7A00]/15 text-[#FF7A00] font-mono-code font-medium border border-[#FF7A00]/30">
                Zero-Loss Fusion
              </span>
            </h1>
            <p className="text-xs text-[#94A3B8] mt-0.5">
              Встраивание дообученных LoRA весов в базовую модель и компиляция готового GGUF для Ollama, LM Studio или llama.cpp
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 font-mono-code text-xs text-[#94A3B8] bg-[#0B0E14] px-3 py-1.5 rounded-lg border border-[#1E2638]">
          <HardDrive className="w-4 h-4 text-[#FF7A00]" />
          <span>Свободно на диске: <strong>48.2 ГБ</strong></span>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* Left: Configuration (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col gap-5 shadow-sm">
            <div className="flex items-center justify-between border-b border-[#1E2638] pb-3">
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                Параметры слияния и целевой формат
              </span>
              <span className="text-xs font-mono-code text-[#10B981]">GGUF v3 Specification</span>
            </div>

            {/* Checkpoint selector */}
            <div className="flex flex-col gap-1.5 font-mono-code text-xs">
              <label className="text-[11px] text-[#94A3B8] uppercase">Выбор сохраненного чекпоинта</label>
              <select
                value={selectedCheckpoint}
                onChange={(e) => setSelectedCheckpoint(e.target.value)}
                className="w-full bg-[#181F2E] border border-[#273248] rounded-lg px-3 py-2 text-white focus:outline-none focus:border-[#FF7A00]"
              >
                <option value="checkpoint-step-1000 (Loss 0.642)">checkpoint-step-1000 (Loss 0.642) — Финальный шаг</option>
                <option value="checkpoint-step-800 (Loss 0.710)">checkpoint-step-800 (Loss 0.710)</option>
                <option value="checkpoint-step-500 (Loss 0.890)">checkpoint-step-500 (Loss 0.890)</option>
              </select>
            </div>

            {/* Merge Mode Toggle */}
            <div className="flex flex-col gap-2">
              <label className="text-xs text-white font-semibold">Тип экспорта</label>
              <div className="grid grid-cols-2 gap-3">
                <div
                  onClick={() => setMergeMode('fused_gguf')}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all flex flex-col gap-1 ${
                    mergeMode === 'fused_gguf'
                      ? 'bg-[#181F2E] border-[#FF7A00] shadow-inner'
                      : 'bg-[#121722] border-[#1E2638] text-[#94A3B8] hover:border-[#273248]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white">Полное слияние (Fused GGUF)</span>
                    {mergeMode === 'fused_gguf' && <span className="text-[#FF7A00] text-xs">✓</span>}
                  </div>
                  <p className="text-[11px] text-[#64748B]">
                    LoRA объединяется с весами базовой модели. Единый автономный файл, готовый к запуску.
                  </p>
                </div>

                <div
                  onClick={() => setMergeMode('lora_adapter')}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all flex flex-col gap-1 ${
                    mergeMode === 'lora_adapter'
                      ? 'bg-[#181F2E] border-[#FF7A00] shadow-inner'
                      : 'bg-[#121722] border-[#1E2638] text-[#94A3B8] hover:border-[#273248]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white">Только LoRA адаптер</span>
                    {mergeMode === 'lora_adapter' && <span className="text-[#FF7A00] text-xs">✓</span>}
                  </div>
                  <p className="text-[11px] text-[#64748B]">
                    Экспорт компактного адаптера (~16–30 МБ) для динамического подключения.
                  </p>
                </div>
              </div>
            </div>

            {/* GGUF Quantization Presets */}
            <div className="flex flex-col gap-2.5 pt-2 border-t border-[#1E2638]">
              <div className="flex justify-between items-center text-xs">
                <label className="text-white font-semibold">Уровень квантования GGUF</label>
                <span className="text-[#FF7A00] font-mono-code font-bold">{quantization}</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono-code text-xs">
                <div
                  onClick={() => setQuantization('Q4_K_M')}
                  className={`p-3 rounded-xl border cursor-pointer transition-all flex flex-col gap-1 ${
                    quantization === 'Q4_K_M'
                      ? 'bg-[#181F2E] border-[#FF7A00]'
                      : 'bg-[#121722] border-[#1E2638] text-[#94A3B8] hover:border-[#273248]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-white font-bold">Q4_K_M</span>
                    <span className="text-[9px] px-1.5 py-0.2 rounded bg-[#10B981]/20 text-[#10B981]">РЕКОМЕНДУЕМ</span>
                  </div>
                  <span className="text-[10px] text-[#94A3B8]">Размер: ~2.1 ГБ</span>
                  <span className="text-[10px] text-[#64748B]">Для 4 ГБ VRAM (RX 570)</span>
                </div>

                <div
                  onClick={() => setQuantization('Q8_0')}
                  className={`p-3 rounded-xl border cursor-pointer transition-all flex flex-col gap-1 ${
                    quantization === 'Q8_0'
                      ? 'bg-[#181F2E] border-[#FF7A00]'
                      : 'bg-[#121722] border-[#1E2638] text-[#94A3B8] hover:border-[#273248]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-white font-bold">Q8_0</span>
                    <span className="text-[9px] text-[#94A3B8]">8-Bit Exact</span>
                  </div>
                  <span className="text-[10px] text-[#94A3B8]">Размер: ~3.4 ГБ</span>
                  <span className="text-[10px] text-[#64748B]">Качество без сжатия</span>
                </div>

                <div
                  onClick={() => setQuantization('F16')}
                  className={`p-3 rounded-xl border cursor-pointer transition-all flex flex-col gap-1 ${
                    quantization === 'F16'
                      ? 'bg-[#181F2E] border-[#FF7A00]'
                      : 'bg-[#121722] border-[#1E2638] text-[#94A3B8] hover:border-[#273248]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-white font-bold">F16</span>
                    <span className="text-[9px] text-[#94A3B8]">Float-16</span>
                  </div>
                  <span className="text-[10px] text-[#94A3B8]">Размер: ~6.2 ГБ</span>
                  <span className="text-[10px] text-[#64748B]">Для CPU / 8GB+ GPU</span>
                </div>
              </div>
            </div>

            {/* Action Trigger Button */}
            <div className="pt-3 border-t border-[#1E2638]">
              <button
                onClick={handleStartExport}
                disabled={isExporting}
                className="btn-pill-primary w-full py-3 text-sm gap-2"
              >
                {isExporting ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Идет слияние и экспорт...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4 fill-white" />
                    <span>Выполнить слияние и экспорт в GGUF</span>
                  </>
                )}
              </button>
            </div>

          </div>
        </div>

        {/* Right: Progress, Details & One-Click Deployment (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          
          {/* Progress Card */}
          <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col gap-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-[#1E2638] pb-3">
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                Статус операции экспорта
              </span>
              <span className="text-xs font-mono-code text-[#FF7A00]">
                {exportProgress}%
              </span>
            </div>

            {/* Progress bar */}
            <div className="w-full bg-[#181F2E] h-2.5 rounded-full overflow-hidden border border-[#273248]">
              <div
                className="h-full bg-gradient-to-r from-[#FF7A00] to-[#FF8F26] rounded-full transition-all duration-300"
                style={{ width: `${exportProgress}%` }}
              />
            </div>

            <p className="text-xs font-mono-code text-[#94A3B8] min-h-[36px]">
              {currentStepText || 'Готов к запуску процедуры экспорта.'}
            </p>

            {isCompleted && (
              <div className="p-3 bg-[#10B981]/10 border border-[#10B981]/30 rounded-xl flex items-center gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-[#10B981]" />
                <span className="text-xs font-medium text-white">
                  Файл готов к запуску в llama.cpp, Ollama или vLLM!
                </span>
              </div>
            )}
          </div>

          {/* Model Deployment Card */}
          <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col gap-4 shadow-sm">
            <span className="text-xs font-bold text-white uppercase tracking-wider">
              Запуск в Ollama / llama.cpp
            </span>

            <div className="p-3 bg-[#0B0E14] border border-[#1E2638] rounded-xl flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs font-mono-code">
                <span className="text-[#94A3B8]">Целевой файл:</span>
                <span className="text-white font-bold">{outputFileName}</span>
              </div>
              <div className="flex items-center justify-between text-xs font-mono-code">
                <span className="text-[#94A3B8]">Формат:</span>
                <span className="text-[#10B981]">GGUF ({quantization})</span>
              </div>
            </div>

            <div className="p-3 bg-[#0B0E14] border border-[#1E2638] rounded-xl flex flex-col gap-2">
              <div className="flex items-center justify-between text-[11px] font-mono-code">
                <span className="text-[#94A3B8]">Команда запуска:</span>
                <button
                  onClick={handleCopyCmd}
                  className="text-[#FF7A00] hover:underline flex items-center gap-1 text-[10px]"
                >
                  {copiedCmd ? <Check className="w-3 h-3 text-[#10B981]" /> : <Copy className="w-3 h-3" />}
                  Скопировать
                </button>
              </div>
              <code className="text-[10px] font-mono-code text-white bg-[#121722] p-2 rounded border border-[#273248] break-all select-all">
                {runCmd}
              </code>
            </div>

          </div>

        </div>

      </div>

    </div>
  );
};
