import React, { useState } from 'react';
import { 
  Play, Pause, Square, Terminal, 
  HelpCircle, Settings2, SlidersHorizontal, Layers, 
  Gauge, Flame 
} from 'lucide-react';
import type { TelemetryData, BeginnerPreset, ProHyperparameters } from '../types';
import { LossChart } from './LossChart';
import { telemetryService } from '../services/api';

interface StudioViewProps {
  telemetry: TelemetryData;
  logs: string[];
}

const BEGINNER_PRESETS: BeginnerPreset[] = [
  {
    id: 'style_chat',
    title: 'Обучение стилю и общению',
    subtitle: 'Чат-бот, тон речи, ролевые диалоги',
    icon: '🎭',
    description: 'Оптимизирован для адаптации характера ответов модели, сохранения вежливости или конкретного стиля без переобучения.',
    loraRank: 16,
    loraAlpha: 32,
    epochs: 3,
    learningRate: 0.0002,
    vramProfile: '3.1 ГБ VRAM'
  },
  {
    id: 'knowledge_facts',
    title: 'Точные знания и факты',
    subtitle: 'База документов, инструкции, статьи',
    icon: '📚',
    description: 'Углубленная настройка весов внимания для надежного запоминания терминов, регламентов компании и фактических ответов.',
    loraRank: 32,
    loraAlpha: 64,
    epochs: 4,
    learningRate: 0.00015,
    vramProfile: '3.4 ГБ VRAM'
  },
  {
    id: 'coding_patterns',
    title: 'Программирование и код',
    subtitle: 'Паттерны, скрипты, синтаксис',
    icon: '💻',
    description: 'Адаптация под структуры кода, генерацию функций и работу с API. Сфокусирован на линейных слоях проекций.',
    loraRank: 32,
    loraAlpha: 64,
    epochs: 3,
    learningRate: 0.0002,
    vramProfile: '3.3 ГБ VRAM'
  },
  {
    id: 'memory_save',
    title: 'Экономия памяти (4 ГБ VRAM)',
    subtitle: 'Пресет под RX 570 4GB без риска OOM',
    icon: '🛡️',
    description: 'Максимально бережный режим с малым рангом LoRA и градиентным накоплением. Гарантированно укладывается в 3.0 ГБ VRAM.',
    loraRank: 8,
    loraAlpha: 16,
    epochs: 2,
    learningRate: 0.00025,
    vramProfile: '2.8 ГБ VRAM'
  }
];

export const StudioView: React.FC<StudioViewProps> = ({ telemetry, logs }) => {
  const [mode, setMode] = useState<'beginner' | 'pro'>('beginner');

  // Beginner Mode State
  const [selectedPreset, setSelectedPreset] = useState<BeginnerPreset>(BEGINNER_PRESETS[3]); // Default to memory save for RX 570
  const [intensityLevel, setIntensityLevel] = useState<number>(2); // 1: Light, 2: Standard, 3: Deep
  const [pacingLevel, setPacingLevel] = useState<'fast' | 'balanced' | 'thorough'>('balanced');

  // Pro Mode State
  const [proParams, setProParams] = useState<ProHyperparameters>({
    loraRank: 16,
    loraAlpha: 32,
    loraDropout: 0.0,
    targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    microBatchSize: 1,
    gradientAccumulation: 4,
    maxSeqLength: 1024,
    sequencePacking: true,
    optimizer: 'adamw_8bit',
    learningRate: 0.0002,
    weightDecay: 0.01,
    lrSchedule: 'cosine',
    warmupRatio: 0.03,
    gradientCheckpointing: true,
    baseModel: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf'
  });

  const [alphaAutoCouple, setAlphaAutoCouple] = useState(true);

  const handleRankChange = (newRank: number) => {
    setProParams(prev => ({
      ...prev,
      loraRank: newRank,
      loraAlpha: alphaAutoCouple ? newRank * 2 : prev.loraAlpha
    }));
  };

  const toggleTargetModule = (mod: string) => {
    setProParams(prev => {
      const exists = prev.targetModules.includes(mod);
      const updated = exists 
        ? prev.targetModules.filter(m => m !== mod)
        : [...prev.targetModules, mod];
      return { ...prev, targetModules: updated };
    });
  };

  const selectAllLinear = () => {
    setProParams(prev => ({
      ...prev,
      targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    }));
  };

  const selectAttentionOnly = () => {
    setProParams(prev => ({
      ...prev,
      targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    }));
  };

  // Intensity labels for Beginner Mode
  const getIntensityLabel = (val: number) => {
    if (val === 1) return 'Легкая полировка (1 эпоха, low lr)';
    if (val === 2) return 'Стандарт (3 эпохи, сбалансировано)';
    return 'Глубокое запоминание (5 эпох, high lr)';
  };

  const formatSeconds = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const effectiveBatchSize = proParams.microBatchSize * proParams.gradientAccumulation * (telemetry.clusterNodesCount || 1);

  return (
    <div className="w-full max-w-[1720px] mx-auto px-4 lg:px-6 py-6 flex flex-col gap-6">
      
      {/* Top Banner: Dual Mode Toggle & Summary Bar */}
      <div className="w-full bg-[#121722] border border-[#1E2638] rounded-2xl p-4 lg:p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#181F2E] border border-[#273248] flex items-center justify-center text-[#FF7A00]">
            <Flame className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              Студия дообучения нейросетей
              <span className="text-xs px-2 py-0.5 rounded-full bg-[#FF7A00]/15 text-[#FF7A00] font-mono-code font-medium border border-[#FF7A00]/30">
                Unsloth Vulkan Engine
              </span>
            </h1>
            <p className="text-xs text-[#94A3B8] mt-0.5">
              Аппаратная оптимизация под архитектуру AMD Polaris (RX 470 / 480 / 570 / 580 4GB/8GB)
            </p>
          </div>
        </div>

        {/* Dual Mode Switch: [ 🟢 Режим новичка | ⚙️ Pro-режим ] */}
        <div className="flex items-center bg-[#0B0E14] p-1 rounded-full border border-[#1E2638] select-none">
          <button
            onClick={() => setMode('beginner')}
            className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold transition-all ${
              mode === 'beginner'
                ? 'bg-[#181F2E] text-white border border-[#273248] shadow-md'
                : 'text-[#94A3B8] hover:text-white border border-transparent'
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-[#10B981]" />
            <span>🟢 Режим новичка</span>
          </button>

          <button
            onClick={() => setMode('pro')}
            className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold transition-all ${
              mode === 'pro'
                ? 'bg-[#181F2E] text-white border border-[#273248] shadow-md'
                : 'text-[#94A3B8] hover:text-white border border-transparent'
            }`}
          >
            <Settings2 className="w-3.5 h-3.5 text-[#FF7A00]" />
            <span>⚙️ Pro-режим</span>
          </button>
        </div>
      </div>

      {/* Main Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* Left Column (Config): 7 cols */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          
          {mode === 'beginner' ? (
            /* ================= BEGINNER MODE ================= */
            <div className="flex flex-col gap-6">
              
              {/* 1-Click Task Presets */}
              <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-white uppercase tracking-wider">
                      1-Click Готовые пресеты задач
                    </span>
                    <div className="group relative cursor-pointer">
                      <HelpCircle className="w-3.5 h-3.5 text-[#94A3B8]" />
                      <div className="hidden group-hover:block absolute left-0 bottom-full mb-2 w-64 p-2 bg-[#181F2E] border border-[#273248] rounded text-[11px] text-[#94A3B8] shadow-lg z-20">
                        Каждый пресет автоматически выставляет оптимальные параметры LoRA, квантования и распределения памяти для вашей задачи.
                      </div>
                    </div>
                  </div>
                  <span className="text-xs font-mono-code text-[#94A3B8]">
                    {selectedPreset.title}
                  </span>
                </div>

                {/* Preset Cards Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {BEGINNER_PRESETS.map((preset) => {
                    const isSelected = selectedPreset.id === preset.id;
                    return (
                      <div
                        key={preset.id}
                        onClick={() => setSelectedPreset(preset)}
                        className={`p-4 rounded-xl border cursor-pointer transition-all flex flex-col justify-between gap-3 ${
                          isSelected
                            ? 'bg-[#181F2E] border-[#FF7A00] shadow-[0_0_15px_rgba(255,122,0,0.15)]'
                            : 'bg-[#121722] border-[#1E2638] hover:border-[#273248] hover:bg-[#181F2E]/40'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-2.5">
                            <span className="text-2xl">{preset.icon}</span>
                            <div>
                              <h3 className="text-xs font-bold text-white leading-tight">
                                {preset.title}
                              </h3>
                              <p className="text-[11px] text-[#94A3B8] leading-tight mt-0.5">
                                {preset.subtitle}
                              </p>
                            </div>
                          </div>
                          {isSelected && (
                            <div className="w-4 h-4 rounded-full bg-[#FF7A00] flex items-center justify-center text-[#0B0E14] text-[10px] font-bold">
                              ✓
                            </div>
                          )}
                        </div>

                        <p className="text-[11px] text-[#64748B] line-clamp-2 leading-relaxed">
                          {preset.description}
                        </p>

                        <div className="flex items-center justify-between pt-2 border-t border-[#1E2638] text-[10px] font-mono-code">
                          <span className="text-[#94A3B8]">Ранг: r={preset.loraRank}</span>
                          <span className="text-[#10B981] font-semibold">{preset.vramProfile}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Human-Friendly Controls Card */}
              <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col gap-5">
                <div className="flex items-center justify-between border-b border-[#1E2638] pb-3">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <SlidersHorizontal className="w-4 h-4 text-[#FF7A00]" />
                    Понятные настройки обучения
                  </h3>
                  <span className="text-xs text-[#94A3B8]">Без сложных формул и терминов</span>
                </div>

                {/* Slider 1: Интенсивность обучения */}
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-center text-xs">
                    <label className="font-semibold text-white flex items-center gap-1.5">
                      Интенсивность обучения
                      <span className="text-[10px] text-[#94A3B8] font-normal">
                        (сколько раз модель изучит датасет)
                      </span>
                    </label>
                    <span className="text-xs font-mono-code font-bold text-[#FF7A00]">
                      {getIntensityLabel(intensityLevel)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="3"
                    step="1"
                    value={intensityLevel}
                    onChange={(e) => setIntensityLevel(parseInt(e.target.value))}
                    className="w-full flame-range cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-[#64748B] font-mono-code px-1">
                    <span>1. Легкая полировка</span>
                    <span>2. Стандарт</span>
                    <span>3. Глубокое запоминание</span>
                  </div>
                  <p className="text-[11px] text-[#94A3B8] bg-[#181F2E] p-2.5 rounded-lg border border-[#1E2638] mt-1">
                    💡 <strong>Подсказка:</strong> Для изменения тона ответов достаточно 1–2 эпох. Для запоминания сложных инструкций и фактов выберите «Стандарт» или «Глубокое запоминание».
                  </p>
                </div>

                {/* Slider 2: Скорость и размер шага */}
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-center text-xs">
                    <label className="font-semibold text-white flex items-center gap-1.5">
                      Скорость и размер шага
                      <span className="text-[10px] text-[#94A3B8] font-normal">
                        (темп обучения модели)
                      </span>
                    </label>
                    <span className="text-xs font-mono-code font-bold text-[#FF7A00] uppercase">
                      {pacingLevel === 'fast' ? '⚡ Быстро' : pacingLevel === 'balanced' ? '⚖️ Сбалансированно' : '🎯 Тщательно'}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    {(['fast', 'balanced', 'thorough'] as const).map((lvl) => (
                      <button
                        key={lvl}
                        onClick={() => setPacingLevel(lvl)}
                        className={`py-2 px-3 rounded-lg text-xs font-medium border transition-all ${
                          pacingLevel === lvl
                            ? 'bg-[#181F2E] border-[#FF7A00] text-white font-semibold shadow-inner'
                            : 'bg-[#121722] border-[#1E2638] text-[#94A3B8] hover:text-white hover:border-[#273248]'
                        }`}
                      >
                        {lvl === 'fast' && '⚡ Быстро'}
                        {lvl === 'balanced' && '⚖️ Сбалансированно'}
                        {lvl === 'thorough' && '🎯 Тщательно'}
                      </button>
                    ))}
                  </div>
                  <p className="text-[11px] text-[#94A3B8] bg-[#181F2E] p-2.5 rounded-lg border border-[#1E2638] mt-1">
                    💡 <strong>Подсказка:</strong> «Сбалансированно» — золотая середина без риска забывания базового языка. «Тщательно» делает очень аккуратные шаги градиента.
                  </p>
                </div>

                {/* Summary Parameters Card */}
                <div className="bg-[#0B0E14] border border-[#1E2638] rounded-xl p-3.5 flex flex-col gap-2">
                  <span className="text-[10px] font-mono-code uppercase text-[#94A3B8] tracking-wider">
                    Результирующая конфигурация (RX 570 4GB Safe):
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono-code">
                    <div className="bg-[#121722] p-2 rounded border border-[#1E2638]">
                      <div className="text-[10px] text-[#64748B]">Ранг LoRA (r)</div>
                      <div className="text-white font-bold">{selectedPreset.loraRank} (Alpha {selectedPreset.loraAlpha})</div>
                    </div>
                    <div className="bg-[#121722] p-2 rounded border border-[#1E2638]">
                      <div className="text-[10px] text-[#64748B]">Эпохи</div>
                      <div className="text-white font-bold">{intensityLevel === 1 ? 1 : intensityLevel === 2 ? 3 : 5}</div>
                    </div>
                    <div className="bg-[#121722] p-2 rounded border border-[#1E2638]">
                      <div className="text-[10px] text-[#64748B]">Learning Rate</div>
                      <div className="text-white font-bold">
                        {pacingLevel === 'fast' ? '3.0e-4' : pacingLevel === 'balanced' ? '2.0e-4' : '1.0e-4'}
                      </div>
                    </div>
                    <div className="bg-[#121722] p-2 rounded border border-[#1E2638]">
                      <div className="text-[10px] text-[#64748B]">VRAM пик</div>
                      <div className="text-[#10B981] font-bold">~3.1 ГБ / 4 ГБ</div>
                    </div>
                  </div>
                </div>

              </div>

            </div>
          ) : (
            /* ================= PRO MODE (UNSLOTH PARITY) ================= */
            <div className="flex flex-col gap-6">
              
              {/* LoRA & Architecture Config */}
              <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col gap-5">
                <div className="flex items-center justify-between border-b border-[#1E2638] pb-3">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <Layers className="w-4 h-4 text-[#FF7A00]" />
                    Гиперпараметры LoRA (Low-Rank Adaptation)
                  </h3>
                  <span className="text-xs font-mono-code text-[#FF7A00]">Unsloth FastLoRA v2</span>
                </div>

                {/* LoRA Rank & Alpha */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* LoRA Rank Selector */}
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between text-xs">
                      <label className="text-white font-semibold">LoRA Rank ($r$)</label>
                      <span className="font-mono-code text-[#FF7A00] font-bold">r = {proParams.loraRank}</span>
                    </div>
                    <div className="grid grid-cols-5 gap-1.5">
                      {[4, 8, 16, 32, 64].map((r) => (
                        <button
                          key={r}
                          onClick={() => handleRankChange(r)}
                          className={`py-1.5 text-xs font-mono-code rounded-lg border transition-all ${
                            proParams.loraRank === r
                              ? 'bg-[#FF7A00] text-white font-bold border-[#FF7A00]'
                              : 'bg-[#181F2E] text-[#94A3B8] border-[#273248] hover:text-white'
                          }`}
                        >
                          {r}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* LoRA Alpha */}
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between text-xs">
                      <label className="text-white font-semibold">LoRA Alpha ($\alpha$)</label>
                      <div className="flex items-center gap-1.5">
                        <input
                          type="checkbox"
                          id="autoAlpha"
                          checked={alphaAutoCouple}
                          onChange={(e) => setAlphaAutoCouple(e.target.checked)}
                          className="accent-[#FF7A00] cursor-pointer"
                        />
                        <label htmlFor="autoAlpha" className="text-[10px] text-[#94A3B8] cursor-pointer">
                          2 × r (Авто)
                        </label>
                      </div>
                    </div>
                    <input
                      type="number"
                      value={proParams.loraAlpha}
                      disabled={alphaAutoCouple}
                      onChange={(e) => setProParams(prev => ({ ...prev, loraAlpha: parseInt(e.target.value) || 1 }))}
                      className="w-full bg-[#181F2E] border border-[#273248] rounded-lg px-3 py-1.5 text-xs text-white font-mono-code focus:outline-none focus:border-[#FF7A00] disabled:opacity-60"
                    />
                  </div>
                </div>

                {/* Target Modules */}
                <div className="flex flex-col gap-2.5">
                  <div className="flex justify-between items-center text-xs">
                    <label className="text-white font-semibold">Целевые модули (Target Modules)</label>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={selectAllLinear}
                        className="text-[10px] text-[#FF7A00] hover:underline font-mono-code"
                      >
                        Все линейные (7)
                      </button>
                      <span className="text-[#64748B]">|</span>
                      <button
                        onClick={selectAttentionOnly}
                        className="text-[10px] text-[#94A3B8] hover:text-white font-mono-code"
                      >
                        Только Attention (4)
                      </button>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    {['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].map((mod) => {
                      const active = proParams.targetModules.includes(mod);
                      return (
                        <div
                          key={mod}
                          onClick={() => toggleTargetModule(mod)}
                          className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg border text-xs font-mono-code cursor-pointer transition-all ${
                            active
                              ? 'bg-[#181F2E] border-[#FF7A00] text-white font-medium'
                              : 'bg-[#121722] border-[#1E2638] text-[#64748B]'
                          }`}
                        >
                          <div className={`w-3.5 h-3.5 rounded flex items-center justify-center text-[9px] ${
                            active ? 'bg-[#FF7A00] text-white' : 'border border-[#273248]'
                          }`}>
                            {active && '✓'}
                          </div>
                          <span>{mod}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Batch & Context Length */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t border-[#1E2638]">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs text-white font-semibold">Micro-Batch Size</label>
                    <select
                      value={proParams.microBatchSize}
                      onChange={(e) => setProParams(prev => ({ ...prev, microBatchSize: parseInt(e.target.value) }))}
                      className="bg-[#181F2E] border border-[#273248] rounded-lg px-2.5 py-1.5 text-xs text-white font-mono-code focus:outline-none focus:border-[#FF7A00]"
                    >
                      <option value={1}>1 (Safe 4GB)</option>
                      <option value={2}>2</option>
                      <option value={4}>4 (Cluster/8GB)</option>
                    </select>
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs text-white font-semibold">Grad Accumulation</label>
                    <select
                      value={proParams.gradientAccumulation}
                      onChange={(e) => setProParams(prev => ({ ...prev, gradientAccumulation: parseInt(e.target.value) }))}
                      className="bg-[#181F2E] border border-[#273248] rounded-lg px-2.5 py-1.5 text-xs text-white font-mono-code focus:outline-none focus:border-[#FF7A00]"
                    >
                      <option value={1}>1</option>
                      <option value={2}>2</option>
                      <option value={4}>4 (Эффект. batch 4)</option>
                      <option value={8}>8 (Эффект. batch 8)</option>
                      <option value={16}>16 (Эффект. batch 16)</option>
                    </select>
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs text-white font-semibold">Max Sequence Length</label>
                    <select
                      value={proParams.maxSeqLength}
                      onChange={(e) => setProParams(prev => ({ ...prev, maxSeqLength: parseInt(e.target.value) }))}
                      className="bg-[#181F2E] border border-[#273248] rounded-lg px-2.5 py-1.5 text-xs text-white font-mono-code focus:outline-none focus:border-[#FF7A00]"
                    >
                      <option value={512}>512 токенов</option>
                      <option value={1024}>1024 токенов (Реком.)</option>
                      <option value={2048}>2048 токенов</option>
                      <option value={4096}>4096 токенов (FlashAttn)</option>
                    </select>
                  </div>
                </div>

                {/* Sequence Packing & Optimization toggles */}
                <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-[#1E2638] text-xs">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={proParams.sequencePacking}
                      onChange={(e) => setProParams(prev => ({ ...prev, sequencePacking: e.target.checked }))}
                      className="accent-[#FF7A00]"
                    />
                    <span className="text-white font-medium">Sequence Packing (Unsloth 5x speedup)</span>
                  </label>

                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={proParams.gradientCheckpointing}
                      onChange={(e) => setProParams(prev => ({ ...prev, gradientCheckpointing: e.target.checked }))}
                      className="accent-[#FF7A00]"
                    />
                    <span className="text-white font-medium">Gradient Checkpointing (–60% VRAM)</span>
                  </label>
                </div>

              </div>

              {/* Optimizer & Learning Rate Schedule */}
              <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col gap-5">
                <div className="flex items-center justify-between border-b border-[#1E2638] pb-3">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <Gauge className="w-4 h-4 text-[#FF7A00]" />
                    Оптимизатор и расписание Learning Rate
                  </h3>
                  <span className="text-xs font-mono-code text-[#94A3B8]">
                    Эффект. Batch: <strong className="text-white">{effectiveBatchSize}</strong>
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs text-white font-semibold">Оптимизатор</label>
                    <select
                      value={proParams.optimizer}
                      onChange={(e) => setProParams(prev => ({ ...prev, optimizer: e.target.value as any }))}
                      className="bg-[#181F2E] border border-[#273248] rounded-lg px-2.5 py-1.5 text-xs text-white font-mono-code focus:outline-none focus:border-[#FF7A00]"
                    >
                      <option value="adamw_8bit">AdamW 8-bit (Экономия 75% памяти)</option>
                      <option value="adamw_fp32">AdamW FP32 (Классический)</option>
                    </select>
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs text-white font-semibold">Learning Rate</label>
                    <input
                      type="text"
                      value={proParams.learningRate}
                      onChange={(e) => setProParams(prev => ({ ...prev, learningRate: parseFloat(e.target.value) || 0.0002 }))}
                      className="bg-[#181F2E] border border-[#273248] rounded-lg px-2.5 py-1.5 text-xs text-white font-mono-code focus:outline-none focus:border-[#FF7A00]"
                    />
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs text-white font-semibold">LR Schedule</label>
                    <select
                      value={proParams.lrSchedule}
                      onChange={(e) => setProParams(prev => ({ ...prev, lrSchedule: e.target.value as any }))}
                      className="bg-[#181F2E] border border-[#273248] rounded-lg px-2.5 py-1.5 text-xs text-white font-mono-code focus:outline-none focus:border-[#FF7A00]"
                    >
                      <option value="cosine">Cosine Annealing</option>
                      <option value="linear">Linear Decay</option>
                      <option value="constant">Constant</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-[#1E2638]">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-[#94A3B8]">Warmup Ratio:</span>
                    <span className="font-mono-code text-white">{proParams.warmupRatio * 100}% первых шагов</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-[#94A3B8]">Weight Decay:</span>
                    <span className="font-mono-code text-white">{proParams.weightDecay}</span>
                  </div>
                </div>

              </div>

            </div>
          )}

        </div>

        {/* Right Column: Real-Time Telemetry, Loss Curve & Actions (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          
          {/* Action Control Panel (Pill Buttons) */}
          <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col gap-4 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                Управление обучением
              </span>
              <span className={`text-[11px] font-mono-code px-2 py-0.5 rounded-full border ${
                telemetry.status === 'training'
                  ? 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/30 font-semibold'
                  : telemetry.status === 'paused'
                  ? 'bg-[#F59E0B]/15 text-[#F59E0B] border-[#F59E0B]/30'
                  : 'bg-[#64748B]/15 text-[#94A3B8] border-[#64748B]/30'
              }`}>
                {telemetry.status === 'training' && '● ВЫПОЛНЯЕТСЯ'}
                {telemetry.status === 'paused' && '⏸ ПАУЗА'}
                {telemetry.status === 'idle' && 'ГОТОВ К ЗАПУСКУ'}
                {telemetry.status === 'completed' && '✓ ЗАВЕРШЕНО'}
              </span>
            </div>

            {/* Pill Buttons Row */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => telemetryService.startTraining()}
                disabled={telemetry.status === 'training'}
                className="btn-pill-primary flex-1 gap-2 py-2.5"
              >
                <Play className="w-4 h-4 fill-white" />
                <span>Запустить обучение</span>
              </button>

              <button
                onClick={() => telemetryService.pauseTraining()}
                disabled={telemetry.status !== 'training'}
                className="btn-pill-secondary gap-1.5 py-2.5 px-4"
              >
                <Pause className="w-4 h-4" />
                <span>Пауза</span>
              </button>

              <button
                onClick={() => telemetryService.stopTraining()}
                disabled={telemetry.status === 'idle'}
                className="btn-pill-danger gap-1.5 py-2.5 px-4"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>Прервать</span>
              </button>
            </div>
          </div>

          {/* Real-time Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {/* Metric 1: Loss */}
            <div className="bg-[#121722] border border-[#1E2638] rounded-xl p-3.5 flex flex-col gap-1">
              <span className="text-[10px] text-[#94A3B8] font-mono-code uppercase tracking-wider">
                Текущий Loss
              </span>
              <div className="text-xl font-bold font-mono-code text-[#FF7A00]">
                {telemetry.loss.toFixed(4)}
              </div>
              <span className="text-[10px] text-[#10B981] font-mono-code">
                ↓ -0.018 за 10 шагов
              </span>
            </div>

            {/* Metric 2: Step Progress */}
            <div className="bg-[#121722] border border-[#1E2638] rounded-xl p-3.5 flex flex-col gap-1">
              <span className="text-[10px] text-[#94A3B8] font-mono-code uppercase tracking-wider">
                Шаг / Всего
              </span>
              <div className="text-lg font-bold font-mono-code text-white">
                {telemetry.step} <span className="text-xs text-[#64748B] font-normal">/ {telemetry.totalSteps}</span>
              </div>
              <span className="text-[10px] text-[#94A3B8] font-mono-code">
                {((telemetry.step / telemetry.totalSteps) * 100).toFixed(1)}% завершено
              </span>
            </div>

            {/* Metric 3: Tokens / sec */}
            <div className="bg-[#121722] border border-[#1E2638] rounded-xl p-3.5 flex flex-col gap-1">
              <span className="text-[10px] text-[#94A3B8] font-mono-code uppercase tracking-wider">
                Скорость (tok/s)
              </span>
              <div className="text-lg font-bold font-mono-code text-white">
                {telemetry.tokensPerSec.toLocaleString()}
              </div>
              <span className="text-[10px] text-[#FF7A00] font-mono-code">
                {telemetry.clusterMode !== 'local' ? '2-Node Dual GPU' : 'Vulkan SPIR-V'}
              </span>
            </div>

            {/* Metric 4: Learning Rate */}
            <div className="bg-[#121722] border border-[#1E2638] rounded-xl p-3.5 flex flex-col gap-1">
              <span className="text-[10px] text-[#94A3B8] font-mono-code uppercase tracking-wider">
                Learning Rate
              </span>
              <div className="text-base font-bold font-mono-code text-white">
                {telemetry.learningRate.toExponential(2)}
              </div>
              <span className="text-[10px] text-[#94A3B8] font-mono-code">
                Cosine schedule
              </span>
            </div>

            {/* Metric 5: Elapsed / ETA */}
            <div className="bg-[#121722] border border-[#1E2638] rounded-xl p-3.5 flex flex-col gap-1">
              <span className="text-[10px] text-[#94A3B8] font-mono-code uppercase tracking-wider">
                Прошло / ETA
              </span>
              <div className="text-base font-bold font-mono-code text-white">
                {formatSeconds(telemetry.elapsedSeconds)}
                <span className="text-xs text-[#64748B] font-normal"> / ~{formatSeconds(telemetry.etaSeconds)}</span>
              </div>
              <span className="text-[10px] text-[#94A3B8] font-mono-code">
                Осталось ~{Math.ceil(telemetry.etaSeconds / 60)} мин.
              </span>
            </div>

            {/* Metric 6: GPU Temp & Power */}
            <div className="bg-[#121722] border border-[#1E2638] rounded-xl p-3.5 flex flex-col gap-1">
              <span className="text-[10px] text-[#94A3B8] font-mono-code uppercase tracking-wider">
                GPU Темп. / Питание
              </span>
              <div className="text-base font-bold font-mono-code text-white flex items-center gap-1.5">
                <span>{telemetry.gpuTempC}°C</span>
                <span className="text-xs text-[#64748B]">/ {telemetry.gpuPowerW}W</span>
              </div>
              <span className="text-[10px] text-[#10B981] font-mono-code">
                Штатный режим (RADV)
              </span>
            </div>
          </div>

          {/* HTML5 Canvas Real-Time Loss Curve */}
          <LossChart telemetry={telemetry} />

          {/* Real-time Engineering Log Terminal (NO AI-slop) */}
          <div className="bg-[#0B0E14] border border-[#1E2638] rounded-xl overflow-hidden flex flex-col">
            <div className="px-3.5 py-2 bg-[#121722] border-b border-[#1E2638] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Terminal className="w-3.5 h-3.5 text-[#94A3B8]" />
                <span className="text-[11px] font-mono-code text-[#94A3B8] uppercase">
                  Vulkan Execution Logs
                </span>
              </div>
              <span className="text-[10px] font-mono-code text-[#64748B]">
                radv-polaris10-compute
              </span>
            </div>
            <div className="p-3 font-mono-code text-[11px] text-[#94A3B8] h-36 overflow-y-auto flex flex-col gap-1 leading-relaxed">
              {logs.slice(-8).map((line, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <span className="text-[#64748B] select-none">&gt;</span>
                  <span className={line.includes('loss=') ? 'text-[#FF7A00]' : line.includes('ALLREDUCE') ? 'text-[#10B981]' : 'text-[#94A3B8]'}>
                    {line}
                  </span>
                </div>
              ))}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
};
