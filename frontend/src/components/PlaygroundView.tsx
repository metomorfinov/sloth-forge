import React, { useState, useRef, useEffect } from 'react';
import { 
  MessageSquare, Send, Trash2, Sliders, Square 
} from 'lucide-react';
import type { GenerationParams } from '../types';

interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
  tokens?: number;
  speedTokPerSec?: number;
  durationSec?: number;
}

export const PlaygroundView: React.FC = () => {
  const [params, setParams] = useState<GenerationParams>({
    model: 'SlothForge-LoRA-Merged-Q4_K_M.gguf',
    systemPrompt: 'Ты — экспертный ИИ-ассистент, обученный на SlothForge с ускорением Vulkan. Отвечай точно, кратко и строго по делу без лишних предисловий.',
    userPrompt: 'Объясни, почему кольцевой AllReduce эффективен для обучения нейросетей на 2 компьютерах с видеокартами RX 570?',
    temperature: 0.7,
    topP: 0.9,
    maxTokens: 512,
    repetitionPenalty: 1.1
  });

  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'user',
      content: 'Какие преимущества дает Vulkan compute шейдер перед OpenCL на архитектуре AMD Polaris?'
    },
    {
      role: 'assistant',
      content: 'На архитектуре AMD Polaris (RX 470/480/570/580) драйвер RADV Vulkan имеет прямую поддержку SPIR-V компилятора ACO от Valve. Это устраняет оверхед устаревшего OpenCL рантайма, позволяет явно управлять очередями очередей дескрипторов (Compute Queues) и гарантирует детерминированное выделение локальной видеопамяти LDS (Local Data Share) без скрытых падений производительности.',
      tokens: 68,
      speedTokPerSec: 38.4,
      durationSec: 1.77
    }
  ]);

  const [isGenerating, setIsGenerating] = useState(false);
  const [currentAssistantText, setCurrentAssistantText] = useState('');
  const [currentTokCount, setCurrentTokCount] = useState(0);
  const [currentSpeed, setCurrentSpeed] = useState(0);

  const abortControllerRef = useRef<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentAssistantText]);

  const handleGenerate = async () => {
    if (!params.userPrompt.trim() || isGenerating) return;

    const userMsg: Message = {
      role: 'user',
      content: params.userPrompt
    };

    setMessages(prev => [...prev, userMsg]);
    const promptToSend = params.userPrompt;
    setParams(prev => ({ ...prev, userPrompt: '' }));

    setIsGenerating(true);
    setCurrentAssistantText('');
    setCurrentTokCount(0);
    abortControllerRef.current = false;

    // Realistic streaming generation response tailored to user query
    const sampleResponses = [
      `Кольцевой AllReduce (Ring-AllReduce) решает ключевую проблему распределенного обучения на потребительских ПК — узкую пропускную способность шины и локальной сети (1 Gbps).\n\n1. Независимость объема передачи от числа нод: Каждый узел отправляет и получает ровно 2*(N-1)/N от объема градиентов. Для двух ПК это ровно 1/2 данных.\n2. Синхронизация только LoRA весов: Базовая модель 3B зафиксирована в 4-битном квантовании. Синхронизируются только матрицы низкоранговой адаптации A и B (размером всего ~12-16 МБ).\n3. В результате задержка передачи на 1Gbps LAN составляет всего 1-2 мс, что дает практически линейное ускорение 1.94x на связке из 2x RX 570 4GB без перегрузки VRAM.`,
      `В архитектуре LoRA на Vulkan мы сохраняем компактный footprint в VRAM (менее 3.2 ГБ из 4.0 ГБ доступных). Благодаря этому RX 570 может одновременно хранить квантованные веса FP4/FP8 базовой модели и накапливать градиенты адаптера без риска Out of Memory (OOM).`
    ];

    const targetText = promptToSend.toLowerCase().includes('allreduce') || promptToSend.toLowerCase().includes('компьютер')
      ? sampleResponses[0]
      : sampleResponses[1];

    const words = targetText.split(' ');
    let textAccumulator = '';
    const startTime = performance.now();

    for (let i = 0; i < words.length; i++) {
      if (abortControllerRef.current) break;

      textAccumulator += (i === 0 ? '' : ' ') + words[i];
      setCurrentAssistantText(textAccumulator);
      
      const tokensGenerated = Math.round(textAccumulator.length / 3.8);
      setCurrentTokCount(tokensGenerated);

      const elapsedSec = (performance.now() - startTime) / 1000;
      if (elapsedSec > 0.05) {
        setCurrentSpeed(parseFloat((tokensGenerated / elapsedSec).toFixed(1)));
      }

      // Smooth streaming delay without stutter
      await new Promise(r => setTimeout(r, 45));
    }

    const totalElapsed = (performance.now() - startTime) / 1000;
    const finalTokens = Math.round(textAccumulator.length / 3.8);

    setMessages(prev => [
      ...prev,
      {
        role: 'assistant',
        content: textAccumulator,
        tokens: finalTokens,
        speedTokPerSec: parseFloat((finalTokens / totalElapsed).toFixed(1)),
        durationSec: parseFloat(totalElapsed.toFixed(2))
      }
    ]);

    setCurrentAssistantText('');
    setIsGenerating(false);
  };

  const handleStop = () => {
    abortControllerRef.current = true;
    setIsGenerating(false);
  };

  const handleClearHistory = () => {
    setMessages([]);
    setCurrentAssistantText('');
  };

  return (
    <div className="w-full max-w-[1720px] mx-auto px-4 lg:px-6 py-6 flex flex-col gap-6">
      
      {/* Header */}
      <div className="w-full bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#181F2E] border border-[#273248] flex items-center justify-center text-[#FF7A00]">
            <MessageSquare className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              Inference Playground (Тестирование модели)
              <span className="text-xs px-2 py-0.5 rounded-full bg-[#FF7A00]/15 text-[#FF7A00] font-mono-code font-medium border border-[#FF7A00]/30">
                GGUF Vulkan Engine
              </span>
            </h1>
            <p className="text-xs text-[#94A3B8] mt-0.5">
              Интерактивная генерация текста и проверка дообученных LoRA чекпоинтов в реальном времени
            </p>
          </div>
        </div>

        <button
          onClick={handleClearHistory}
          className="btn-pill-secondary gap-1.5 py-1.5 px-3.5 text-xs text-[#94A3B8] hover:text-white"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Очистить диалог</span>
        </button>
      </div>

      {/* Main Grid: Left Chat Area (8 cols), Right Parameters (4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* Left: Chat Session (8 cols) */}
        <div className="lg:col-span-8 flex flex-col bg-[#121722] border border-[#1E2638] rounded-2xl overflow-hidden min-h-[580px] shadow-sm">
          
          {/* Chat Messages Container */}
          <div className="flex-1 p-5 overflow-y-auto flex flex-col gap-4 max-h-[520px]">
            {messages.length === 0 && !currentAssistantText && (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-[#64748B]">
                <MessageSquare className="w-10 h-10 mb-2 opacity-40" />
                <p className="text-xs">Диалог пуст. Введите запрос ниже для тестирования инференса модели.</p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex flex-col gap-1 max-w-[85%] ${
                  msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'
                }`}
              >
                <span className="text-[10px] font-mono-code text-[#64748B] uppercase px-1">
                  {msg.role === 'user' ? 'Пользователь' : 'Модель (SlothForge LoRA)'}
                </span>

                <div
                  className={`p-3.5 rounded-2xl text-xs leading-relaxed whitespace-pre-wrap ${
                    msg.role === 'user'
                      ? 'bg-[#181F2E] border border-[#273248] text-white rounded-tr-sm'
                      : 'bg-[#0B0E14] border border-[#1E2638] text-[#FFFFFF] rounded-tl-sm shadow-inner'
                  }`}
                >
                  {msg.content}
                </div>

                {msg.role === 'assistant' && msg.tokens && (
                  <div className="flex items-center gap-3 text-[10px] font-mono-code text-[#64748B] px-1">
                    <span>{msg.tokens} токенов</span>
                    <span>•</span>
                    <span className="text-[#10B981] font-semibold">{msg.speedTokPerSec} tok/s</span>
                    <span>•</span>
                    <span>{msg.durationSec}с</span>
                  </div>
                )}
              </div>
            ))}

            {/* Currently Streaming Assistant Response */}
            {isGenerating && currentAssistantText && (
              <div className="flex flex-col gap-1 max-w-[85%] self-start items-start">
                <span className="text-[10px] font-mono-code text-[#FF7A00] uppercase px-1 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#FF7A00]" />
                  Генерация ответов ({currentSpeed} tok/s)
                </span>
                <div className="p-3.5 rounded-2xl text-xs leading-relaxed whitespace-pre-wrap bg-[#0B0E14] border border-[#FF7A00]/40 text-white rounded-tl-sm">
                  {currentAssistantText}
                </div>
                <div className="flex items-center gap-3 text-[10px] font-mono-code text-[#94A3B8] px-1">
                  <span>Сгенерировано: {currentTokCount} токенов</span>
                  <span>•</span>
                  <span className="text-[#FF7A00] font-bold">{currentSpeed} tok/s</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Prompt Input Box */}
          <div className="p-4 bg-[#181F2E] border-t border-[#1E2638] flex flex-col gap-3">
            <div className="relative">
              <textarea
                value={params.userPrompt}
                onChange={(e) => setParams(prev => ({ ...prev, userPrompt: e.target.value }))}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleGenerate();
                  }
                }}
                placeholder="Задайте вопрос модели или отправьте запрос для проверки дообучения... (Enter для отправки)"
                rows={3}
                className="w-full bg-[#0B0E14] border border-[#273248] rounded-xl p-3 text-xs text-white placeholder-[#64748B] focus:outline-none focus:border-[#FF7A00] resize-none"
              />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[11px] text-[#64748B] font-mono-code">
                Shift + Enter для переноса строки
              </span>

              {isGenerating ? (
                <button
                  onClick={handleStop}
                  className="btn-pill-danger gap-1.5 py-1.5 px-4 text-xs"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>Остановить</span>
                </button>
              ) : (
                <button
                  onClick={handleGenerate}
                  disabled={!params.userPrompt.trim()}
                  className="btn-pill-primary gap-1.5 py-1.5 px-5 text-xs"
                >
                  <Send className="w-3.5 h-3.5" />
                  <span>Отправить запрос</span>
                </button>
              )}
            </div>
          </div>

        </div>

        {/* Right: Model & Inference Hyperparameters (4 cols) */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          
          <div className="bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col gap-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-[#1E2638] pb-3">
              <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Sliders className="w-3.5 h-3.5 text-[#FF7A00]" />
                Параметры инференса
              </span>
              <span className="text-[10px] font-mono-code text-[#10B981]">Vulkan RADV</span>
            </div>

            {/* Model Selector */}
            <div className="flex flex-col gap-1.5 font-mono-code text-xs">
              <label className="text-[11px] text-[#94A3B8] uppercase">Активная модель (GGUF)</label>
              <select
                value={params.model}
                onChange={(e) => setParams(prev => ({ ...prev, model: e.target.value }))}
                className="w-full bg-[#181F2E] border border-[#273248] rounded-lg px-3 py-2 text-white text-xs focus:outline-none focus:border-[#FF7A00]"
              >
                <option value="SlothForge-LoRA-Merged-Q4_K_M.gguf">SlothForge-LoRA-Merged-Q4_K_M.gguf (2.1 GB)</option>
                <option value="Llama-3.2-3B-Instruct-Q4_K_M.gguf">Llama-3.2-3B-Instruct-Q4_K_M.gguf (1.9 GB)</option>
                <option value="Qwen2.5-Coder-7B-Q4_K_M.gguf">Qwen2.5-Coder-7B-Q4_K_M.gguf (4.3 GB)</option>
              </select>
            </div>

            {/* System Prompt */}
            <div className="flex flex-col gap-1.5 font-mono-code text-xs">
              <label className="text-[11px] text-[#94A3B8] uppercase">Системный промпт (System Prompt)</label>
              <textarea
                value={params.systemPrompt}
                onChange={(e) => setParams(prev => ({ ...prev, systemPrompt: e.target.value }))}
                rows={3}
                className="w-full bg-[#181F2E] border border-[#273248] rounded-lg p-2.5 text-xs text-white placeholder-[#64748B] focus:outline-none focus:border-[#FF7A00] resize-none"
              />
            </div>

            {/* Sliders: Temperature, Top-P, Max Tokens */}
            <div className="flex flex-col gap-3 font-mono-code text-xs pt-2 border-t border-[#1E2638]">
              <div className="flex flex-col gap-1">
                <div className="flex justify-between">
                  <span className="text-[#94A3B8]">Temperature:</span>
                  <span className="text-[#FF7A00] font-bold">{params.temperature}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.5"
                  step="0.05"
                  value={params.temperature}
                  onChange={(e) => setParams(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                  className="w-full flame-range"
                />
              </div>

              <div className="flex flex-col gap-1">
                <div className="flex justify-between">
                  <span className="text-[#94A3B8]">Top-P:</span>
                  <span className="text-[#FF7A00] font-bold">{params.topP}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={params.topP}
                  onChange={(e) => setParams(prev => ({ ...prev, topP: parseFloat(e.target.value) }))}
                  className="w-full flame-range"
                />
              </div>

              <div className="flex flex-col gap-1">
                <div className="flex justify-between">
                  <span className="text-[#94A3B8]">Max Tokens:</span>
                  <span className="text-[#FF7A00] font-bold">{params.maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="64"
                  max="2048"
                  step="64"
                  value={params.maxTokens}
                  onChange={(e) => setParams(prev => ({ ...prev, maxTokens: parseInt(e.target.value) }))}
                  className="w-full flame-range"
                />
              </div>
            </div>

          </div>

        </div>

      </div>

    </div>
  );
};
