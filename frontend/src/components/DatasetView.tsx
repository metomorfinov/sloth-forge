import React, { useState } from 'react';
import { 
  Database, UploadCloud, FileText, CheckCircle2, 
  Search, BarChart3, ChevronLeft, ChevronRight 
} from 'lucide-react';
import type { DatasetEntry } from '../types';

const SAMPLE_ENTRIES: DatasetEntry[] = [
  {
    id: 1,
    instruction: 'Напиши функцию на Python для вычисления скользящего среднего ряда чисел.',
    input: 'numbers = [10, 20, 30, 40, 50], window = 3',
    output: 'def moving_average(numbers, window):\n    import numpy as np\n    return np.convolve(numbers, np.ones(window)/window, mode="valid").tolist()',
    tokenCount: 78
  },
  {
    id: 2,
    instruction: 'Объясни назначение градиентного накопления (Gradient Accumulation) при нехватке VRAM.',
    input: '',
    output: 'Gradient Accumulation позволяет эмулировать большой размер батча (batch size), вычисляя градиенты для небольших микро-батчей и суммируя их без обновления весов. Шаг оптимизатора выполняется только после накопления заданного числа шагов.',
    tokenCount: 142
  },
  {
    id: 3,
    instruction: 'Сформируй корректный конфиг LoRA для обучения модели Llama-3 8B на 16-битном оптимизаторе.',
    input: '',
    output: '{\n  "r": 16,\n  "lora_alpha": 32,\n  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],\n  "lora_dropout": 0.05,\n  "bias": "none"\n}',
    tokenCount: 95
  },
  {
    id: 4,
    instruction: 'Переведи технический термин «Zero-Redundancy Optimizer» на русский язык и дай краткую справку.',
    input: '',
    output: '«Оптимизатор с нулевой избыточностью» (ZeRO). Технология распределения состояний оптимизатора, градиентов и параметров модели между несколькими узлами для устранения дублирования данных в памяти GPU.',
    tokenCount: 88
  },
  {
    id: 5,
    instruction: 'Как рассчитать необходимый объем VRAM для дообучения 3B модели с помощью LoRA?',
    input: 'Модель: Llama 3.2 3B, LoRA r=16, seq_len=1024',
    output: 'Веса 4-бит: ~1.8 ГБ. Градиенты и состояние оптимизатора LoRA (r=16): ~0.4 ГБ. Активации при seq_len=1024: ~0.8 ГБ. Итого пиковое потребление: ~3.0-3.2 ГБ VRAM, что безопасно укладывается в лимит 4 ГБ.',
    tokenCount: 164
  }
];

export const DatasetView: React.FC = () => {
  const [format, setFormat] = useState<'alpaca' | 'chatml' | 'sharegpt'>('alpaca');
  const [entries] = useState<DatasetEntry[]>(SAMPLE_ENTRIES);
  const [searchQuery, setSearchQuery] = useState('');
  const [fileName, setFileName] = useState('slothforge_instructions_ru_train.jsonl');
  const [isDragging, setIsDragging] = useState(false);

  const filteredEntries = entries.filter(e =>
    e.instruction.toLowerCase().includes(searchQuery.toLowerCase()) ||
    e.output.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Distribution buckets (0-64, 65-128, 129-256, 257-512, 513+)
  const buckets = [
    { label: '< 64 tok', count: 420, percent: 18 },
    { label: '64–128 tok', count: 1140, percent: 46 },
    { label: '129–256 tok', count: 680, percent: 27 },
    { label: '257–512 tok', count: 180, percent: 7 },
    { label: '512+ tok', count: 30, percent: 2 },
  ];

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFileName(e.dataTransfer.files[0].name);
    }
  };

  return (
    <div className="w-full max-w-[1720px] mx-auto px-4 lg:px-6 py-6 flex flex-col gap-6">
      
      {/* Header */}
      <div className="w-full bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#181F2E] border border-[#273248] flex items-center justify-center text-[#FF7A00]">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              Рецепты датасетов (Dataset Recipes)
              <span className="text-xs px-2 py-0.5 rounded-full bg-[#10B981]/15 text-[#10B981] font-mono-code font-medium border border-[#10B981]/30">
                JSONL / CSV / ShareGPT
              </span>
            </h1>
            <p className="text-xs text-[#94A3B8] mt-0.5">
              Подготовка, форматирование и расчет длины токенов перед запуском обучения
            </p>
          </div>
        </div>

        {/* Format Selector Pills */}
        <div className="flex items-center bg-[#0B0E14] p-1 rounded-full border border-[#1E2638] select-none">
          {(['alpaca', 'chatml', 'sharegpt'] as const).map((fmt) => (
            <button
              key={fmt}
              onClick={() => setFormat(fmt)}
              className={`px-3.5 py-1.5 rounded-full text-xs font-mono-code transition-all ${
                format === fmt
                  ? 'bg-[#181F2E] text-white border border-[#273248] font-bold'
                  : 'text-[#94A3B8] hover:text-white border border-transparent'
              }`}
            >
              {fmt.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Top Grid: Upload Area & Token Distribution (2 columns) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Upload Zone (6 cols) */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`lg:col-span-6 bg-[#121722] border-2 border-dashed rounded-2xl p-6 flex flex-col items-center justify-center text-center gap-3 transition-all cursor-pointer ${
            isDragging
              ? 'border-[#FF7A00] bg-[#181F2E]'
              : 'border-[#1E2638] hover:border-[#273248] hover:bg-[#181F2E]/40'
          }`}
        >
          <div className="w-12 h-12 rounded-full bg-[#181F2E] border border-[#273248] flex items-center justify-center text-[#FF7A00]">
            <UploadCloud className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white">
              Перетащите файл датасета сюда или нажмите для выбора
            </h3>
            <p className="text-xs text-[#94A3B8] mt-1">
              Поддерживаются форматы: .jsonl, .json, .csv, .parquet, .txt
            </p>
          </div>

          <div className="flex items-center gap-3 mt-2">
            <span className="px-3 py-1 rounded-lg bg-[#0B0E14] border border-[#1E2638] text-xs font-mono-code text-white flex items-center gap-2">
              <FileText className="w-3.5 h-3.5 text-[#FF7A00]" />
              {fileName}
            </span>
            <span className="text-[11px] font-mono-code text-[#10B981] flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              2,450 валидных сэмплов
            </span>
          </div>
        </div>

        {/* Token Length Distribution (6 cols) */}
        <div className="lg:col-span-6 bg-[#121722] border border-[#1E2638] rounded-2xl p-5 flex flex-col justify-between gap-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-[#1E2638] pb-3">
            <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-[#FF7A00]" />
              Распределение длины токенов (Token Distribution)
            </span>
            <span className="text-xs font-mono-code text-[#94A3B8]">
              Медиана: <strong className="text-white">112 токенов</strong>
            </span>
          </div>

          <div className="flex flex-col gap-2.5">
            {buckets.map((b) => (
              <div key={b.label} className="flex items-center gap-3 font-mono-code text-xs">
                <span className="w-24 text-[#94A3B8] text-[11px]">{b.label}</span>
                <div className="flex-1 bg-[#181F2E] h-3.5 rounded-full overflow-hidden border border-[#1E2638]">
                  <div
                    className="h-full bg-gradient-to-r from-[#FF7A00] to-[#FF8F26] rounded-full"
                    style={{ width: `${b.percent}%` }}
                  />
                </div>
                <span className="w-16 text-right text-white font-semibold text-[11px]">
                  {b.count} ({b.percent}%)
                </span>
              </div>
            ))}
          </div>

          <div className="pt-2 border-t border-[#1E2638] flex items-center justify-between text-[11px] font-mono-code text-[#64748B]">
            <span>Оптимальный Max Seq Length: 1024 токенов</span>
            <span className="text-[#10B981]">Потери при усечении: 0.0%</span>
          </div>
        </div>

      </div>

      {/* Dataset Preview Table */}
      <div className="w-full bg-[#121722] border border-[#1E2638] rounded-2xl overflow-hidden shadow-sm flex flex-col">
        
        {/* Table Controls Bar */}
        <div className="p-4 bg-[#181F2E] border-b border-[#1E2638] flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <span className="text-xs font-bold text-white uppercase tracking-wider">
              Предпросмотр сэмплов датасета
            </span>
            <span className="text-xs font-mono-code text-[#94A3B8]">
              Показано 5 из 2,450
            </span>
          </div>

          <div className="relative">
            <Search className="w-3.5 h-3.5 text-[#94A3B8] absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Поиск по сэмплам..."
              className="bg-[#0B0E14] border border-[#273248] rounded-full pl-8 pr-3 py-1.5 text-xs text-white placeholder-[#64748B] focus:outline-none focus:border-[#FF7A00]"
            />
          </div>
        </div>

        {/* Table Content */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-[#1E2638] bg-[#121722] text-[#94A3B8] font-mono-code text-[11px] uppercase">
                <th className="py-3 px-4 w-12">#</th>
                <th className="py-3 px-4 w-2/5">Инструкция (Instruction)</th>
                <th className="py-3 px-4 w-2/5">Ожидаемый ответ (Output)</th>
                <th className="py-3 px-4 w-28 text-right">Токены</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E2638]">
              {filteredEntries.map((row) => (
                <tr key={row.id} className="hover:bg-[#181F2E]/50 transition-colors">
                  <td className="py-3 px-4 font-mono-code text-[#64748B]">
                    {row.id}
                  </td>
                  <td className="py-3 px-4 text-white font-medium">
                    {row.instruction}
                    {row.input && (
                      <div className="mt-1 text-[11px] font-mono-code text-[#94A3B8] bg-[#0B0E14] p-1.5 rounded border border-[#1E2638]">
                        Вход: {row.input}
                      </div>
                    )}
                  </td>
                  <td className="py-3 px-4 text-[#94A3B8] font-mono-code text-[11px] leading-relaxed whitespace-pre-wrap">
                    {row.output}
                  </td>
                  <td className="py-3 px-4 text-right font-mono-code text-white">
                    <span className="px-2 py-0.5 rounded bg-[#181F2E] border border-[#273248]">
                      {row.tokenCount}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Table Footer Pagination */}
        <div className="p-3 bg-[#121722] border-t border-[#1E2638] flex items-center justify-between text-xs font-mono-code text-[#94A3B8]">
          <span>Страница 1 из 490</span>
          <div className="flex items-center gap-1">
            <button className="p-1 rounded hover:bg-[#181F2E] border border-[#1E2638] disabled:opacity-40">
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button className="p-1 rounded hover:bg-[#181F2E] border border-[#1E2638]">
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>

      </div>

    </div>
  );
};
