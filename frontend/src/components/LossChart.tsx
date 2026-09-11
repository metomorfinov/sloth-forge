import React, { useRef, useEffect, useState } from 'react';
import type { TelemetryData } from '../types';

interface LossChartProps {
  telemetry: TelemetryData;
}

export const LossChart: React.FC<LossChartProps> = ({ telemetry }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<{ step: number; loss: number; x: number; y: number } | null>(null);

  const history = telemetry.lossHistory;

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI
    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Padding for axes
    const padLeft = 45;
    const padRight = 20;
    const padTop = 20;
    const padBottom = 30;

    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Clear background
    ctx.fillStyle = '#121722';
    ctx.fillRect(0, 0, width, height);

    if (history.length < 2) {
      ctx.fillStyle = '#94A3B8';
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Ожидание данных телеметрии обучения...', width / 2, height / 2);
      return;
    }

    // Determine scale bounds
    let minLoss = Infinity;
    let maxLoss = -Infinity;
    const maxStep = Math.max(telemetry.totalSteps, 100);

    for (const pt of history) {
      if (pt.loss < minLoss) minLoss = pt.loss;
      if (pt.loss > maxLoss) maxLoss = pt.loss;
    }

    // Give some breathing room on Y-axis
    const lossMargin = (maxLoss - minLoss) * 0.1 || 0.2;
    const yMin = Math.max(0, minLoss - lossMargin);
    const yMax = maxLoss + lossMargin;

    // Helper functions for coordinates
    const getX = (step: number) => padLeft + (step / maxStep) * plotW;
    const getY = (loss: number) => padTop + plotH - ((loss - yMin) / (yMax - yMin)) * plotH;

    // Draw grid lines
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#1E2638';
    ctx.fillStyle = '#64748B';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';

    // Horizontal Y grid lines (4 lines)
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const val = yMin + (i / yTicks) * (yMax - yMin);
      const yPos = getY(val);

      ctx.beginPath();
      ctx.moveTo(padLeft, yPos);
      ctx.lineTo(width - padRight, yPos);
      ctx.stroke();

      ctx.fillText(val.toFixed(2), padLeft - 8, yPos + 3);
    }

    // Vertical X grid lines (5 lines)
    ctx.textAlign = 'center';
    const xTicks = 5;
    for (let i = 0; i <= xTicks; i++) {
      const stepVal = Math.round((i / xTicks) * maxStep);
      const xPos = getX(stepVal);

      ctx.beginPath();
      ctx.moveTo(xPos, padTop);
      ctx.lineTo(xPos, height - padBottom);
      ctx.stroke();

      ctx.fillText(stepVal.toString(), xPos, height - padBottom + 16);
    }

    // Draw subtle gradient area under curve
    const gradient = ctx.createLinearGradient(0, padTop, 0, height - padBottom);
    gradient.addColorStop(0, 'rgba(255, 122, 0, 0.25)');
    gradient.addColorStop(1, 'rgba(255, 122, 0, 0.00)');

    ctx.beginPath();
    ctx.moveTo(getX(history[0].step), getY(history[0].loss));
    for (let i = 1; i < history.length; i++) {
      ctx.lineTo(getX(history[i].step), getY(history[i].loss));
    }
    ctx.lineTo(getX(history[history.length - 1].step), height - padBottom);
    ctx.lineTo(getX(history[0].step), height - padBottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw smooth amber line
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#FF7A00';
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    ctx.moveTo(getX(history[0].step), getY(history[0].loss));
    for (let i = 1; i < history.length; i++) {
      ctx.lineTo(getX(history[i].step), getY(history[i].loss));
    }
    ctx.stroke();

    // Draw current head dot
    const latest = history[history.length - 1];
    const latestX = getX(latest.step);
    const latestY = getY(latest.loss);

    ctx.beginPath();
    ctx.arc(latestX, latestY, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = '#FF8F26';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#FFFFFF';
    ctx.stroke();

    // If hovered, draw crosshair
    if (hoveredPoint) {
      ctx.strokeStyle = '#94A3B8';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);

      ctx.beginPath();
      ctx.moveTo(hoveredPoint.x, padTop);
      ctx.lineTo(hoveredPoint.x, height - padBottom);
      ctx.moveTo(padLeft, hoveredPoint.y);
      ctx.lineTo(width - padRight, hoveredPoint.y);
      ctx.stroke();

      ctx.setLineDash([]); // Reset line dash

      ctx.beginPath();
      ctx.arc(hoveredPoint.x, hoveredPoint.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();
    }
  }, [history, telemetry.totalSteps, hoveredPoint]);

  // Mouse hover event for interactive crosshair
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || history.length < 2) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;

    const padLeft = 45;
    const padRight = 20;
    const plotW = rect.width - padLeft - padRight;
    const maxStep = Math.max(telemetry.totalSteps, 100);

    // Calculate closest step
    const targetStep = Math.round(((mouseX - padLeft) / plotW) * maxStep);
    const closest = history.reduce((prev, curr) =>
      Math.abs(curr.step - targetStep) < Math.abs(prev.step - targetStep) ? curr : prev
    );

    if (closest) {
      let minLoss = Infinity;
      let maxLoss = -Infinity;
      for (const pt of history) {
        if (pt.loss < minLoss) minLoss = pt.loss;
        if (pt.loss > maxLoss) maxLoss = pt.loss;
      }
      const lossMargin = (maxLoss - minLoss) * 0.1 || 0.2;
      const yMin = Math.max(0, minLoss - lossMargin);
      const yMax = maxLoss + lossMargin;

      const padTop = 20;
      const padBottom = 30;
      const plotH = rect.height - padTop - padBottom;

      const x = padLeft + (closest.step / maxStep) * plotW;
      const y = padTop + plotH - ((closest.loss - yMin) / (yMax - yMin)) * plotH;

      setHoveredPoint({
        step: closest.step,
        loss: closest.loss,
        x,
        y
      });
    }
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
  };

  return (
    <div className="w-full flex flex-col bg-[#121722] border border-[#1E2638] rounded-xl overflow-hidden shadow-sm">
      {/* Chart Header Bar */}
      <div className="px-4 py-2.5 bg-[#181F2E] border-b border-[#1E2638] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-[#FF7A00]" />
          <span className="text-xs font-semibold text-white tracking-wide uppercase">
            Кривая потерь (Real-Time Loss Curve)
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono-code text-[#94A3B8]">
          <span>
            Мин. Loss: <strong className="text-white">{Math.min(...history.map(h => h.loss), telemetry.loss).toFixed(4)}</strong>
          </span>
          <span>
            Текущий: <strong className="text-[#FF7A00]">{telemetry.loss.toFixed(4)}</strong>
          </span>
        </div>
      </div>

      {/* Canvas container */}
      <div
        ref={containerRef}
        className="relative w-full h-56 sm:h-64 cursor-crosshair"
      >
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          className="w-full h-full block"
        />

        {/* Hover Tooltip Overlay */}
        {hoveredPoint && (
          <div
            className="absolute pointer-events-none transform -translate-x-1/2 -translate-y-full px-2.5 py-1.5 rounded bg-[#181F2E] border border-[#273248] text-xs shadow-xl font-mono-code z-10"
            style={{
              left: `${hoveredPoint.x}px`,
              top: `${Math.max(hoveredPoint.y - 10, 30)}px`
            }}
          >
            <div className="text-white font-semibold">Шаг: {hoveredPoint.step}</div>
            <div className="text-[#FF7A00]">Loss: {hoveredPoint.loss.toFixed(4)}</div>
          </div>
        )}
      </div>

      {/* Chart Footer with Legend and Status */}
      <div className="px-4 py-2 bg-[#121722] border-t border-[#1E2638] flex items-center justify-between text-[11px] font-mono-code text-[#94A3B8]">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#FF7A00]" /> Обучение (Train Loss)
          </span>
          <span className="text-[#64748B]">|</span>
          <span>Сглаживание: EMA (0.9)</span>
        </div>
        <div className="text-[#64748B]">
          Обновление: 60 fps (Vulkan Queue Sync)
        </div>
      </div>
    </div>
  );
};
