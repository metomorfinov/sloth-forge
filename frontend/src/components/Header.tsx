import React from 'react';
import { Flame, Cpu, HardDrive, Network, Sliders, MessageSquare, Database, Download } from 'lucide-react';
import type { TelemetryData } from '../types';

interface NavItem {
  id: 'studio' | 'cluster' | 'playground' | 'datasets' | 'export';
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
}

interface HeaderProps {
  currentTab: 'studio' | 'cluster' | 'playground' | 'datasets' | 'export';
  setCurrentTab: (tab: 'studio' | 'cluster' | 'playground' | 'datasets' | 'export') => void;
  telemetry: TelemetryData;
  isLiveWs: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  setCurrentTab,
  telemetry,
  isLiveWs
}) => {
  const vramPercent = Math.min(100, Math.round((telemetry.vramUsedMb / telemetry.vramTotalMb) * 100));

  const navItems: NavItem[] = [
    { id: 'studio', label: 'Fine-Tuning Studio', icon: Sliders },
    { id: 'cluster', label: '2-PC Cluster', icon: Network, badge: telemetry.clusterMode !== 'local' ? '2x active' : undefined },
    { id: 'playground', label: 'Playground', icon: MessageSquare },
    { id: 'datasets', label: 'Datasets', icon: Database },
    { id: 'export', label: 'Export & Merge', icon: Download },
  ];

  return (
    <header className="w-full bg-[#0B0E14] border-b border-[#1E2638] sticky top-0 z-50 select-none">
      {/* Top Primary Bar */}
      <div className="max-w-[1720px] mx-auto px-4 lg:px-6 h-16 flex items-center justify-between gap-4">
        {/* Left: Logo & Unsloth Style Branding */}
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => setCurrentTab('studio')}>
            <div className="w-9 h-9 rounded-xl bg-[#181F2E] border border-[#273248] flex items-center justify-center shadow-inner">
              <Flame className="w-5 h-5 text-[#FF7A00]" />
            </div>
            <div className="flex flex-col">
              <div className="flex items-center gap-1.5 leading-none">
                <span className="font-extrabold tracking-tight text-white text-lg">SLOTH</span>
                <span className="font-extrabold tracking-tight text-[#FF7A00] text-lg">FORGE</span>
              </div>
              <span className="text-[10px] text-[#94A3B8] tracking-wider font-mono-code uppercase mt-0.5">
                Vulkan AI Studio
              </span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="hidden md:flex items-center gap-1 bg-[#121722] p-1 rounded-full border border-[#1E2638]">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentTab(item.id)}
                  className={`relative flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-[#181F2E] text-white border border-[#273248] shadow-sm'
                      : 'text-[#94A3B8] hover:text-white hover:bg-[#181F2E]/50 border border-transparent'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-[#FF7A00]' : 'text-[#94A3B8]'}`} />
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className="px-1.5 py-0.2 text-[9px] font-semibold bg-[#FF7A00]/20 text-[#FF7A00] rounded-full border border-[#FF7A00]/40">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Right: Real-time Telemetry & Hardware Badges */}
        <div className="flex items-center gap-3">
          {/* Hardware Status Badge */}
          <div className="hidden xl:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#121722] border border-[#1E2638]">
            <Cpu className="w-4 h-4 text-[#FF7A00]" />
            <div className="flex flex-col text-left">
              <span className="text-[11px] font-semibold text-white leading-tight">
                AMD Radeon RX 570 <span className="text-[#94A3B8] font-normal">(RADV POLARIS10)</span>
              </span>
              <span className="text-[10px] font-mono-code text-[#94A3B8] leading-tight">
                Vulkan 1.4 API • Compute Queue
              </span>
            </div>
          </div>

          {/* Live VRAM Usage Meter */}
          <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg bg-[#121722] border border-[#1E2638] min-w-[210px]">
            <HardDrive className="w-4 h-4 text-[#94A3B8]" />
            <div className="flex-1 flex flex-col gap-1">
              <div className="flex justify-between text-[11px] font-mono-code">
                <span className="text-[#94A3B8]">VRAM:</span>
                <span className="text-white font-medium">
                  {telemetry.vramUsedMb.toLocaleString()} / {telemetry.vramTotalMb.toLocaleString()} MB ({vramPercent}%)
                </span>
              </div>
              <div className="w-full bg-[#181F2E] h-1.5 rounded-full overflow-hidden border border-[#273248]">
                <div
                  className="h-full bg-gradient-to-r from-[#FF7A00] to-[#FF8F26] rounded-full transition-all duration-300"
                  style={{ width: `${vramPercent}%` }}
                />
              </div>
            </div>
          </div>

          {/* Backend Badge */}
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[#181F2E] border border-[#273248]">
            <div className="w-2 h-2 rounded-full bg-[#10B981]" />
            <span className="text-[11px] font-mono-code text-white font-medium">
              Vulkan Native (Polaris-64)
            </span>
          </div>

          {/* WS Connectivity indicator */}
          <div
            className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono-code text-[#94A3B8] bg-[#121722] border border-[#1E2638]"
            title={isLiveWs ? 'Connected to local Vulkan WebSocket daemon' : 'Running responsive standalone simulation'}
          >
            <div className={`w-1.5 h-1.5 rounded-full ${isLiveWs ? 'bg-[#10B981]' : 'bg-[#FF7A00]'}`} />
            <span>{isLiveWs ? 'WS LIVE' : 'SYNC'}</span>
          </div>
        </div>
      </div>

      {/* Mobile Nav Drawer Row */}
      <div className="md:hidden flex items-center justify-around border-t border-[#1E2638] px-2 py-1.5 bg-[#121722]">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setCurrentTab(item.id)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs ${
                isActive ? 'bg-[#181F2E] text-white border border-[#273248]' : 'text-[#94A3B8]'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
    </header>
  );
};
