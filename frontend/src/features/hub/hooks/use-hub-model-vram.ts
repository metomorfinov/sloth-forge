// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import type { GpuInfo } from "@/hooks/use-gpu-info";
import {
  type VramFitStatus,
  checkVramFit,
  estimateLoadingVram,
} from "@/lib/vram";
import { requiredGgufMemoryGb } from "@/lib/gguf-fit";
import { formatBytes } from "@/features/hub/lib/format";
import type { SelectedModelView } from "../types";

export interface ModelVramInfo {
  est: number;
  status: VramFitStatus;
}

export function useHubModelVram(
  selectedModel: SelectedModelView | null,
  gpu: GpuInfo,
): { vramInfo: ModelVramInfo | null; minMemory: string | null } {
  const vramInfo = useMemo<ModelVramInfo | null>(() => {
    if (!selectedModel) {
      return null;
    }
    const gpuGb = gpu.available ? gpu.memoryTotalGb : 0;
    if (selectedModel.isGguf) {
      const sizeBytes =
        selectedModel.cachedBytes ||
        selectedModel.estimatedSizeBytes ||
        (selectedModel.totalParams ? selectedModel.totalParams * 0.5 : 0);
      if (!sizeBytes) return null;
      const est = Math.round(requiredGgufMemoryGb(sizeBytes) * 10) / 10;
      const status = checkVramFit(est, gpuGb);
      return status ? { est, status } : null;
    }
    if (!selectedModel.totalParams) {
      return null;
    }
    const est = estimateLoadingVram(selectedModel.totalParams, "qlora");
    const status = checkVramFit(est, gpuGb);
    return status ? { est, status } : null;
  }, [gpu, selectedModel]);

  const minMemory = useMemo(() => {
    if (!selectedModel) return null;
    if (selectedModel.isGguf) {
      if (selectedModel.cachedBytes)
        return formatBytes(selectedModel.cachedBytes);
      if (selectedModel.estimatedSizeBytes)
        return formatBytes(selectedModel.estimatedSizeBytes);
      return null;
    }
    if (selectedModel.estimatedSizeBytes)
      return formatBytes(selectedModel.estimatedSizeBytes);
    if (vramInfo) return `~${vramInfo.est} GB`;
    if (selectedModel.cachedBytes)
      return formatBytes(selectedModel.cachedBytes);
    return null;
  }, [selectedModel, vramInfo]);

  return { vramInfo, minMemory };
}
