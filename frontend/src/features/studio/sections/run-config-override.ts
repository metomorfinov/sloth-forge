// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseBackendTrainingMethod } from "@/features/training";

/** Shape of the Training Config popover's data when it is driven by a saved
 * run snapshot instead of the editable form store. */
export interface RunConfigOverride {
  trainingMethod?: string;
  epochs?: number;
  batchSize?: number;
  learningRate?: string;
  maxSteps?: number;
  contextLength?: number;
  warmupSteps?: number;
  optimizerType?: string;
  loraRank?: number;
  loraAlpha?: number;
  loraDropout?: number;
  loraVariant?: string;
  device?: string;
  targetDevice?: string;
  backend?: string;
  gpuName?: string;
  vramTotalMb?: number;
}

/** Default device configuration for AMD Polaris 10 (RADV / Vulkan) */
export const POLARIS_VULKAN_DEFAULTS = {
  gpuName: "AMD Radeon RX 570 (RADV POLARIS10)",
  targetDevice: "AMD Radeon RX 470/480/570/580 (Polaris 10)",
  backend: "Vulkan Native (Polaris-64)",
  deviceType: "vulkan",
  vramTotalMb: 4096,
  batchSize: 1,
  gradientAccumulation: 4,
  contextLength: 1024,
  optimizerType: "adamw_8bit",
  loraRank: 16,
  loraAlpha: 32,
  loraDropout: 0.05,
} as const;

/** Map a saved run's config (GET /api/train/runs/{id} `detail.config`) into the
 * Training Config popover's override shape. Shared by the History view and the
 * live Current Run view so both read the same authoritative run snapshot
 * instead of the editable form store (#6853). */
export function mapRunConfigToOverride(
  config: Record<string, unknown> | null | undefined,
): RunConfigOverride | undefined {
  if (!config) {
    return undefined;
  }
  return {
    trainingMethod: parseBackendTrainingMethod(
      config.training_type,
      config.load_in_4bit,
    ),
    epochs: (config.num_epochs ?? config.epochs) as number | undefined,
    batchSize: (config.batch_size ?? config.micro_batch_size ?? config.batchSize) as number | undefined,
    learningRate: (config.learning_rate ?? config.learningRate) != null
      ? String(config.learning_rate ?? config.learningRate)
      : undefined,
    maxSteps: (config.max_steps ?? config.total_steps ?? config.maxSteps) as number | undefined,
    contextLength: (config.max_seq_length ?? config.contextLength ?? config.context_length) as number | undefined,
    warmupSteps: (config.warmup_steps ?? config.warmupSteps) as number | undefined,
    optimizerType: (config.optim ?? config.optimizer ?? config.optimizerType) as string | undefined,
    loraRank: (config.lora_r ?? config.loraRank ?? config.lora_rank) as number | undefined,
    loraAlpha: (config.lora_alpha ?? config.loraAlpha) as number | undefined,
    loraDropout: (config.lora_dropout ?? config.loraDropout) as number | undefined,
    loraVariant: config.use_rslora
      ? "rslora"
      : config.use_loftq
        ? "loftq"
        : config.use_dora
          ? "dora"
          : "lora",
    targetDevice:
      (config.target_device as string | undefined) ??
      (config.targetDevice as string | undefined) ??
      POLARIS_VULKAN_DEFAULTS.targetDevice,
    device:
      (config.device as string | undefined) ??
      (config.device_name as string | undefined) ??
      POLARIS_VULKAN_DEFAULTS.gpuName,
    backend:
      (config.backend as string | undefined) ??
      POLARIS_VULKAN_DEFAULTS.backend,
    gpuName:
      (config.gpu_name as string | undefined) ??
      POLARIS_VULKAN_DEFAULTS.gpuName,
    vramTotalMb:
      (config.vram_total_mb as number | undefined) ??
      POLARIS_VULKAN_DEFAULTS.vramTotalMb,
  };
}
