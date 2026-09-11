// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The leading boundary skips family-version digits ("Qwen3-") and MoE active-param
// notation ("A3B"), so "Qwen3-30B-A3B" reads as 30B total, not 3B active.
const PARAM_COUNT_RE = /(?:^|[-_])(\d+(?:\.\d+)?)[Bb](?:[-_]|$)/;

function matchParamCount(id: string): RegExpMatchArray | null {
  const name = id.split("/").pop() ?? id;
  return name.match(PARAM_COUNT_RE);
}

export function extractParamLabel(id: string): string | null {
  const m = matchParamCount(id);
  if (m) return `${m[1]}B`;
  const lower = id.toLowerCase();
  if (/(?:^|[-_/. ])glm[-_]?5(?:\.\d+)?(?:$|[-_/. ])/i.test(lower)) {
    return "32B";
  }
  if (/(?:^|[-_/. ])glm[-_]?4(?:\.\d+)?(?:$|[-_/. ])/i.test(lower)) {
    return "9B";
  }
  return null;
}

export function parseParamCountB(id: string): number | null {
  const m = matchParamCount(id);
  if (m) {
    const v = Number.parseFloat(m[1]);
    return Number.isFinite(v) ? v : null;
  }
  const lower = id.toLowerCase();
  if (/(?:^|[-_/. ])glm[-_]?5(?:\.\d+)?(?:$|[-_/. ])/i.test(lower)) {
    return 32;
  }
  if (/(?:^|[-_/. ])glm[-_]?4(?:\.\d+)?(?:$|[-_/. ])/i.test(lower)) {
    return 9;
  }
  return null;
}
