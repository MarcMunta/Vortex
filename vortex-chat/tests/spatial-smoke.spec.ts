import { expect, test, type Route } from "@playwright/test";
import type { SpatialPanelModel, SpatialSessionState } from "../types";

const createSession = (now: number): SpatialSessionState => ({
  session_id: "playwright-spatial-smoke",
  selected_object_id: null,
  selected_region: null,
  active_panel_ids: [],
  active_presentation_id: null,
  active_page_index: 0,
  interaction_mode: "inspect",
  last_voice_command: null,
  last_gesture_event: null,
  camera_state: null,
  gesture_state: null,
  focused_item: null,
  recent_multimodal_summary: "Workspace multimodal listo.",
  panels: [],
  updated_at: now,
  created_at: now,
});

const createPanel = (
  base: Partial<SpatialPanelModel> & Pick<SpatialPanelModel, "id" | "type" | "title">,
  now: number,
): SpatialPanelModel => ({
  id: base.id,
  type: base.type,
  title: base.title,
  content: base.content || "",
  source: base.source || {},
  transform: {
    x: 180,
    y: 140,
    z: 0,
    scale: 1,
    rotation: 0,
    skew_x: 0,
    skew_y: 0,
    tilt_x: 0,
    tilt_y: 0,
    perspective: 1100,
    width: 360,
    height: 220,
    ...(base.transform || {}),
  },
  page_index: base.page_index ?? 0,
  page_count: base.page_count ?? 1,
  selected: base.selected ?? false,
  locked: base.locked ?? false,
  created_at: base.created_at ?? now,
  updated_at: base.updated_at ?? now,
});

const json = async (route: Route, payload: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  });
};

test("spatial workspace handles core multimodal flows", async ({ page }, testInfo) => {
  const vaultPath = testInfo.outputPath("obsidian-smoke-vault");
  const screenshotPath = testInfo.outputPath("spatial-workspace-smoke.png");
  let now = Date.now();
  let session = createSession(now);
  let obsidianStatus = {
    ok: true,
    enabled: true,
    vault_path: vaultPath,
    resolved_vault_path: vaultPath,
    available: true,
    validated: true,
    folders: {
      session: "Projects/Vortex/Sessions",
    },
    last_saved_note: null as string | null,
  };

  const multimodalStatus = () => ({
    ok: true,
    voice: {
      ok: true,
      enabled: true,
      push_to_talk: true,
      vad_enabled: true,
      whisper_model: "small",
      tts_model: "mock",
      asr_backend: "mock-whisper",
      tts_backend: "browser",
      asr_available: true,
      tts_available: true,
      output_dir: `${vaultPath}/audio`,
    },
    spatial: session,
    obsidian: obsidianStatus,
    fusion: {
      enabled: true,
      summary: "Workspace multimodal listo.",
      refs: [],
    },
  });

  await page.route("**/control/**", async (route) => {
    const url = new URL(route.request().url());
    const pathname = url.pathname;

    if (pathname === "/control/status") {
      await json(route, {
        ok: true,
        bootstrap: { running: false, stage: "idle", updated_at: Date.now() / 1000 },
        docker: { ready: true, reason: "docker_ready" },
        model: { cached: true, model_id: "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ" },
        runtime: {
          api_ready: true,
          runtime_ready: true,
          runtime_mode: "primary",
          fallback_active: false,
          fallback_backend: null,
          status: {
            ok: true,
            chat_ready: true,
            engine_ready: true,
            model_ready: true,
            training_ready: true,
            offline_ready: true,
            web_disabled: true,
            engine_kind: "sglang",
            engine_base_url: "http://127.0.0.1:30000",
            active_model: "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
          },
        },
        frontend: { ready: true, port: 4173, url: "http://127.0.0.1:4173" },
        internet: { allowlist: ["docs.python.org", "react.dev"] },
        multimodal: multimodalStatus(),
        autonomy: {
          enabled: false,
          boot_mode: "manual",
          state: "idle",
          active_agents: [],
          training_queue: [],
        },
        runs: [],
      });
      return;
    }

    if (pathname === "/control/multimodal/status") {
      await json(route, { ok: true, status: multimodalStatus() });
      return;
    }

    if (pathname === "/control/multimodal/stream") {
      await route.fulfill({
        status: 200,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
        body: `data: ${JSON.stringify({ ts: Date.now() / 1000, status: multimodalStatus() })}\n\n`,
      });
      return;
    }

    if (pathname === "/control/obsidian/config") {
      const body = route.request().postDataJSON() as { enabled?: boolean; vault_path?: string };
      obsidianStatus = {
        ...obsidianStatus,
        enabled: body.enabled ?? true,
        vault_path: body.vault_path || obsidianStatus.vault_path,
        resolved_vault_path: body.vault_path || obsidianStatus.resolved_vault_path,
      };
      await json(route, { ...obsidianStatus, ok: true });
      return;
    }

    await json(route, { ok: true });
  });

  await page.route("**/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const pathname = url.pathname;

    if (pathname === "/v1/status") {
      await json(route, {
        ok: true,
        chat_ready: true,
        offline_ready: true,
        engine_ready: true,
        model_ready: true,
        training_ready: true,
        web_disabled: true,
        engine_kind: "sglang",
        engine_base_url: "http://127.0.0.1:30000",
        active_model: "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        backends: ["sglang"],
        adapters: {},
        metrics: { chat_requests: 0, avg_latency_ms: 18, completion_tokens_est: 0 },
        episodes: 0,
        knowledge_chunks: 0,
        autolearn: {
          total_web_chunks: 0,
          total_code_chunks: 0,
          total_analyses: 0,
          total_proposals: 0,
          discovered_urls: [],
        },
      });
      return;
    }

    if (pathname === "/v1/chat/sessions") {
      await json(route, { ok: true, sessions: [] });
      return;
    }

    if (pathname === "/v1/chat/sessions/sync") {
      await json(route, { ok: true });
      return;
    }

    if (pathname === "/v1/self-edits/proposals") {
      await json(route, { ok: true, data: [] });
      return;
    }

    if (pathname === "/v1/voice/status") {
      await json(route, multimodalStatus().voice);
      return;
    }

    if (pathname === "/v1/obsidian/status") {
      await json(route, obsidianStatus);
      return;
    }

    if (pathname === "/v1/obsidian/config") {
      const body = route.request().postDataJSON() as { enabled?: boolean; vault_path?: string };
      obsidianStatus = {
        ...obsidianStatus,
        enabled: body.enabled ?? true,
        vault_path: body.vault_path || obsidianStatus.vault_path,
        resolved_vault_path: body.vault_path || obsidianStatus.resolved_vault_path,
      };
      await json(route, obsidianStatus);
      return;
    }

    if (pathname === "/v1/obsidian/save") {
      const notePath = `${vaultPath}/Projects/Vortex/Sessions/spatial-smoke.md`;
      obsidianStatus = { ...obsidianStatus, last_saved_note: notePath };
      await json(route, { ok: true, path: notePath });
      return;
    }

    if (pathname === "/v1/spatial/session" && route.request().method() === "GET") {
      await json(route, { ok: true, session });
      return;
    }

    if (pathname === "/v1/spatial/session" && route.request().method() === "POST") {
      const body = route.request().postDataJSON() as Partial<SpatialSessionState>;
      session = {
        ...session,
        ...body,
        selected_region: body.selected_region ?? session.selected_region,
        panels: body.panels ?? session.panels,
        active_panel_ids: body.active_panel_ids ?? session.active_panel_ids,
        selected_object_id: body.selected_object_id ?? session.selected_object_id,
        updated_at: Date.now(),
      };
      await json(route, { ok: true, session });
      return;
    }

    if (pathname === "/v1/spatial/events") {
      await json(route, { ok: true, session });
      return;
    }

    if (pathname === "/v1/spatial/panels/open") {
      const body = route.request().postDataJSON() as SpatialPanelModel;
      const incoming = {
        ...body,
        transform: { ...body.transform },
      };
      session = {
        ...session,
        panels: [...session.panels.filter((panel) => panel.id !== incoming.id), incoming],
        selected_object_id: incoming.id,
        active_panel_ids: [...new Set([...session.active_panel_ids, incoming.id])],
        updated_at: Date.now(),
      };
      await json(route, { ok: true, session, panel: incoming });
      return;
    }

    if (pathname === "/v1/spatial/panels/update") {
      const body = route.request().postDataJSON() as { panel_id: string } & Partial<SpatialPanelModel>;
      session = {
        ...session,
        panels: session.panels.map((panel) => (
          panel.id === body.panel_id
            ? {
                ...panel,
                ...body,
                transform: body.transform ? { ...panel.transform, ...body.transform } : panel.transform,
                updated_at: Date.now(),
              }
            : panel
        )),
        updated_at: Date.now(),
      };
      const updated = session.panels.find((panel) => panel.id === body.panel_id);
      await json(route, { ok: true, session, panel: updated });
      return;
    }

    if (pathname === "/v1/spatial/panels/navigate") {
      const body = route.request().postDataJSON() as { panel_id: string; delta: number };
      session = {
        ...session,
        active_presentation_id: body.panel_id,
        panels: session.panels.map((panel) => (
          panel.id === body.panel_id
            ? {
                ...panel,
                page_index: Math.max(0, Math.min(panel.page_count - 1, panel.page_index + body.delta)),
                updated_at: Date.now(),
              }
            : panel
        )),
        updated_at: Date.now(),
      };
      const updated = session.panels.find((panel) => panel.id === body.panel_id);
      await json(route, { ok: true, session, panel: updated });
      return;
    }

    if (pathname === "/v1/voice/transcribe") {
      now += 1;
      const presentationPanel = createPanel({
        id: `presentation-${now}`,
        type: "presentation",
        title: "Presentación local",
        content: "Slide 1",
        page_count: 3,
        page_index: 0,
        source: {
          pages: ["Open this presentation here", "Move this left", "Tilt this panel"],
        },
      }, now);
      session = {
        ...session,
        last_voice_command: "open this presentation here",
        selected_object_id: presentationPanel.id,
        active_panel_ids: [...new Set([...session.active_panel_ids, presentationPanel.id])],
        active_presentation_id: presentationPanel.id,
        panels: [...session.panels, presentationPanel],
        updated_at: Date.now(),
      };
      await json(route, {
        ok: true,
        transcript: "open this presentation here",
        detected_language: "en",
        intent: { kind: "open_presentation" },
        action_result: { session },
      });
      return;
    }

    await json(route, { ok: true });
  });

  await page.goto("/", { waitUntil: "networkidle" });
  const navGroup = page.locator("header > div").nth(1).locator(":scope > div").nth(0);
  const navBox = await navGroup.boundingBox();
  if (!navBox) throw new Error("header nav geometry unavailable");
  await page.mouse.click(navBox.x + navBox.width * 0.25, navBox.y + navBox.height / 2);
  await expect(page.getByText("Camara, voz, gestos y paneles pseudo-3D en shell Vortex.")).toBeVisible();

  await page.getByTestId("spatial-open-note").click();
  const notePanel = page.locator('[data-panel-type="note"]').last();
  await expect(notePanel).toBeVisible();

  const noteBoxBefore = await notePanel.boundingBox();
  const noteHeader = notePanel.locator('[data-testid^="panel-header-"]').first();
  const noteHeaderBox = await noteHeader.boundingBox();
  if (!noteBoxBefore || !noteHeaderBox) throw new Error("note panel geometry unavailable");

  await page.mouse.move(noteHeaderBox.x + noteHeaderBox.width / 2, noteHeaderBox.y + noteHeaderBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(noteHeaderBox.x + noteHeaderBox.width / 2 + 120, noteHeaderBox.y + noteHeaderBox.height / 2 + 70, { steps: 12 });
  await page.mouse.up();

  const noteBoxAfter = await notePanel.boundingBox();
  if (!noteBoxAfter) throw new Error("note panel moved geometry unavailable");
  expect(Math.abs(noteBoxAfter.x - noteBoxBefore.x)).toBeGreaterThan(30);

  await page.getByTestId("spatial-toggle-region").click();
  await page.waitForTimeout(250);
  const stage = page.getByTestId("spatial-stage");
  const stageBox = await stage.boundingBox();
  if (!stageBox) throw new Error("stage geometry unavailable");

  await page.mouse.move(stageBox.x + stageBox.width - 360, stageBox.y + stageBox.height - 280);
  await page.mouse.down();
  await page.mouse.move(stageBox.x + stageBox.width - 90, stageBox.y + stageBox.height - 90, { steps: 12 });
  await page.mouse.up();

  await expect(page.getByText(/\d+,\s*\d+\s*-\s*\d+x\d+/)).toBeVisible();

  await page.getByTestId("voice-command-input").fill("open this presentation here");
  await Promise.all([
    page.waitForResponse((response) => response.url().includes("/v1/voice/transcribe") && response.ok()),
    page.getByTestId("voice-command-run").click(),
  ]);

  const presentationPanel = page.locator('[data-panel-type="presentation"]').last();
  await expect(presentationPanel).toBeVisible();
  await expect(presentationPanel).toContainText("1/3");

  await page.locator('[data-testid^="panel-next-"]').last().click();
  await expect(presentationPanel).toContainText("2/3");

  await page.getByTestId("obsidian-vault-input").fill(vaultPath);
  await Promise.all([
    page.waitForResponse((response) => response.url().includes("/v1/obsidian/config") && response.ok()),
    page.getByTestId("obsidian-vault-save").click(),
  ]);

  await Promise.all([
    page.waitForResponse((response) => response.url().includes("/v1/obsidian/save") && response.ok()),
    page.getByTestId("voice-save-obsidian").click(),
  ]);

  const obsidianPanel = page.locator('[data-panel-type="obsidian"]').last();
  await expect(obsidianPanel).toBeVisible();

  await page.screenshot({
    path: screenshotPath,
    fullPage: true,
  });
});
