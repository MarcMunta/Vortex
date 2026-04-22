import { expect, test } from "@playwright/test";

test("spatial workspace handles core multimodal flows", async ({ page, request }) => {
  const apiUrl = "http://127.0.0.1:8000";
  const appUrl = "http://127.0.0.1:4173";
  const vaultPath = "D:/GitHub/Vortex/output/playwright/obsidian-smoke-vault";
  const now = Date.now();

  await request.post(`${apiUrl}/v1/obsidian/config`, {
    data: { enabled: true, vault_path: vaultPath },
  });

  await request.post(`${apiUrl}/v1/spatial/session`, {
    data: {
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
      recent_multimodal_summary: null,
      panels: [],
      updated_at: now,
      created_at: now,
    },
  });

  await page.goto(appUrl, { waitUntil: "networkidle" });
  await page.getByText("Spatial", { exact: true }).first().click();
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

  const noteStyleBefore = await notePanel.getAttribute("style");
  const noteTiltButton = notePanel.locator('[data-testid^="panel-tilt-"]').first();
  const noteTiltBox = await noteTiltButton.boundingBox();
  if (!noteTiltBox) throw new Error("tilt button geometry unavailable");

  await page.mouse.move(noteTiltBox.x + noteTiltBox.width / 2, noteTiltBox.y + noteTiltBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(noteTiltBox.x + noteTiltBox.width / 2 + 48, noteTiltBox.y + noteTiltBox.height / 2 - 22, { steps: 10 });
  await page.mouse.up();

  const noteStyleAfter = await notePanel.getAttribute("style");
  expect(noteStyleAfter).not.toBe(noteStyleBefore);

  await page.getByTestId("spatial-toggle-region").click();
  await page.waitForTimeout(250);
  const stage = page.getByTestId("spatial-stage");
  const stageBox = await stage.boundingBox();
  if (!stageBox) throw new Error("stage geometry unavailable");

  await page.mouse.move(stageBox.x + stageBox.width - 360, stageBox.y + stageBox.height - 280);
  await page.mouse.down();
  await page.mouse.move(stageBox.x + stageBox.width - 90, stageBox.y + stageBox.height - 90, { steps: 12 });
  await page.mouse.up();

  await expect.poll(async () => {
    const response = await request.get(`${apiUrl}/v1/spatial/session`);
    const data = await response.json();
    return Number(data?.session?.selected_region?.width || 0);
  }).toBeGreaterThan(0);

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
    path: "D:/GitHub/Vortex/output/playwright/spatial-workspace-smoke.png",
    fullPage: true,
  });
});
