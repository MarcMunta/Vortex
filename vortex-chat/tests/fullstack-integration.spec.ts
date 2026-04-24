import { expect, test } from "@playwright/test";

test("frontend, API and control plane agree on integration status", async ({ page, request }) => {
  const api = process.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
  const control = process.env.VITE_CONTROL_BASE_URL || "http://127.0.0.1:8765";

  const apiStatus = await request.get(`${api}/v1/status`);
  expect(apiStatus.ok()).toBeTruthy();
  expect((await apiStatus.json()).chat_ready).toBeTruthy();

  const controlStatus = await request.get(`${control}/control/status`);
  expect(controlStatus.ok()).toBeTruthy();
  const controlPayload = await controlStatus.json();
  expect(controlPayload.runtime.api_ready).toBeTruthy();
  expect(controlPayload.multimodal.ok).toBeTruthy();

  const runs = await request.get(`${control}/control/training/runs`);
  expect(runs.ok()).toBeTruthy();
  expect((await runs.json()).runs[0].run_id).toBe("integration-run");

  await page.goto("/", { waitUntil: "networkidle" });
  await expect(page.locator("body")).toContainText(/Vortex|integration|Entrenamiento|Model/i);
});
