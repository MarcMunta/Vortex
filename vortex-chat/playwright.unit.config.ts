import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testIgnore: ["**/spatial-smoke.spec.ts", "**/fullstack-integration.spec.ts"],
  timeout: 30_000,
  fullyParallel: true,
  reporter: "line",
});
