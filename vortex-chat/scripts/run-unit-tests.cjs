const { spawnSync } = require("node:child_process");

const passthrough = process.argv.slice(2).filter((arg) => arg !== "--run");
const args = [
  "playwright",
  "test",
  "--config=playwright.unit.config.ts",
  "--reporter=line",
  ...passthrough,
];

const result = spawnSync("npx", args, { stdio: "inherit", shell: process.platform === "win32" });
process.exit(result.status ?? 1);
