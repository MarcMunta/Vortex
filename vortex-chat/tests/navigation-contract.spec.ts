import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (relativePath: string) => fs.readFileSync(path.join(root, relativePath), "utf8");

test("sidebar exposes only chat, spatial, and compact brand", () => {
  const sidebar = read("components/Sidebar.tsx");
  expect(sidebar).toContain("nav_chat");
  expect(sidebar).toContain("nav_spatial");
  expect(sidebar).not.toContain("nav_analysis");
  expect(sidebar).not.toContain("nav_training");
  expect(sidebar).not.toContain("nav_edits");
  expect(sidebar).not.toContain("nav_terminal");
  expect(sidebar).not.toContain("Local core");
  expect(sidebar).not.toContain("Frontend principal");
});

test("visible view contract removed legacy product panels", () => {
  const types = read("types.ts");
  const app = read("App.tsx");
  expect(types).toContain("export type ViewType = 'chat' | 'spatial'");
  expect(types).not.toContain("'analysis'");
  expect(types).not.toContain("'training'");
  expect(types).not.toContain("'edits'");
  expect(types).not.toContain("'terminal'");
  expect(app).not.toContain("TrainingView");
  expect(app).not.toContain("AnalysisView");
  expect(app).not.toContain("TerminalView");
  expect(app).not.toContain("SelfEditsView");
});

test("translations do not expose removed nav as primary navigation", () => {
  const translations = read("translations.ts");
  expect(translations).not.toContain("nav_analysis");
  expect(translations).not.toContain("nav_training");
  expect(translations).not.toContain("nav_edits");
  expect(translations).not.toContain("nav_terminal");
});

test("local training actions are absent from composer and app shell", () => {
  const chatInput = read("components/ChatInput.tsx");
  const app = read("App.tsx");
  expect(chatInput).not.toContain("allowAutoTrain");
  expect(chatInput).not.toContain("Encolar para aprendizaje");
  expect(chatInput).not.toContain("Queue for learning");
  expect(app).not.toContain("startTraining");
  expect(app).not.toContain("submitFeedback");
});

test("stack status is kept out of the top bar", () => {
  const header = read("app/AppHeader.tsx");
  const app = read("App.tsx");
  expect(header).not.toContain("OperationalStatus");
  expect(header).not.toContain("activeEngineLabel");
  expect(app).not.toContain("TopBarStackStatus");
  expect(app).not.toContain("Revisar Stack");
});

test("settings own theme and project permissions", () => {
  const types = read("types.ts");
  const shell = read("app/shellUtils.ts");
  const settings = read("components/SettingsModal.tsx");
  const chatInput = read("components/ChatInput.tsx");
  expect(types).toContain("export type PermissionLevel = 'none' | 'read' | 'edit' | 'full'");
  expect(types).toContain("export type ThemeMode = 'light' | 'dark' | 'system'");
  expect(shell).toContain("permissionsFromProject");
  expect(settings).toContain("Añadir workspace");
  expect(settings).toContain("Tema de la app");
  expect(chatInput).toContain("Gestionar proyectos");
});

test("chat send is not hard-blocked by status polling", () => {
  const app = read("App.tsx");
  expect(app).toContain("const sendDisabledReason = undefined");
  expect(app).not.toContain("if (baseSendDisabledReason)");
  expect(app).not.toContain("Selecciona un proyecto para usar agente");
});
