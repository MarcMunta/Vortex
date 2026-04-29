import React from "react";
import { motion } from "framer-motion";
import { BrainCircuit, Code2, DatabaseZap } from "lucide-react";
import VortexLogo from "../components/VortexLogo";
import { OperationalStatus } from "../types";

type ChatHomeStateProps = {
  activeEngineLabel: string;
  activeModelLabel: string;
  language: "es" | "en";
  onLoadDemo: () => void;
  operationalStatus: OperationalStatus | null;
  readyLabel: string;
  sendDisabledReason?: string;
  statusBody?: string;
  statusHeadline: string;
};

export const ChatHomeState: React.FC<ChatHomeStateProps> = ({
  activeEngineLabel,
  activeModelLabel,
  language,
  onLoadDemo,
  operationalStatus,
  readyLabel,
  sendDisabledReason,
  statusBody,
  statusHeadline,
}) => {
  const cards = language === "es"
    ? [
        { icon: Code2, label: "Chat", value: "Qwen Coder local" },
        { icon: BrainCircuit, label: "Agente", value: "Contexto ampliado" },
        { icon: DatabaseZap, label: "Memoria", value: "Obsidian curado" },
      ]
    : [
        { icon: Code2, label: "Chat", value: "Local Qwen Coder" },
        { icon: BrainCircuit, label: "Agent", value: "Larger context" },
        { icon: DatabaseZap, label: "Memory", value: "Curated Obsidian" },
      ];

  return (
    <div className="mx-auto flex w-full max-w-[940px] flex-col items-center justify-center gap-8 text-center">
      <div className="flex flex-col items-center gap-5">
        <div className="inline-flex items-center gap-3 rounded-full border border-border/60 bg-background px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-muted-foreground shadow-sm">
          <VortexLogo size={20} alt="Vortex" />
          <span>Vortex</span>
          <span className="h-1.5 w-1.5 rounded-full bg-primary" />
          <span className="text-primary">{activeEngineLabel}</span>
        </div>

        <div className="space-y-4">
          <h2 className="max-w-3xl text-4xl font-extrabold leading-[1.02] tracking-tight text-foreground lg:text-5xl">
            {language === "es"
              ? "Chat local y agente de programacion."
              : "Local chat and coding agent."}
          </h2>
          <p className="mx-auto max-w-2xl text-[15px] leading-7 text-muted-foreground lg:text-base">
            {language === "es"
              ? "Vortex prioriza Qwen, contexto de repo y memoria Obsidian curada. Internet y entrenamiento local no se activan por defecto."
              : "Vortex prioritizes Qwen, repo context, and curated Obsidian memory. Internet and local training stay off by default."}
          </p>
        </div>

        <div className="grid w-full gap-3 md:grid-cols-3">
          {cards.map((card) => {
            const Icon = card.icon;
            return (
              <div key={card.label} className="surface-panel rounded-xl px-5 py-5 text-left">
                <Icon size={18} className="text-primary" />
                <p className="mt-4 text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">{card.label}</p>
                <p className="mt-2 text-[15px] font-bold tracking-tight text-foreground">{card.value}</p>
              </div>
            );
          })}
        </div>

        <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={onLoadDemo} className="rounded-full bg-foreground px-6 py-3 text-[10px] font-black uppercase tracking-[0.14em] text-background transition-all dark:bg-primary dark:text-primary-foreground">
          {language === "es" ? "Cargar ejemplo" : "Load example"}
        </motion.button>

        <p className="max-w-2xl text-sm font-medium text-muted-foreground">{statusBody}</p>
      </div>

      <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: "easeOut" }} className="surface-panel relative isolate w-full max-w-[820px] overflow-hidden rounded-2xl p-5 text-foreground">
        <div className="relative z-10 flex flex-col gap-5">
          <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-[0.14em] text-muted-foreground">
            <span>{language === "es" ? "Estado" : "Status"}</span>
            <span>{readyLabel}</span>
          </div>
          <div className="flex items-center gap-4 rounded-xl border border-border/60 bg-muted/20 px-4 py-4">
            <VortexLogo size={48} alt="Vortex mark" className="max-w-full" />
            <div className="text-left">
              <p className="text-sm font-bold tracking-tight text-foreground">{statusHeadline}</p>
              <p className="mt-1 text-sm leading-6 text-muted-foreground">
                {operationalStatus?.ok
                  ? (language === "es" ? "Listo para chat, agente y memoria curada." : "Ready for chat, agent work, and curated memory.")
                  : sendDisabledReason}
              </p>
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
              <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">Runtime</p>
              <p className="mt-2 text-sm font-bold tracking-tight text-foreground">{activeEngineLabel}</p>
            </div>
            <div className="rounded-xl border border-border/60 bg-muted/20 p-4">
              <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                {language === "es" ? "Modelo" : "Model"}
              </p>
              <p className="mt-2 break-words text-sm font-bold tracking-tight text-foreground">{activeModelLabel}</p>
            </div>
          </div>
          <div className="rounded-xl border border-border/60 bg-muted/20 p-4 text-left">
            <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
              {language === "es" ? "Entrenamiento" : "Training"}
            </p>
            <p className="mt-2 text-sm font-semibold text-muted-foreground">
              {language === "es" ? "Google Cloud pendiente de configurar." : "Google Cloud pending configuration."}
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};
