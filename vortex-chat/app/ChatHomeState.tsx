import React from "react";
import { BrainCircuit, Code2, DatabaseZap } from "lucide-react";
import VortexLogo from "../components/VortexLogo";

type ChatHomeStateProps = {
  activeEngineLabel: string;
  activeModelLabel: string;
  language: "es" | "en";
  readyLabel: string;
  sendDisabledReason?: string;
  statusBody?: string;
  statusHeadline: string;
};

export const ChatHomeState: React.FC<ChatHomeStateProps> = ({
  activeEngineLabel,
  language,
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

        <div className="rounded-full border border-border/60 bg-background px-4 py-2 text-xs font-semibold text-muted-foreground">
          {readyLabel}: {statusHeadline} {sendDisabledReason || statusBody || ""}
        </div>
      </div>
    </div>
  );
};
