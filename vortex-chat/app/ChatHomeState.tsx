import React from "react";
import { motion } from "framer-motion";
import VortexLogo from "../components/VortexLogo";
import { OperationalStatus } from "../types";

type HeroCard = {
  label: string;
  value: string;
};

type ChatHomeStateProps = {
  activeEngineLabel: string;
  activeModelLabel: string;
  heroCards: HeroCard[];
  language: "es" | "en";
  onLoadDemo: () => void;
  onOpenAnalysis: () => void;
  onOpenTraining: () => void;
  operationalStatus: OperationalStatus | null;
  readyLabel: string;
  sendDisabledReason?: string;
  statusBody?: string;
  statusHeadline: string;
};

export const ChatHomeState: React.FC<ChatHomeStateProps> = ({
  activeEngineLabel,
  activeModelLabel,
  heroCards,
  language,
  onLoadDemo,
  onOpenAnalysis,
  onOpenTraining,
  operationalStatus,
  readyLabel,
  sendDisabledReason,
  statusBody,
  statusHeadline,
}) => {
  return (
    <div className="mx-auto flex w-full max-w-[980px] flex-col items-center justify-center gap-8 text-center">
      <div className="flex flex-col items-center gap-5">
        <div className="inline-flex items-center gap-3 rounded-full border border-border/60 bg-background px-4 py-2 text-[10px] font-black uppercase tracking-[0.14em] text-muted-foreground shadow-sm">
          <VortexLogo size={20} alt="Vortex" />
          <span>{language === "es" ? "Vortex local core" : "Vortex local core"}</span>
          <span className="h-1.5 w-1.5 rounded-full bg-primary" />
          <span className="text-primary">{activeEngineLabel}</span>
        </div>

        <div className="space-y-4">
          <h2 className="max-w-3xl text-4xl font-extrabold tracking-[-0.045em] leading-[1.02] text-foreground lg:text-5xl">
            {language === "es"
              ? "Una sola consola para chatear, controlar y mejorar Vortex."
              : "One console to chat, control, and improve Vortex."}
          </h2>
          <p className="mx-auto max-w-2xl text-[15px] leading-7 text-muted-foreground lg:text-base">
            {language === "es"
              ? "La interfaz principal se comporta como una app de trabajo real: limpia, local y centrada en conversación, control del runtime y entrenamiento visible."
              : "The main interface behaves like a real work app: clean, local, and focused on conversation, runtime control, and visible training."}
          </p>
        </div>

        <div className="grid w-full gap-3 md:grid-cols-3">
          {heroCards.map((card) => (
            <div key={card.label} className="surface-panel rounded-[1.2rem] px-5 py-5 text-left">
              <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">{card.label}</p>
              <p className="mt-2 text-[15px] font-bold tracking-tight text-foreground">{card.value}</p>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap items-center justify-center gap-4">
          <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={onLoadDemo} className="rounded-full bg-foreground px-6 py-3 text-[10px] font-black uppercase tracking-[0.14em] text-background transition-all dark:bg-primary dark:text-primary-foreground">
            {language === "es" ? "Inicializar vortex" : "Initialize vortex"}
          </motion.button>
          <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={onOpenAnalysis} className="rounded-full border border-border/70 bg-background px-6 py-3 text-[10px] font-black uppercase tracking-[0.14em] text-foreground shadow-sm">
            {language === "es" ? "Abrir control" : "Open control"}
          </motion.button>
          <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={onOpenTraining} className="rounded-full border border-primary/25 bg-primary/[0.10] px-6 py-3 text-[10px] font-black uppercase tracking-[0.14em] text-primary shadow-sm">
            {language === "es" ? "Entrenamiento" : "Training"}
          </motion.button>
        </div>

        <p className="max-w-2xl text-sm font-medium text-muted-foreground">{statusBody}</p>
      </div>

      <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: "easeOut" }} className="surface-panel relative isolate w-full max-w-[820px] overflow-hidden rounded-[1.6rem] p-5 text-foreground">
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.10),transparent)] dark:bg-[linear-gradient(180deg,rgba(255,255,255,0.03),transparent)]" />

        <div className="relative z-10 flex flex-col gap-5">
          <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-[0.14em] text-muted-foreground">
            <span>{language === "es" ? "Estado base" : "Base status"}</span>
            <span>{readyLabel}</span>
          </div>

          <div className="flex items-center gap-4 rounded-[1.2rem] border border-border/60 bg-muted/20 px-4 py-4">
            <motion.div animate={{ rotate: [0, 2, -2, 0], scale: [1, 1.01, 1] }} transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}>
              <VortexLogo size={52} alt="Vortex mark" className="max-w-full" />
            </motion.div>
            <div className="text-left">
              <p className="text-sm font-bold tracking-tight text-foreground">{statusHeadline}</p>
              <p className="mt-1 text-sm leading-6 text-muted-foreground">
                {operationalStatus?.ok
                  ? (language === "es"
                    ? "Consulta, agente, navegación puntual y entrenamiento siguen visibles desde la misma interfaz."
                    : "Query, agent mode, prompt browsing, and training stay visible from the same interface.")
                  : sendDisabledReason}
              </p>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-[1rem] border border-border/60 bg-muted/20 p-4">
              <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                {language === "es" ? "Runtime" : "Runtime"}
              </p>
              <p className="mt-2 text-sm font-bold tracking-tight text-foreground">{activeEngineLabel}</p>
              <p className="mt-1 text-xs text-muted-foreground">{operationalStatus?.engine_base_url || "127.0.0.1"}</p>
            </div>
            <div className="rounded-[1rem] border border-border/60 bg-muted/20 p-4">
              <p className="text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
                {language === "es" ? "Modelo activo" : "Active model"}
              </p>
              <p className="mt-2 text-sm font-bold tracking-tight text-foreground break-words">{activeModelLabel}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {operationalStatus?.web_disabled
                  ? (language === "es" ? "Internet solo al activarlo" : "Internet only when enabled")
                  : (language === "es" ? "Política web editable" : "Editable web policy")}
              </p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};
