import React from "react";
import { BarChart3, FileCode, FlaskConical, Globe, Layers3, MessageSquare, PanelLeft, Terminal as TerminalIcon, Zap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import TopBarStackStatus from "../components/TopBarStackStatus";
import VortexLogo from "../components/VortexLogo";
import { ControlStatus, Language, OperationalStatus, ViewType } from "../types";

type AppHeaderProps = {
  activeView: ViewType;
  controlStatus: ControlStatus | null;
  headerVisible: boolean;
  isCommandPaletteOpen: boolean;
  isSidebarOpen: boolean;
  language: Language;
  onBootstrap: () => Promise<unknown> | unknown;
  onModelInit: () => Promise<unknown> | unknown;
  onOpenCommandPalette: () => void;
  onOpenTraining: () => void;
  onRestartRuntime: () => Promise<unknown> | unknown;
  onSelectView: (view: ViewType) => void;
  onSetLanguage: (language: Language) => void;
  onShowSidebar: () => void;
  onStartAutonomy: () => Promise<unknown> | unknown;
  onStartTraining: () => Promise<unknown> | unknown;
  onStopAutonomy: () => Promise<unknown> | unknown;
  operationalStatus: OperationalStatus | null;
  springConfig: {
    type: "spring";
    damping: number;
    stiffness: number;
    mass: number;
  };
};

export const AppHeader: React.FC<AppHeaderProps> = ({
  activeView,
  controlStatus,
  headerVisible,
  isSidebarOpen,
  language,
  onBootstrap,
  onModelInit,
  onOpenCommandPalette,
  onOpenTraining,
  onRestartRuntime,
  onSelectView,
  onSetLanguage,
  onShowSidebar,
  onStartAutonomy,
  onStartTraining,
  onStopAutonomy,
  operationalStatus,
  springConfig,
}) => {
  const activeEngineLabel = (operationalStatus?.engine_kind || "local").toUpperCase();

  return (
    <motion.header
      initial={false}
      animate={{ y: headerVisible ? 0 : -100, opacity: headerVisible ? 1 : 0 }}
      transition={springConfig}
      className="absolute top-0 left-0 right-0 z-40 flex h-[72px] items-center justify-between border-b border-border/60 bg-background/88 px-5 backdrop-blur-xl pointer-events-auto accelerated lg:px-8"
    >
      <div className="flex items-center gap-8">
        <AnimatePresence mode="wait">
          {!isSidebarOpen && (
            <motion.button
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              whileHover={{ scale: 1.1, backgroundColor: "hsla(var(--muted-foreground) / 0.1)" }}
              whileTap={{ scale: 0.9 }}
              onClick={onShowSidebar}
              className="p-3.5 rounded-2xl transition-all"
            >
              <PanelLeft size={24} />
            </motion.button>
          )}
        </AnimatePresence>
        <div className="flex items-center gap-4">
          <motion.div whileHover={{ rotate: -8, scale: 1.04 }} transition={{ type: "spring", stiffness: 320, damping: 18 }}>
            <VortexLogo size={40} alt="Vortex" />
          </motion.div>
          <div className="flex flex-col">
            <div className="flex items-center gap-3">
              <h1 className="text-[18px] font-black tracking-tight leading-none">Vortex</h1>
              <span className="rounded-full border border-border/60 bg-muted/25 px-3 py-1 text-[9px] font-black uppercase tracking-[0.14em] text-primary">
                {activeEngineLabel}
              </span>
            </div>
            <span className="mt-1 text-[9px] font-black uppercase tracking-[0.14em] text-muted-foreground">
              {language === "es" ? "kernel del sistema" : "system kernel"}
            </span>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <motion.button
          whileHover={{ scale: 1.06, backgroundColor: "hsla(var(--muted) / 0.8)" }}
          whileTap={{ scale: 0.94 }}
          onClick={() => onSetLanguage(language === "es" ? "en" : "es")}
          className="flex h-11 w-11 items-center justify-center overflow-hidden rounded-full border border-border/60 bg-muted/20 transition-all shadow-sm"
        >
          <img
            src={language === "es" ? "https://flagcdn.com/w80/es.png" : "https://flagcdn.com/w80/us.png"}
            alt={language}
            className="h-6 w-6 rounded-full object-cover select-none"
          />
        </motion.button>
        <div className="flex items-center gap-1 rounded-[1rem] border border-border/60 bg-muted/20 p-1 relative">
          {(["chat", "spatial", "analysis", "training", "edits", "terminal"] as ViewType[]).map((view) => (
            <button
              key={view}
              onClick={() => onSelectView(view)}
              className={`relative rounded-[0.85rem] p-2.5 transition-all z-10 ${activeView === view ? "text-primary-foreground" : "text-muted-foreground dark:text-zinc-400 hover:text-foreground"}`}
            >
              {view === "chat" ? <MessageSquare size={16} /> : view === "spatial" ? <Layers3 size={16} /> : view === "analysis" ? <BarChart3 size={16} /> : view === "training" ? <FlaskConical size={16} /> : view === "edits" ? <FileCode size={16} /> : <TerminalIcon size={16} />}
              {activeView === view && (
                <motion.div layoutId="header-nav-indicator" className="absolute inset-0 bg-primary rounded-[0.85rem] -z-10" transition={springConfig} />
              )}
            </button>
          ))}
        </div>
        <TopBarStackStatus
          status={operationalStatus}
          controlStatus={controlStatus}
          language={language}
          onBootstrap={onBootstrap}
          onModelInit={onModelInit}
          onRestartRuntime={onRestartRuntime}
          onStartTraining={onStartTraining}
          onOpenTraining={onOpenTraining}
          onStartAutonomy={onStartAutonomy}
          onStopAutonomy={onStopAutonomy}
        />
        <motion.button
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.95 }}
          onClick={onOpenCommandPalette}
          className="flex items-center gap-3 rounded-[1rem] border border-border/60 bg-muted/20 px-4 py-2.5 transition-all hover:bg-background"
        >
          <Zap size={16} className="text-primary" />
          <kbd className="hidden lg:inline-block rounded-lg border bg-background px-2 py-0.5 text-[8px] font-black opacity-40">ALT+K</kbd>
        </motion.button>
      </div>
    </motion.header>
  );
};
