import React from "react";
import { Layers3, MessageSquare, PanelLeft, Zap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import VortexLogo from "../components/VortexLogo";
import { Language, ViewType } from "../types";

type AppHeaderProps = {
  activeView: ViewType;
  headerVisible: boolean;
  isSidebarOpen: boolean;
  language: Language;
  onOpenCommandPalette: () => void;
  onSelectView: (view: ViewType) => void;
  onSetLanguage: (language: Language) => void;
  onShowSidebar: () => void;
  springConfig: {
    type: "spring";
    damping: number;
    stiffness: number;
    mass: number;
  };
};

export const AppHeader: React.FC<AppHeaderProps> = ({
  activeView,
  headerVisible,
  isSidebarOpen,
  language,
  onOpenCommandPalette,
  onSelectView,
  onSetLanguage,
  onShowSidebar,
  springConfig,
}) => {
  return (
    <motion.header
      initial={false}
      animate={{ y: headerVisible ? 0 : -100, opacity: headerVisible ? 1 : 0 }}
      transition={springConfig}
      className="absolute left-0 right-0 top-0 z-40 flex h-[68px] items-center justify-between border-b border-border/60 bg-background/88 px-5 backdrop-blur-xl pointer-events-auto accelerated lg:px-8"
    >
      <div className="flex items-center gap-6">
        <AnimatePresence mode="wait">
          {!isSidebarOpen && (
            <motion.button
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              whileHover={{ scale: 1.08, backgroundColor: "hsla(var(--muted-foreground) / 0.1)" }}
              whileTap={{ scale: 0.92 }}
              onClick={onShowSidebar}
              className="rounded-xl p-3 transition-all"
              aria-label={language === "es" ? "Abrir lateral" : "Open sidebar"}
            >
              <PanelLeft size={22} />
            </motion.button>
          )}
        </AnimatePresence>
        <div className="flex items-center gap-3">
          <VortexLogo size={34} alt="Vortex" />
          <div className="flex items-center gap-3">
            <h1 className="text-[17px] font-black leading-none tracking-tight">Vortex</h1>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="relative flex items-center gap-1 rounded-xl border border-border/60 bg-muted/20 p-1">
          {(["chat", "spatial"] as ViewType[]).map((view) => (
            <button
              key={view}
              onClick={() => onSelectView(view)}
              className={`relative z-10 rounded-lg p-2.5 transition-all ${activeView === view ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground"}`}
              aria-label={view === "chat" ? "Chat" : "Spatial"}
              title={view === "chat" ? "Chat" : "Spatial"}
            >
              {view === "chat" ? <MessageSquare size={16} /> : <Layers3 size={16} />}
              {activeView === view && (
                <motion.div layoutId="header-nav-indicator" className="absolute inset-0 -z-10 rounded-lg bg-primary" transition={springConfig} />
              )}
            </button>
          ))}
        </div>
        <motion.button
          whileHover={{ scale: 1.06, backgroundColor: "hsla(var(--muted) / 0.8)" }}
          whileTap={{ scale: 0.94 }}
          onClick={() => onSetLanguage(language === "es" ? "en" : "es")}
          className="flex h-10 w-10 items-center justify-center overflow-hidden rounded-full border border-border/60 bg-muted/20 transition-all shadow-sm"
          aria-label={language === "es" ? "Cambiar a ingles" : "Switch to Spanish"}
        >
          <img
            src={language === "es" ? "https://flagcdn.com/w80/es.png" : "https://flagcdn.com/w80/us.png"}
            alt={language}
            className="h-5 w-5 rounded-full object-cover select-none"
          />
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.95 }}
          onClick={onOpenCommandPalette}
          className="flex items-center gap-3 rounded-xl border border-border/60 bg-muted/20 px-4 py-2.5 transition-all hover:bg-background"
        >
          <Zap size={16} className="text-primary" />
          <kbd className="hidden rounded-lg border bg-background px-2 py-0.5 text-[8px] font-black opacity-40 lg:inline-block">ALT+K</kbd>
        </motion.button>
      </div>
    </motion.header>
  );
};
