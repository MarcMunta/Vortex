import React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { SpatialPanelModel } from "../../types";
import { buildPanelMatrix } from "./matrix";

type TransformMode = "move" | "scale" | "rotate" | "perspective" | "skew";

type PresentationPanelProps = {
  panel: SpatialPanelModel;
  focused: boolean;
  language: "es" | "en";
  onFocus: (panelId: string) => void;
  onStartTransform: (panelId: string, mode: TransformMode, event: React.PointerEvent<HTMLElement>) => void;
  onNavigate: (panelId: string, delta: number) => void;
};

const PresentationPanel: React.FC<PresentationPanelProps> = ({
  panel,
  focused,
  language,
  onFocus,
  onStartTransform,
  onNavigate,
}) => {
  const source = panel.source || {};
  const pages = Array.isArray(source.pages) ? source.pages.map((item) => String(item)) : [];
  const currentPage = pages[panel.page_index] || panel.content || `${language === "es" ? "Slide" : "Slide"} ${panel.page_index + 1}`;

  return (
    <div
      className={`spatial-panel absolute left-0 top-0 ${focused ? "is-focused" : ""}`}
      data-testid={`spatial-panel-${panel.id}`}
      data-panel-id={panel.id}
      data-panel-type={panel.type}
      style={{
        width: `${panel.transform.width}px`,
        height: `${panel.transform.height}px`,
        transform: buildPanelMatrix(panel.transform),
        transformOrigin: "0 0",
      }}
      onPointerDown={(event) => {
        event.stopPropagation();
        onFocus(panel.id);
      }}
    >
      <div data-testid={`panel-header-${panel.id}`} className="flex items-center justify-between gap-3 border-b border-white/10 px-4 py-3" onPointerDown={(event) => onStartTransform(panel.id, "move", event)}>
        <div className="min-w-0">
          <p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{language === "es" ? "Presentacion" : "Presentation"}</p>
          <p className="truncate text-sm font-bold tracking-tight text-foreground">{panel.title}</p>
        </div>
        <p className="rounded-full border border-white/10 px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground">
          {panel.page_index + 1}/{panel.page_count}
        </p>
      </div>
      <div className="flex h-[calc(100%-61px)] flex-col justify-between px-4 py-4">
        <div className="flex-1 rounded-[1.2rem] border border-white/10 bg-[linear-gradient(180deg,rgba(0,174,255,0.10),transparent_40%),rgba(255,255,255,0.03)] p-4">
          <p className="text-[11px] font-black uppercase tracking-[0.14em] text-primary">{language === "es" ? "Frame activo" : "Active frame"}</p>
          <div className="mt-3 flex h-[calc(100%-28px)] items-center justify-center rounded-[1rem] border border-white/10 bg-black/20 p-6 text-center text-lg font-semibold tracking-tight text-foreground">
            {currentPage}
          </div>
        </div>
        <div className="mt-4 flex items-center justify-between gap-3">
          <button type="button" data-testid={`panel-prev-${panel.id}`} className="inline-flex items-center gap-2 rounded-full border border-white/10 px-3 py-2 text-[11px] font-black uppercase tracking-[0.12em] text-muted-foreground" onClick={() => onNavigate(panel.id, -1)}>
            <ChevronLeft size={14} /> {language === "es" ? "Prev" : "Prev"}
          </button>
          <div className="flex items-center gap-2">
            <button type="button" data-testid={`panel-rotate-${panel.id}`} className="rounded-full border border-white/10 px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground" onPointerDown={(event) => onStartTransform(panel.id, "rotate", event)}>
              {language === "es" ? "Rotar" : "Rotate"}
            </button>
            <button type="button" data-testid={`panel-tilt-${panel.id}`} className="rounded-full border border-white/10 px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground" onPointerDown={(event) => onStartTransform(panel.id, "perspective", event)}>
              {language === "es" ? "Tilt" : "Tilt"}
            </button>
            <button type="button" data-testid={`panel-skew-${panel.id}`} className="rounded-full border border-white/10 px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground" onPointerDown={(event) => onStartTransform(panel.id, "skew", event)}>
              {language === "es" ? "Skew" : "Skew"}
            </button>
          </div>
          <button type="button" data-testid={`panel-next-${panel.id}`} className="inline-flex items-center gap-2 rounded-full border border-white/10 px-3 py-2 text-[11px] font-black uppercase tracking-[0.12em] text-muted-foreground" onClick={() => onNavigate(panel.id, 1)}>
            {language === "es" ? "Next" : "Next"} <ChevronRight size={14} />
          </button>
        </div>
      </div>
      <div data-testid={`panel-scale-${panel.id}`} className="absolute bottom-3 right-3 flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-black/30 text-[10px] font-black uppercase tracking-[0.12em] text-foreground" onPointerDown={(event) => onStartTransform(panel.id, "scale", event)}>
        {language === "es" ? "Scale" : "Scale"}
      </div>
    </div>
  );
};

export default PresentationPanel;
