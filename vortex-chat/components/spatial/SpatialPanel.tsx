import React from "react";
import { SpatialPanelModel } from "../../types";
import { buildPanelMatrix } from "./matrix";

type TransformMode = "move" | "scale" | "rotate" | "perspective" | "skew";

type SpatialPanelProps = {
  panel: SpatialPanelModel;
  focused: boolean;
  language: "es" | "en";
  onFocus: (panelId: string) => void;
  onStartTransform: (panelId: string, mode: TransformMode, event: React.PointerEvent<HTMLDivElement>) => void;
};

const SpatialPanel: React.FC<SpatialPanelProps> = ({
  panel,
  focused,
  language,
  onFocus,
  onStartTransform,
}) => {
  const source = panel.source || {};
  const iframeUrl = typeof source.url === "string" ? source.url : null;
  const imageUrl = typeof source.imageUrl === "string" ? source.imageUrl : null;
  const notePath = typeof source.path === "string" ? source.path : null;

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
      <div
        data-testid={`panel-header-${panel.id}`}
        className="flex items-center justify-between gap-3 border-b border-white/10 px-4 py-3"
        onPointerDown={(event) => onStartTransform(panel.id, "move", event)}
      >
        <div className="min-w-0">
          <p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">{panel.type}</p>
          <p className="truncate text-sm font-bold tracking-tight text-foreground">{panel.title}</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            data-testid={`panel-rotate-${panel.id}`}
            className="rounded-full border border-white/10 px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground"
            onPointerDown={(event) => onStartTransform(panel.id, "rotate", event)}
          >
            {language === "es" ? "Rotar" : "Rotate"}
          </button>
          <button
            type="button"
            data-testid={`panel-tilt-${panel.id}`}
            className="rounded-full border border-white/10 px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground"
            onPointerDown={(event) => onStartTransform(panel.id, "perspective", event)}
          >
            {language === "es" ? "Tilt" : "Tilt"}
          </button>
          <button
            type="button"
            data-testid={`panel-skew-${panel.id}`}
            className="rounded-full border border-white/10 px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-muted-foreground"
            onPointerDown={(event) => onStartTransform(panel.id, "skew", event)}
          >
            {language === "es" ? "Skew" : "Skew"}
          </button>
        </div>
      </div>

      <div className="flex h-[calc(100%-61px)] flex-col overflow-hidden px-4 py-4 text-sm text-foreground">
        {panel.type === "browser" && iframeUrl ? (
          <iframe title={panel.title} src={iframeUrl} className="h-full w-full rounded-[1rem] border border-white/10 bg-white" />
        ) : panel.type === "pdf" && iframeUrl ? (
          <iframe title={panel.title} src={iframeUrl} className="h-full w-full rounded-[1rem] border border-white/10 bg-white" />
        ) : panel.type === "pdf" ? (
          <div className="custom-scrollbar h-full overflow-auto rounded-[1rem] border border-white/10 bg-[linear-gradient(180deg,rgba(0,174,255,0.08),transparent_22%),rgba(255,255,255,0.03)] p-4">
            <p className="text-[10px] font-black uppercase tracking-[0.12em] text-primary">PDF</p>
            <p className="mt-3 text-sm leading-6 text-muted-foreground">
              {panel.content || (language === "es" ? "PDF listo para vista local." : "PDF ready for local preview.")}
            </p>
          </div>
        ) : panel.type === "image" && imageUrl ? (
          <img src={imageUrl} alt={panel.title} className="h-full w-full rounded-[1rem] object-cover" />
        ) : panel.type === "obsidian" ? (
          <div className="custom-scrollbar h-full overflow-auto rounded-[1rem] border border-white/10 bg-[linear-gradient(180deg,rgba(0,174,255,0.08),transparent_20%),rgba(255,255,255,0.03)] p-3">
            <p className="text-[10px] font-black uppercase tracking-[0.12em] text-primary">Obsidian</p>
            {notePath ? <p className="mt-2 text-[11px] text-muted-foreground">{notePath}</p> : null}
            <div className="mt-3 whitespace-pre-wrap text-sm leading-6 text-foreground">{panel.content}</div>
          </div>
        ) : panel.type === "sketch" ? (
          <div className="h-full rounded-[1rem] border border-dashed border-white/10 bg-[radial-gradient(circle_at_top,rgba(0,174,255,0.14),transparent_48%),linear-gradient(180deg,rgba(255,255,255,0.05),transparent)]" />
        ) : (
          <div className="custom-scrollbar h-full overflow-auto rounded-[1rem] border border-white/10 bg-white/[0.03] p-3 text-sm leading-6 text-muted-foreground">
            {panel.content || (language === "es" ? "Panel listo para contenido espacial." : "Panel ready for spatial content.")}
          </div>
        )}
      </div>

      <div
        data-testid={`panel-scale-${panel.id}`}
        className="absolute bottom-3 right-3 flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-black/30 text-[10px] font-black uppercase tracking-[0.12em] text-foreground"
        onPointerDown={(event) => onStartTransform(panel.id, "scale", event)}
      >
        {language === "es" ? "Scale" : "Scale"}
      </div>
      <div
        data-testid={`panel-skew-handle-${panel.id}`}
        className="absolute bottom-3 left-3 flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-black/30 text-[10px] font-black uppercase tracking-[0.12em] text-foreground"
        onPointerDown={(event) => onStartTransform(panel.id, "skew", event)}
      >
        Sk
      </div>
    </div>
  );
};

export default SpatialPanel;
