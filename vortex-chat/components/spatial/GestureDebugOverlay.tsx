import React from "react";

type GestureDebugOverlayProps = {
  language: "es" | "en";
  trackerMode: string;
  cameraReady: boolean;
  gestureLabel: string;
  detail?: string | null;
};

const GestureDebugOverlay: React.FC<GestureDebugOverlayProps> = ({
  language,
  trackerMode,
  cameraReady,
  gestureLabel,
  detail,
}) => (
  <div className="pointer-events-none absolute left-5 top-5 z-20 rounded-[1.1rem] border border-border/70 bg-background/80 px-4 py-3 text-left shadow-[0_20px_50px_-38px_rgba(0,0,0,0.75)] backdrop-blur-xl">
    <p className="text-[10px] font-black uppercase tracking-[0.16em] text-primary">
      {language === "es" ? "Gesture debug" : "Gesture debug"}
    </p>
    <div className="mt-2 flex items-center gap-2 text-xs text-foreground">
      <span className={`h-2.5 w-2.5 rounded-full ${cameraReady ? "bg-emerald-400" : "bg-amber-400"}`} />
      <span>{cameraReady ? (language === "es" ? "Camara lista" : "Camera ready") : (language === "es" ? "Camara pausada" : "Camera paused")}</span>
    </div>
    <p className="mt-2 text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">{trackerMode}</p>
    <p className="mt-2 text-sm font-bold tracking-tight text-foreground">{gestureLabel}</p>
    {detail ? <p className="mt-2 max-w-[240px] text-xs leading-5 text-muted-foreground">{detail}</p> : null}
  </div>
);

export default GestureDebugOverlay;
