import React, { useEffect, useMemo, useRef, useState } from "react";
import { SpatialPanelModel, SpatialRegion, SpatialSessionState } from "../../types";
import PresentationPanel from "./PresentationPanel";
import SpatialPanel from "./SpatialPanel";

type TransformMode = "move" | "scale" | "rotate" | "perspective" | "skew";

type TransformCanvasProps = {
  language: "es" | "en";
  session: SpatialSessionState;
  focusedPanelId: string | null;
  selectionMode: boolean;
  onFocusPanel: (panelId: string | null) => void;
  onPreviewPanelUpdate: (panelId: string, patch: Partial<SpatialPanelModel>) => void;
  onCommitPanelUpdate: (panelId: string, patch: Partial<SpatialPanelModel>) => void;
  onNavigatePanel: (panelId: string, delta: number) => void;
  onRegionChange: (region: SpatialRegion | null) => void;
};

type PointerOp =
  | {
      kind: "panel";
      panelId: string;
      mode: TransformMode;
      pointerId: number;
      startX: number;
      startY: number;
      startTransform: SpatialPanelModel["transform"];
    }
  | {
      kind: "region";
      pointerId: number;
      startX: number;
      startY: number;
      currentX: number;
      currentY: number;
    };

const buildRegion = (startX: number, startY: number, currentX: number, currentY: number): SpatialRegion => ({
  x: Math.min(startX, currentX),
  y: Math.min(startY, currentY),
  width: Math.abs(currentX - startX),
  height: Math.abs(currentY - startY),
});

const TransformCanvas: React.FC<TransformCanvasProps> = ({
  language,
  session,
  focusedPanelId,
  selectionMode,
  onFocusPanel,
  onPreviewPanelUpdate,
  onCommitPanelUpdate,
  onNavigatePanel,
  onRegionChange,
}) => {
  const stageRef = useRef<HTMLDivElement>(null);
  const sessionRef = useRef(session);
  const opRef = useRef<PointerOp | null>(null);
  const [draftRegion, setDraftRegion] = useState<SpatialRegion | null>(null);

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  const getStagePoint = (clientX: number, clientY: number) => {
    const rect = stageRef.current?.getBoundingClientRect();
    if (!rect) return { x: clientX, y: clientY };
    return { x: clientX - rect.left, y: clientY - rect.top };
  };

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      const op = opRef.current;
      if (!op) return;
      const point = getStagePoint(event.clientX, event.clientY);
      if (op.kind === "region") {
        const region = buildRegion(op.startX, op.startY, point.x, point.y);
        opRef.current = { ...op, currentX: point.x, currentY: point.y };
        setDraftRegion(region);
        return;
      }
      const panel = sessionRef.current.panels.find((item) => item.id === op.panelId);
      if (!panel) return;
      const dx = point.x - op.startX;
      const dy = point.y - op.startY;
      if (op.mode === "move") {
        onPreviewPanelUpdate(panel.id, {
          transform: {
            ...panel.transform,
            ...op.startTransform,
            x: op.startTransform.x + dx,
            y: op.startTransform.y + dy,
          },
        } as Partial<SpatialPanelModel>);
      } else if (op.mode === "scale") {
        const nextScale = Math.max(0.2, op.startTransform.scale + (dx + dy) / 360);
        onPreviewPanelUpdate(panel.id, {
          transform: {
            ...panel.transform,
            ...op.startTransform,
            scale: nextScale,
          },
        } as Partial<SpatialPanelModel>);
      } else if (op.mode === "rotate") {
        onPreviewPanelUpdate(panel.id, {
          transform: {
            ...panel.transform,
            ...op.startTransform,
            rotation: op.startTransform.rotation + dx * 0.25,
          },
        } as Partial<SpatialPanelModel>);
      } else if (op.mode === "perspective") {
        onPreviewPanelUpdate(panel.id, {
          transform: {
            ...panel.transform,
            ...op.startTransform,
            tilt_y: op.startTransform.tilt_y + dx * 0.08,
            tilt_x: op.startTransform.tilt_x - dy * 0.08,
          },
        } as Partial<SpatialPanelModel>);
      } else if (op.mode === "skew") {
        onPreviewPanelUpdate(panel.id, {
          transform: {
            ...panel.transform,
            ...op.startTransform,
            skew_x: op.startTransform.skew_x + dx * 0.08,
            skew_y: op.startTransform.skew_y + dy * 0.08,
          },
        } as Partial<SpatialPanelModel>);
      }
    };

    const handlePointerUp = () => {
      const op = opRef.current;
      if (!op) return;
      if (op.kind === "region") {
        const region = buildRegion(op.startX, op.startY, op.currentX, op.currentY);
        onRegionChange(region.width > 10 && region.height > 10 ? region : null);
        setDraftRegion(null);
      } else {
        const panel = sessionRef.current.panels.find((item) => item.id === op.panelId);
        if (panel) {
          onCommitPanelUpdate(panel.id, { transform: panel.transform } as Partial<SpatialPanelModel>);
        }
      }
      opRef.current = null;
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [onCommitPanelUpdate, onPreviewPanelUpdate, onRegionChange]);

  const beginTransform = (panelId: string, mode: TransformMode, event: React.PointerEvent<HTMLDivElement>) => {
    const panel = sessionRef.current.panels.find((item) => item.id === panelId);
    if (!panel) return;
    const point = getStagePoint(event.clientX, event.clientY);
    opRef.current = {
      kind: "panel",
      panelId,
      mode,
      pointerId: event.pointerId,
      startX: point.x,
      startY: point.y,
      startTransform: { ...panel.transform },
    };
    onFocusPanel(panelId);
  };

  const stagePanels = useMemo(() => (
    [...session.panels].sort((left, right) => {
      const leftFocused = left.id === focusedPanelId ? 1 : 0;
      const rightFocused = right.id === focusedPanelId ? 1 : 0;
      if (leftFocused !== rightFocused) return leftFocused - rightFocused;
      return left.updated_at - right.updated_at;
    })
  ), [focusedPanelId, session.panels]);

  return (
    <div
      ref={stageRef}
      data-testid="spatial-stage"
      className="spatial-stage relative h-full min-h-[620px] overflow-hidden rounded-[1.8rem] border border-border/70 bg-[radial-gradient(circle_at_top,rgba(0,174,255,0.12),transparent_34%),linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01))]"
      onPointerDown={(event) => {
        if (selectionMode) {
          const point = getStagePoint(event.clientX, event.clientY);
          opRef.current = {
            kind: "region",
            pointerId: event.pointerId,
            startX: point.x,
            startY: point.y,
            currentX: point.x,
            currentY: point.y,
          };
          setDraftRegion({ x: point.x, y: point.y, width: 0, height: 0 });
          return;
        }
        onFocusPanel(null);
      }}
    >
      <div className="absolute inset-0 spatial-grid" />

      {stagePanels.map((panel) =>
        panel.type === "presentation" ? (
          <PresentationPanel
            key={panel.id}
            panel={panel}
            focused={focusedPanelId === panel.id}
            language={language}
            onFocus={onFocusPanel}
            onStartTransform={beginTransform}
            onNavigate={onNavigatePanel}
          />
        ) : (
          <SpatialPanel
            key={panel.id}
            panel={panel}
            focused={focusedPanelId === panel.id}
            language={language}
            onFocus={onFocusPanel}
            onStartTransform={beginTransform}
          />
        ),
      )}

      {(draftRegion || session.selected_region) ? (
        <div
          className="pointer-events-none absolute rounded-[1.2rem] border border-primary/70 bg-primary/[0.08] shadow-[0_0_0_1px_rgba(0,174,255,0.2)]"
          style={{
            left: `${(draftRegion || session.selected_region)?.x || 0}px`,
            top: `${(draftRegion || session.selected_region)?.y || 0}px`,
            width: `${(draftRegion || session.selected_region)?.width || 0}px`,
            height: `${(draftRegion || session.selected_region)?.height || 0}px`,
          }}
        >
          <span className="absolute -top-7 left-0 rounded-full border border-primary/30 bg-background/90 px-3 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-primary">
            {language === "es" ? "Drop zone" : "Drop zone"}
          </span>
        </div>
      ) : null}
    </div>
  );
};

export default TransformCanvas;
