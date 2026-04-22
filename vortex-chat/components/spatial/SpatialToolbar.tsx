import React from "react";
import {
  Camera,
  Crosshair,
  FileText,
  Globe,
  Hand,
  Image,
  Layers3,
  PenTool,
  Save,
  ScanSearch,
  StickyNote,
  Volume2,
} from "lucide-react";

type SpatialToolbarProps = {
  language: "es" | "en";
  cameraEnabled: boolean;
  gestureEnabled: boolean;
  selectionMode: boolean;
  perspectiveMode: boolean;
  voiceEnabled: boolean;
  onToggleCamera: () => void;
  onToggleGestures: () => void;
  onToggleSelectionMode: () => void;
  onTogglePerspectiveMode: () => void;
  onOpenNotePanel: () => void;
  onOpenPresentationPanel: () => void;
  onOpenBrowserPanel: () => void;
  onOpenPdfPanel: () => void;
  onOpenImagePanel: () => void;
  onOpenSketchPanel: () => void;
  onSaveToObsidian: () => void;
};

const pillClass = (active: boolean) =>
  `inline-flex items-center gap-2 rounded-full border px-3 py-2 text-[11px] font-black uppercase tracking-[0.12em] transition-all ${
    active
      ? "border-primary/40 bg-primary/[0.14] text-foreground"
      : "border-border/70 bg-background/85 text-muted-foreground hover:border-primary/20 hover:text-foreground"
  }`;

const SpatialToolbar: React.FC<SpatialToolbarProps> = ({
  language,
  cameraEnabled,
  gestureEnabled,
  selectionMode,
  perspectiveMode,
  voiceEnabled,
  onToggleCamera,
  onToggleGestures,
  onToggleSelectionMode,
  onTogglePerspectiveMode,
  onOpenNotePanel,
  onOpenPresentationPanel,
  onOpenBrowserPanel,
  onOpenPdfPanel,
  onOpenImagePanel,
  onOpenSketchPanel,
  onSaveToObsidian,
}) => (
  <div className="flex flex-wrap items-center gap-2">
    <button type="button" data-testid="spatial-toggle-camera" onClick={onToggleCamera} className={pillClass(cameraEnabled)}>
      <Camera size={14} /> {language === "es" ? "Camara" : "Camera"}
    </button>
    <button type="button" data-testid="spatial-toggle-gestures" onClick={onToggleGestures} className={pillClass(gestureEnabled)}>
      <Hand size={14} /> {language === "es" ? "Gestos" : "Gestures"}
    </button>
    <button type="button" data-testid="spatial-toggle-region" onClick={onToggleSelectionMode} className={pillClass(selectionMode)}>
      <Crosshair size={14} /> {language === "es" ? "Region" : "Region"}
    </button>
    <button type="button" data-testid="spatial-toggle-tilt-mode" onClick={onTogglePerspectiveMode} className={pillClass(perspectiveMode)}>
      <ScanSearch size={14} /> {language === "es" ? "Tilt" : "Tilt"}
    </button>
    <button type="button" data-testid="spatial-open-note" onClick={onOpenNotePanel} className={pillClass(false)}>
      <StickyNote size={14} /> {language === "es" ? "Nota" : "Note"}
    </button>
    <button type="button" data-testid="spatial-open-presentation" onClick={onOpenPresentationPanel} className={pillClass(false)}>
      <Layers3 size={14} /> {language === "es" ? "Presentacion" : "Presentation"}
    </button>
    <button type="button" data-testid="spatial-open-browser" onClick={onOpenBrowserPanel} className={pillClass(false)}>
      <Globe size={14} /> Browser
    </button>
    <button type="button" data-testid="spatial-open-pdf" onClick={onOpenPdfPanel} className={pillClass(false)}>
      <FileText size={14} /> PDF
    </button>
    <button type="button" data-testid="spatial-open-image" onClick={onOpenImagePanel} className={pillClass(false)}>
      <Image size={14} /> Image
    </button>
    <button type="button" data-testid="spatial-open-sketch" onClick={onOpenSketchPanel} className={pillClass(false)}>
      <PenTool size={14} /> Sketch
    </button>
    <button type="button" data-testid="spatial-save-obsidian" onClick={onSaveToObsidian} className={pillClass(false)}>
      <Save size={14} /> Obsidian
    </button>
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 text-[11px] font-black uppercase tracking-[0.12em] ${
      voiceEnabled ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-200" : "border-border/70 bg-background/85 text-muted-foreground"
    }`}>
      <Volume2 size={14} /> {voiceEnabled ? (language === "es" ? "Voz lista" : "Voice ready") : (language === "es" ? "Voz fallback" : "Voice fallback")}
    </span>
  </div>
);

export default SpatialToolbar;
