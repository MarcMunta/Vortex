import { SpatialPanelModel, SpatialRegion, SpatialSessionState } from "../../types";

export const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export const createSpatialPanel = (
  type: SpatialPanelModel["type"],
  title: string,
  content: string,
  region?: SpatialRegion | null,
  source?: Record<string, unknown>,
): SpatialPanelModel => {
  const width = Math.max(region?.width || 340, 280);
  const height = Math.max(region?.height || 240, 220);
  return {
    id: `local-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    type,
    title,
    content,
    source: source || {},
    transform: {
      x: region?.x || 120,
      y: region?.y || 120,
      z: 0,
      scale: 1,
      rotation: 0,
      skew_x: 0,
      skew_y: 0,
      tilt_x: 0,
      tilt_y: 0,
      perspective: 1100,
      width,
      height,
    },
    page_index: 0,
    page_count: Array.isArray(source?.pages) ? source.pages.length : 1,
    created_at: Date.now(),
    updated_at: Date.now(),
  };
};

export const createDefaultSpatialSession = (): SpatialSessionState => ({
  session_id: "default",
  selected_object_id: null,
  selected_region: null,
  active_panel_ids: [],
  active_presentation_id: null,
  active_page_index: 0,
  interaction_mode: "inspect",
  last_voice_command: null,
  last_gesture_event: null,
  camera_state: null,
  gesture_state: null,
  focused_item: null,
  recent_multimodal_summary: null,
  panels: [
    createSpatialPanel("note", "Planner", "Arquitectura multimodal local.\n- voz\n- gestos\n- panels\n- obsidian"),
    createSpatialPanel(
      "presentation",
      "Spatial deck",
      "Slide 1",
      { x: 480, y: 110, width: 430, height: 300 },
      {
        pages: [
          "Slide 1 - Workspace live",
          "Slide 2 - Gesture fusion",
          "Slide 3 - Voice + Obsidian",
        ],
      },
    ),
  ],
  updated_at: Date.now(),
  created_at: Date.now(),
});

export const hitTestSpatialPanels = (
  panels: SpatialPanelModel[],
  point: { x: number; y: number },
): SpatialPanelModel | null => {
  const ordered = [...panels].sort((left, right) => right.updated_at - left.updated_at);
  for (const panel of ordered) {
    const left = panel.transform.x;
    const top = panel.transform.y;
    const right = left + panel.transform.width;
    const bottom = top + panel.transform.height;
    if (point.x >= left && point.x <= right && point.y >= top && point.y <= bottom) {
      return panel;
    }
  }
  return null;
};

export const buildObsidianPreviewPanel = (
  title: string,
  content: string,
  path: string,
  region?: SpatialRegion | null,
): SpatialPanelModel => createSpatialPanel("obsidian", title, content, region, { path });
