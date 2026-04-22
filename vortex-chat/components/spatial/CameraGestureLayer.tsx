import React, { useEffect, useRef, useState } from "react";

export type GestureSignal = {
  kind: "gesture";
  gesture: string;
  confidence: number;
  ts: number;
  trackerMode: string;
  position?: { x: number; y: number };
  secondaryPosition?: { x: number; y: number };
  deltaX?: number;
  deltaY?: number;
  spreadDelta?: number;
  twistDelta?: number;
};

type CameraGestureLayerProps = {
  enabled: boolean;
  gestureEnabled: boolean;
  language: "es" | "en";
  modelAssetPath?: string;
  onGesture: (event: GestureSignal) => void;
  onStatusChange?: (payload: { cameraReady: boolean; trackerMode: string; error?: string | null }) => void;
};

const distance = (left: { x: number; y: number }, right: { x: number; y: number }) =>
  Math.hypot(left.x - right.x, left.y - right.y);

const CameraGestureLayer: React.FC<CameraGestureLayerProps> = ({
  enabled,
  gestureEnabled,
  language,
  modelAssetPath,
  onGesture,
  onStatusChange,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const handLandmarkerRef = useRef<any>(null);
  const lastGestureRef = useRef<string>("");
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const pinchActiveRef = useRef(false);
  const pointerDragRef = useRef<{ active: boolean; lastX: number; lastY: number } | null>(null);
  const dwellRef = useRef<{ x: number; y: number; startedAt: number } | null>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [trackerMode, setTrackerMode] = useState("simulated");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    onStatusChange?.({ cameraReady, trackerMode, error });
  }, [cameraReady, error, onStatusChange, trackerMode]);

  useEffect(() => {
    const startCamera = async () => {
      if (!enabled) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 960 },
            height: { ideal: 540 },
            facingMode: "user",
          },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
        setCameraReady(true);
        setError(null);
      } catch (cameraError) {
        setCameraReady(false);
        setError(cameraError instanceof Error ? cameraError.message : "camera_error");
      }
    };

    const stopCamera = () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (animationRef.current) window.cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
      handLandmarkerRef.current = null;
      setCameraReady(false);
    };

    void startCamera();
    return stopCamera;
  }, [enabled]);

  useEffect(() => {
    const loadTracker = async () => {
      if (!enabled || !gestureEnabled || !cameraReady || !videoRef.current) return;
      try {
        const visionModule: any = await import("@mediapipe/tasks-vision");
        const fileset = await visionModule.FilesetResolver.forVisionTasks(
          "/mediapipe/wasm",
        );
        const localModel = modelAssetPath || "/models/hand_landmarker.task";
        let landmarker: any;
        let modeLabel = "mediapipe-local";
        try {
          landmarker = await visionModule.HandLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: localModel },
            runningMode: "VIDEO",
            numHands: 2,
          });
        } catch {
          landmarker = await visionModule.HandLandmarker.createFromOptions(fileset, {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            },
            runningMode: "VIDEO",
            numHands: 2,
          });
          modeLabel = "mediapipe-remote-fallback";
        }
        handLandmarkerRef.current = landmarker;
        setTrackerMode(modeLabel);
        setError(null);
      } catch (trackerError) {
        handLandmarkerRef.current = null;
        setTrackerMode("simulated");
        setError(trackerError instanceof Error ? trackerError.message : "tracker_error");
      }
    };
    void loadTracker();
  }, [cameraReady, enabled, gestureEnabled, modelAssetPath]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !enabled || !gestureEnabled) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const emitGesture = (event: GestureSignal) => {
      lastGestureRef.current = event.gesture;
      onGesture(event);
    };

    const render = () => {
      if (!canvasRef.current || !videoRef.current) return;
      const nextCanvas = canvasRef.current;
      const nextVideo = videoRef.current;
      nextCanvas.width = nextVideo.videoWidth || 960;
      nextCanvas.height = nextVideo.videoHeight || 540;
      ctx.clearRect(0, 0, nextCanvas.width, nextCanvas.height);
      const tracker = handLandmarkerRef.current;
      if (tracker && nextVideo.readyState >= 2) {
        const results = tracker.detectForVideo(nextVideo, performance.now());
        const landmarks = results?.landmarks || [];
        if (landmarks.length > 0) {
          const primary = landmarks[0];
          const wrist = primary[0];
          const thumb = primary[4];
          const index = primary[8];
          const middle = primary[12];
          const ring = primary[16];
          const pinky = primary[20];
          const point = { x: index.x, y: index.y };
          ctx.fillStyle = "rgba(0,174,255,0.9)";
          ctx.beginPath();
          ctx.arc(point.x * nextCanvas.width, point.y * nextCanvas.height, 10, 0, Math.PI * 2);
          ctx.fill();
          const pinchDistance = distance(thumb, index);
          const openness = (
            distance(wrist, index) +
            distance(wrist, middle) +
            distance(wrist, ring) +
            distance(wrist, pinky)
          ) / 4;
          const lastPoint = lastPointRef.current;
          const deltaX = lastPoint ? point.x - lastPoint.x : 0;
          const deltaY = lastPoint ? point.y - lastPoint.y : 0;
          lastPointRef.current = point;
          const dwell = dwellRef.current;
          if (!dwell || Math.abs(dwell.x - point.x) > 0.02 || Math.abs(dwell.y - point.y) > 0.02) {
            dwellRef.current = { x: point.x, y: point.y, startedAt: Date.now() };
          } else if (Date.now() - dwell.startedAt > 700 && !pinchActiveRef.current) {
            emitGesture({
              kind: "gesture",
              gesture: "dwell",
              confidence: 0.72,
              ts: Date.now(),
              trackerMode,
              position: point,
            });
            dwellRef.current = { x: point.x, y: point.y, startedAt: Date.now() + 999999 };
          }

          if (pinchDistance < 0.055) {
            emitGesture({
              kind: "gesture",
              gesture: pinchActiveRef.current ? "pinch_hold" : "pinch_start",
              confidence: 0.92,
              ts: Date.now(),
              trackerMode,
              position: point,
              deltaX,
              deltaY,
            });
            pinchActiveRef.current = true;
          } else if (pinchActiveRef.current) {
            emitGesture({
              kind: "gesture",
              gesture: "pinch_release",
              confidence: 0.82,
              ts: Date.now(),
              trackerMode,
              position: point,
            });
            pinchActiveRef.current = false;
            dwellRef.current = { x: point.x, y: point.y, startedAt: Date.now() };
          } else if (openness > 0.68) {
            emitGesture({
              kind: "gesture",
              gesture: "open_palm",
              confidence: 0.78,
              ts: Date.now(),
              trackerMode,
              position: point,
            });
          } else if (openness < 0.28) {
            emitGesture({
              kind: "gesture",
              gesture: "fist",
              confidence: 0.72,
              ts: Date.now(),
              trackerMode,
              position: point,
            });
          } else {
            emitGesture({
              kind: "gesture",
              gesture: Math.abs(deltaX) > 0.08 ? (deltaX > 0 ? "swipe_right" : "swipe_left") : "point",
              confidence: 0.7,
              ts: Date.now(),
              trackerMode,
              position: point,
              deltaX,
              deltaY,
            });
          }

          if (landmarks.length >= 2) {
            const secondary = landmarks[1][8];
            const spread = distance(point, secondary);
            const previous = lastPointRef.current;
            emitGesture({
              kind: "gesture",
              gesture: spread > 0.35 ? "two_hand_spread" : "two_hand_pinch",
              confidence: 0.74,
              ts: Date.now(),
              trackerMode,
              position: point,
              secondaryPosition: { x: secondary.x, y: secondary.y },
              spreadDelta: previous ? spread - distance(previous, secondary) : 0,
              twistDelta: deltaX * 24,
            });
          }
        }
      }
      animationRef.current = window.requestAnimationFrame(render);
    };

    animationRef.current = window.requestAnimationFrame(render);
    return () => {
      if (animationRef.current) window.cancelAnimationFrame(animationRef.current);
    };
  }, [cameraReady, enabled, gestureEnabled, onGesture, trackerMode]);

  return (
    <div
      data-testid="camera-gesture-layer"
      aria-label="camera-gesture-layer"
      className="pointer-events-auto absolute right-5 top-5 z-10 h-[220px] w-[320px] overflow-hidden rounded-[1.4rem] border border-border/70 bg-background/65 shadow-[0_30px_70px_-45px_rgba(0,0,0,0.95)] backdrop-blur-xl"
      onPointerDown={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        pointerDragRef.current = { active: true, lastX: event.clientX, lastY: event.clientY };
        onGesture({
          kind: "gesture",
          gesture: "pinch_start",
          confidence: 0.95,
          ts: Date.now(),
          trackerMode: "simulated",
          position: { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height },
        });
      }}
      onPointerMove={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const position = { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height };
        const drag = pointerDragRef.current;
        if (drag?.active) {
          onGesture({
            kind: "gesture",
            gesture: "pinch_hold",
            confidence: 0.95,
            ts: Date.now(),
            trackerMode: "simulated",
            position,
            deltaX: (event.clientX - drag.lastX) / rect.width,
            deltaY: (event.clientY - drag.lastY) / rect.height,
          });
          pointerDragRef.current = { active: true, lastX: event.clientX, lastY: event.clientY };
        } else {
          onGesture({
            kind: "gesture",
            gesture: "point",
            confidence: 0.68,
            ts: Date.now(),
            trackerMode: "simulated",
            position,
          });
        }
      }}
      onPointerUp={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        pointerDragRef.current = null;
        onGesture({
          kind: "gesture",
          gesture: "pinch_release",
          confidence: 0.92,
          ts: Date.now(),
          trackerMode: "simulated",
          position: { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height },
        });
      }}
      onWheel={(event) => {
        onGesture({
          kind: "gesture",
          gesture: event.deltaY < 0 ? "two_hand_spread" : "two_hand_pinch",
          confidence: 0.8,
          ts: Date.now(),
          trackerMode: "simulated",
          spreadDelta: event.deltaY < 0 ? 0.08 : -0.08,
        });
      }}
      tabIndex={0}
      onKeyDown={(event) => {
        const map: Record<string, string> = {
          ArrowLeft: "swipe_left",
          ArrowRight: "swipe_right",
          Escape: "cancel",
          " ": "open_palm",
          Enter: "fist",
          r: "twist",
          p: "perspective_mode_trigger",
        };
        const gesture = map[event.key];
        if (!gesture) return;
        onGesture({
          kind: "gesture",
          gesture,
          confidence: 0.88,
          ts: Date.now(),
          trackerMode: "simulated",
          twistDelta: gesture === "twist" ? 12 : undefined,
        });
      }}
    >
      <video ref={videoRef} muted playsInline className="h-full w-full object-cover opacity-75" data-testid="camera-video" />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/75 to-transparent px-4 pb-4 pt-12 text-left text-xs">
        <p className="font-black uppercase tracking-[0.14em] text-primary">{trackerMode}</p>
        <p className="mt-1 text-[11px] leading-5 text-white/80">
          {error
            ? `${language === "es" ? "Fallback activo" : "Fallback active"}: ${error}`
            : (language === "es" ? "Click/drag, rueda o flechas para simular gestos." : "Click/drag, wheel, or arrows to simulate gestures.")}
        </p>
      </div>
    </div>
  );
};

export default CameraGestureLayer;
