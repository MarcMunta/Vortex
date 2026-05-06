import { useEffect, useState } from "react";
import { controlService } from "../services/controlService";
import { vortexService } from "../services/vortexService";
import { ControlStatus, LogEntry, OperationalStatus } from "../types";

type UseSystemStatusArgs = {
  addLog: (level: LogEntry["level"], message: string) => void;
  language: "es" | "en";
};

export const useSystemStatus = ({ addLog, language }: UseSystemStatusArgs) => {
  const [operationalStatus, setOperationalStatus] = useState<OperationalStatus | null>(null);
  const [controlStatus, setControlStatus] = useState<ControlStatus | null>(null);
  const [selfEditsPendingCount, setSelfEditsPendingCount] = useState(0);

  useEffect(() => {
    let disposed = false;

    const pollStatus = async () => {
      const [runtimeStatus, nextControlStatus] = await Promise.all([
        vortexService.fetchOperationalStatus(),
        controlService.fetchStatus(),
      ]);
      if (disposed) return;
      setOperationalStatus(runtimeStatus);
      setControlStatus(nextControlStatus);
    };

    void pollStatus();
    const timer = window.setInterval(pollStatus, 5000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let disposed = false;
    let prevEpisodes = -1;
    let prevRequests = -1;
    let prevKnowledge = -1;
    let prevBackendKey = "";
    let prevWebChunks = -1;
    let prevCodeChunks = -1;
    let prevAnalyses = -1;
    let prevProposals = -1;
    let prevDiscoveredUrls = -1;

    const poll = async () => {
      try {
        const data = await vortexService.fetchOperationalStatus();
        if (!data || disposed) return;

        const statusData = data as any;
        const backends = statusData.backends || [];
        const adaptersLoaded = statusData.adapters
          ? Object.values(statusData.adapters).filter(Boolean).length
          : 0;
        const metrics = statusData.metrics || {};
        const episodes = statusData.episodes || 0;
        const knowledge = statusData.knowledge_chunks || 0;
        const autolearn = statusData.autolearn || {};

        const backendKey = `${backends.join(",")}|${adaptersLoaded}`;
        if (backendKey !== prevBackendKey) {
          prevBackendKey = backendKey;
          if (adaptersLoaded > 0) {
            addLog("LEARN", language === "es"
              ? `Adaptadores LoRA activos: ${adaptersLoaded} en ${backends.join(", ")}.`
              : `Active LoRA adapters: ${adaptersLoaded} on ${backends.join(", ")}.`);
          } else if (backends.length > 0) {
            addLog("INFO", language === "es"
              ? `Backend: ${backends.join(", ")} — modelo cargado y listo.`
              : `Backend: ${backends.join(", ")} — model loaded and ready.`);
          }
        }

        if (prevEpisodes >= 0 && episodes > prevEpisodes) {
          const diff = episodes - prevEpisodes;
          addLog("LEARN", language === "es"
            ? `+${diff} episodio${diff > 1 ? "s" : ""} registrado${diff > 1 ? "s" : ""} (total: ${episodes}).`
            : `+${diff} new episode${diff > 1 ? "s" : ""} logged (total: ${episodes}).`);
        } else if (prevEpisodes < 0 && episodes > 0) {
          addLog("INFO", language === "es"
            ? `Episodios almacenados: ${episodes}.`
            : `Stored episodes: ${episodes}.`);
        }
        prevEpisodes = episodes;

        if (prevKnowledge >= 0 && knowledge > prevKnowledge) {
          const diff = knowledge - prevKnowledge;
          addLog("LEARN", language === "es"
            ? `+${diff} chunk${diff > 1 ? "s" : ""} de conocimiento indexado${diff > 1 ? "s" : ""} (total: ${knowledge}).`
            : `+${diff} knowledge chunk${diff > 1 ? "s" : ""} indexed (total: ${knowledge}).`);
        } else if (prevKnowledge < 0 && knowledge > 0) {
          addLog("INFO", language === "es"
            ? `Base de conocimiento: ${knowledge} chunks indexados.`
            : `Knowledge base: ${knowledge} chunks indexed.`);
        }
        prevKnowledge = knowledge;

        const webChunks = autolearn.total_web_chunks || 0;
        if (prevWebChunks >= 0 && webChunks > prevWebChunks) {
          const diff = webChunks - prevWebChunks;
          addLog("SEARCH", language === "es"
            ? `Autolearn: +${diff} fragmentos web ingestados (total: ${webChunks}).`
            : `Autolearn: +${diff} web chunks ingested (total: ${webChunks}).`);
        } else if (prevWebChunks < 0 && webChunks > 0) {
          addLog("SEARCH", language === "es"
            ? `Autolearn: ${webChunks} fragmentos web en base de conocimiento.`
            : `Autolearn: ${webChunks} web chunks in knowledge base.`);
        }
        prevWebChunks = webChunks;

        const codeChunks = autolearn.total_code_chunks || 0;
        if (prevCodeChunks >= 0 && codeChunks > prevCodeChunks) {
          const diff = codeChunks - prevCodeChunks;
          addLog("LEARN", language === "es"
            ? `Autolearn: +${diff} fragmentos de código propio indexados.`
            : `Autolearn: +${diff} self-code chunks indexed.`);
        } else if (prevCodeChunks < 0 && codeChunks > 0) {
          addLog("LEARN", language === "es"
            ? `Autolearn: ${codeChunks} fragmentos de código propio indexados.`
            : `Autolearn: ${codeChunks} self-code chunks indexed.`);
        }
        prevCodeChunks = codeChunks;

        const analyses = autolearn.total_analyses || 0;
        const proposals = autolearn.total_proposals || 0;
        if (prevAnalyses >= 0 && analyses > prevAnalyses) {
          const diff = analyses - prevAnalyses;
          addLog("LEARN", language === "es"
            ? `Autolearn: ${diff} archivo${diff > 1 ? "s" : ""} analizado${diff > 1 ? "s" : ""} — ${proposals} propuestas generadas.`
            : `Autolearn: ${diff} file${diff > 1 ? "s" : ""} analyzed — ${proposals} proposals generated.`);
        }
        prevAnalyses = analyses;
        if (prevProposals >= 0 && proposals > prevProposals) {
          const diff = proposals - prevProposals;
          addLog("SYSTEM", language === "es"
            ? `Autolearn: +${diff} propuesta${diff > 1 ? "s" : ""} de auto-mejora generada${diff > 1 ? "s" : ""}.`
            : `Autolearn: +${diff} self-improvement proposal${diff > 1 ? "s" : ""} generated.`);
        }
        prevProposals = proposals;

        const discoveredUrls = (autolearn.discovered_urls || []).length;
        if (prevDiscoveredUrls >= 0 && discoveredUrls > prevDiscoveredUrls) {
          const diff = discoveredUrls - prevDiscoveredUrls;
          addLog("SEARCH", language === "es"
            ? `Autolearn: +${diff} URL${diff > 1 ? "s" : ""} descubierta${diff > 1 ? "s" : ""} por el modelo (total: ${discoveredUrls}).`
            : `Autolearn: +${diff} URL${diff > 1 ? "s" : ""} discovered by model (total: ${discoveredUrls}).`);
        } else if (prevDiscoveredUrls < 0 && discoveredUrls > 0) {
          addLog("SEARCH", language === "es"
            ? `Autolearn: ${discoveredUrls} URLs descubiertas para aprendizaje autónomo.`
            : `Autolearn: ${discoveredUrls} URLs discovered for autonomous learning.`);
        }
        prevDiscoveredUrls = discoveredUrls;

        if (prevRequests >= 0 && metrics.chat_requests > prevRequests) {
          const diff = metrics.chat_requests - prevRequests;
          const latency = metrics.avg_latency_ms || 0;
          const tokens = metrics.completion_tokens_est || 0;
          addLog("INFO", language === "es"
            ? `+${diff} petición${diff > 1 ? "es" : ""} procesada${diff > 1 ? "s" : ""} — latencia media: ${latency}ms, tokens generados: ${tokens}.`
            : `+${diff} request${diff > 1 ? "s" : ""} processed — avg latency: ${latency}ms, tokens generated: ${tokens}.`);
        }
        prevRequests = metrics.chat_requests || 0;
      } catch {
        // Backend offline.
      }
    };

    const initialTimer = window.setTimeout(poll, 2000);
    const interval = window.setInterval(poll, 20000);
    return () => {
      disposed = true;
      window.clearTimeout(initialTimer);
      window.clearInterval(interval);
    };
  }, [addLog, language]);

  useEffect(() => {
    let disposed = false;

    const fetchPending = async () => {
      try {
        const resp = await fetch("/v1/self-edits/proposals?status=pending");
        if (!resp.ok) return;
        const payload = await resp.json().catch(() => ({}));
        const nextCount = Array.isArray(payload?.data) ? payload.data.length : 0;
        if (!disposed) setSelfEditsPendingCount(nextCount);
      } catch {
        // Ignore poll failures.
      }
    };

    void fetchPending();
    const interval = window.setInterval(fetchPending, 8000);
    return () => {
      disposed = true;
      window.clearInterval(interval);
    };
  }, []);

  return {
    controlStatus,
    operationalStatus,
    selfEditsPendingCount,
    setControlStatus,
    setOperationalStatus,
    setSelfEditsPendingCount,
  };
};
