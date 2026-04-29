import React from 'react';
import {
  Activity,
  AlertTriangle,
  Cpu,
  FlaskConical,
  Globe,
  HardDriveDownload,
  RefreshCw,
  ShieldCheck,
} from 'lucide-react';
import { ControlStatus, Language, OperationalStatus } from '../types';

interface LocalStackStatusProps {
  status: OperationalStatus | null;
  controlStatus: ControlStatus | null;
  language: Language;
  onBootstrap?: () => Promise<unknown> | unknown;
  onModelInit?: () => Promise<unknown> | unknown;
  onRestartRuntime?: () => Promise<unknown> | unknown;
  onStartTraining?: () => Promise<unknown> | unknown;
}

const isExternalRuntime = (kind?: string | null): boolean => {
  const normalized = String(kind || '').trim().toLowerCase();
  return normalized === 'external' || normalized === 'sglang' || normalized === 'vllm' || normalized === 'ollama';
};

const runtimeDisplayLabel = (status: OperationalStatus | null, ready: boolean): string => {
  const kind = String(status?.engine_kind || status?.active_backend || '').trim().toLowerCase();
  if ((kind === 'hf' || status?.active_backend === 'hf') && ready) return 'HF LOCAL';
  if (ready && isExternalRuntime(kind) && status?.engine_base_url) return status.engine_base_url;
  return ready ? 'LOCAL' : 'Pendiente';
};

const LocalStackStatus: React.FC<LocalStackStatusProps> = ({
  status,
  controlStatus,
  language,
  onBootstrap,
  onModelInit,
  onRestartRuntime,
  onStartTraining,
}) => {
  const ready = Boolean(status?.ok && status?.engine_ready && status?.model_ready);
  const modelCached = Boolean(controlStatus?.model?.cached);
  const dockerReady = Boolean(controlStatus?.docker?.ready);
  const bootstrapRunning = Boolean(controlStatus?.bootstrap?.running);
  const runtimeUrl = runtimeDisplayLabel(status, ready);
  const modelLabel = status?.active_model || controlStatus?.model?.model_id || 'local model';
  const engineLabel = (status?.engine_kind || status?.active_backend || 'local').toUpperCase();
  const reason = ready
    ? (language === 'es' ? 'Stack local listo para responder y entrenar.' : 'Local stack is ready to answer and train.')
    : status?.degraded_reason
      || status?.engine_reason
      || status?.model_reason
      || controlStatus?.bootstrap?.message
      || controlStatus?.docker?.reason
      || (language === 'es' ? 'Falta completar el arranque local.' : 'Local startup is still incomplete.');

  const statusItems = [
    {
      label: language === 'es' ? 'Runtime' : 'Runtime',
      value: ready ? (language === 'es' ? 'Activo' : 'Live') : (language === 'es' ? 'Pendiente' : 'Pending'),
      caption: engineLabel,
      icon: <Cpu size={14} />,
    },
    {
      label: language === 'es' ? 'Modelo' : 'Model',
      value: modelCached ? (language === 'es' ? 'En caché' : 'Cached') : (language === 'es' ? 'Sin descargar' : 'Not cached'),
      caption: modelLabel,
      icon: <HardDriveDownload size={14} />,
    },
    {
      label: language === 'es' ? 'Internet' : 'Internet',
      value: status?.web_disabled
        ? (language === 'es' ? 'Por prompt' : 'Per prompt')
        : (language === 'es' ? 'Disponible' : 'Available'),
      caption: status?.web_disabled
        ? (language === 'es' ? 'Local por defecto' : 'Local by default')
        : (language === 'es' ? 'Revisar política' : 'Review policy'),
      icon: <Globe size={14} />,
    },
    {
      label: language === 'es' ? 'Entreno' : 'Training',
      value: status?.training_ready
        ? (language === 'es' ? 'Listo' : 'Ready')
        : (language === 'es' ? 'Pendiente' : 'Pending'),
      caption: bootstrapRunning
        ? (controlStatus?.bootstrap?.stage || 'bootstrap')
        : (language === 'es' ? 'Control manual' : 'Manual control'),
      icon: <FlaskConical size={14} />,
    },
  ];

  const ActionButton = ({
    label,
    onClick,
    icon,
    primary = false,
    muted = false,
  }: {
    label: string;
    onClick?: () => Promise<unknown> | unknown;
    icon: React.ReactNode;
    primary?: boolean;
    muted?: boolean;
  }) => (
    <button
      type="button"
      onClick={() => {
        void onClick?.();
      }}
      className={`inline-flex items-center gap-2 rounded-full border px-4 py-2.5 text-[10px] font-black uppercase tracking-[0.22em] transition-all ${
        primary
          ? 'border-primary/25 bg-primary/[0.10] text-primary hover:bg-primary/[0.16]'
          : muted
            ? 'border-border/60 bg-background/65 text-foreground/70 hover:border-primary/25 hover:text-foreground'
            : 'border-border/50 bg-background/80 text-foreground hover:border-primary/25'
      }`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );

  return (
    <section
      className={`mx-auto mb-4 max-w-[980px] rounded-[2rem] border px-5 py-4 shadow-[0_24px_80px_-48px_rgba(0,0,0,0.55)] backdrop-blur-2xl ${
        ready
          ? 'border-emerald-500/20 bg-emerald-500/[0.08]'
          : 'border-primary/20 bg-primary/[0.08]'
      }`}
    >
      <div className="flex flex-col gap-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.24em] text-foreground/75">
              {ready ? <ShieldCheck size={14} /> : <AlertTriangle size={14} />}
              <span>{language === 'es' ? 'Stack local' : 'Local stack'}</span>
              {bootstrapRunning && (
                <span className="rounded-full border border-primary/20 bg-primary/[0.10] px-2 py-1 text-[9px] text-primary">
                  {controlStatus?.bootstrap?.stage || 'bootstrap'}
                </span>
              )}
            </div>
            <p className="text-base font-black tracking-tight text-foreground">
              {ready
                ? (language === 'es' ? 'Vortex operativo en local.' : 'Vortex is operational locally.')
                : (language === 'es' ? 'El chat seguirá bloqueado hasta completar el runtime.' : 'Chat stays blocked until the runtime is ready.')}
            </p>
            <p className="max-w-2xl text-sm leading-7 text-muted-foreground">{reason}</p>
          </div>

          <div className="rounded-[1.5rem] border border-border/50 bg-background/70 px-4 py-3 text-right">
            <div className="flex items-center justify-end gap-2 text-sm font-black tracking-tight text-foreground">
              <Activity size={14} />
              <span>{runtimeUrl}</span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {dockerReady
                ? (language === 'es' ? 'Docker listo' : 'Docker ready')
                : (language === 'es' ? 'Docker pendiente' : 'Docker pending')}
            </p>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-4">
          {statusItems.map((item) => (
            <div key={item.label} className="rounded-[1.45rem] border border-border/50 bg-background/72 px-4 py-3">
              <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.22em] text-muted-foreground">
                {item.icon}
                <span>{item.label}</span>
              </div>
              <p className="mt-3 text-sm font-black tracking-tight text-foreground">{item.value}</p>
              <p className="mt-1 break-words text-xs text-muted-foreground">{item.caption}</p>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <ActionButton
            label={language === 'es' ? 'Descargar modelo' : 'Download model'}
            onClick={onModelInit}
            icon={<HardDriveDownload size={14} />}
            primary={!modelCached}
            muted={modelCached}
          />
          <ActionButton
            label={language === 'es' ? 'Iniciar runtime' : 'Start runtime'}
            onClick={onBootstrap}
            icon={<Activity size={14} />}
            primary
          />
          <ActionButton
            label={language === 'es' ? 'Reiniciar' : 'Restart'}
            onClick={onRestartRuntime}
            icon={<RefreshCw size={14} />}
            muted
          />
          <ActionButton
            label={language === 'es' ? 'Entreno rápido' : 'Quick train'}
            onClick={onStartTraining}
            icon={<FlaskConical size={14} />}
            muted={!controlStatus?.ok}
          />
        </div>
      </div>
    </section>
  );
};

export default LocalStackStatus;
