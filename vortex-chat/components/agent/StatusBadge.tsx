import React from 'react';
import { motion } from 'framer-motion';
import { Loader2, CheckCircle2, XCircle, Ban } from 'lucide-react';
import type { AgentRunStatus, Language } from '../../types';

interface StatusBadgeProps {
  status: AgentRunStatus;
  language: Language;
  elapsedMs?: number;
}

const formatElapsed = (ms: number): string => {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
};

const StatusBadge: React.FC<StatusBadgeProps> = ({ status, language, elapsedMs }) => {
  const config = {
    idle: {
      icon: <div className="w-2 h-2 rounded-full bg-zinc-500" />,
      label: language === 'es' ? 'Inactivo' : 'Idle',
      className: 'bg-zinc-500/10 text-zinc-400 border-zinc-500/20',
    },
    running: {
      icon: <Loader2 size={12} className="animate-spin" />,
      label: language === 'es' ? 'Ejecutando' : 'Running',
      className: 'bg-primary/10 text-primary border-primary/20',
    },
    completed: {
      icon: <CheckCircle2 size={12} />,
      label: language === 'es' ? 'Completado' : 'Completed',
      className: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    },
    failed: {
      icon: <XCircle size={12} />,
      label: language === 'es' ? 'Error' : 'Failed',
      className: 'bg-red-500/10 text-red-400 border-red-500/20',
    },
    cancelled: {
      icon: <Ban size={12} />,
      label: language === 'es' ? 'Cancelado' : 'Cancelled',
      className: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    },
  };

  const { icon, label, className } = config[status];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.16em] ${className}`}
    >
      {icon}
      <span>{label}</span>
      {elapsedMs !== undefined && elapsedMs > 0 && (
        <span className="opacity-60 tabular-nums">{formatElapsed(elapsedMs)}</span>
      )}
    </motion.div>
  );
};

export default StatusBadge;
