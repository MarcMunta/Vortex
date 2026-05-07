import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Wrench, ChevronDown, ChevronRight, CheckCircle2, XCircle } from 'lucide-react';
import type { Language } from '../../types';

interface ToolCallBlockProps {
  tool: string;
  args: Record<string, unknown>;
  result?: { ok: boolean; output: string };
  language: Language;
}

const ToolCallBlock: React.FC<ToolCallBlockProps> = ({ tool, args, result, language }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const toolDisplayNames: Record<string, string> = {
    read_file: language === 'es' ? 'Leer archivo' : 'Read file',
    write_file: language === 'es' ? 'Escribir archivo' : 'Write file',
    delete_file: language === 'es' ? 'Eliminar archivo' : 'Delete file',
    run_command: language === 'es' ? 'Ejecutar comando' : 'Run command',
    run_tests: language === 'es' ? 'Ejecutar tests' : 'Run tests',
    search_web: language === 'es' ? 'Buscar en web' : 'Search web',
    open_docs: language === 'es' ? 'Abrir documentación' : 'Open docs',
    open_browser: language === 'es' ? 'Abrir navegador' : 'Open browser',
    grep: language === 'es' ? 'Buscar en código' : 'Search code',
    list_tree: language === 'es' ? 'Listar archivos' : 'List files',
    propose_patch: language === 'es' ? 'Proponer parche' : 'Propose patch',
    apply_patch: language === 'es' ? 'Aplicar parche' : 'Apply patch',
  };

  const displayName = toolDisplayNames[tool] || tool;
  const argsPreview = Object.entries(args)
    .filter(([, value]) => typeof value === 'string' && String(value).length < 80)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .slice(0, 2)
    .join(' ');

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg border border-zinc-700/40 bg-zinc-800/50 overflow-hidden"
    >
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-3 px-3.5 py-2 cursor-pointer hover:bg-zinc-700/30 transition-colors"
      >
        <Wrench size={12} className="text-primary/60 shrink-0" />
        <span className="text-[11px] font-bold text-foreground/80">{displayName}</span>
        {argsPreview && (
          <span className="text-[10px] font-mono text-zinc-500 truncate flex-1">{argsPreview}</span>
        )}
        {result && (
          result.ok
            ? <CheckCircle2 size={12} className="text-emerald-400 shrink-0" />
            : <XCircle size={12} className="text-red-400 shrink-0" />
        )}
        <div className="text-zinc-600 shrink-0">
          {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </div>
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden border-t border-zinc-700/30"
          >
            <div className="px-3.5 py-2.5 space-y-2">
              {/* Args */}
              <div>
                <div className="text-[9px] font-black uppercase tracking-[0.2em] text-zinc-600 mb-1">
                  {language === 'es' ? 'Argumentos' : 'Arguments'}
                </div>
                <pre className="text-[10px] font-mono text-zinc-400 bg-black/20 rounded-md px-3 py-2 overflow-x-auto custom-scrollbar max-h-[120px] overflow-y-auto">
                  {JSON.stringify(args, null, 2)}
                </pre>
              </div>

              {/* Result */}
              {result && (
                <div>
                  <div className="text-[9px] font-black uppercase tracking-[0.2em] text-zinc-600 mb-1">
                    {language === 'es' ? 'Resultado' : 'Result'}
                  </div>
                  <pre className={`text-[10px] font-mono rounded-md px-3 py-2 overflow-x-auto custom-scrollbar max-h-[150px] overflow-y-auto ${
                    result.ok ? 'text-emerald-400/80 bg-emerald-500/5' : 'text-red-400/80 bg-red-500/5'
                  }`}>
                    {result.output || (result.ok ? 'OK' : 'Error')}
                  </pre>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default ToolCallBlock;
