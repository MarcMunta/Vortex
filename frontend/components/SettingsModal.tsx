import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { X, GripVertical, Monitor, Moon, Sun, Type, Globe } from 'lucide-react';
import { Reorder, AnimatePresence, motion } from 'framer-motion';
import { UserSettings, FontSize, Language } from '../types';
import { translations } from '../translations';
import { VortexApiError, VortexModelInfo, listModels } from '../services/vortexClient';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: UserSettings;
  onUpdateSettings: (settings: UserSettings) => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, settings, onUpdateSettings }) => {
  const t = translations[settings.language];
  const [availableModels, setAvailableModels] = useState<VortexModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const refreshModels = useCallback(async () => {
    setModelsLoading(true);
    setModelsError(null);
    try {
      const models = await listModels({ baseUrl: settings.llm.baseUrl, token: settings.llm.token });
      setAvailableModels(models);
    } catch (err: any) {
      const msg = err instanceof VortexApiError ? err.message : 'Failed to load models';
      setModelsError(msg);
      setAvailableModels([]);
    } finally {
      setModelsLoading(false);
    }
  }, [settings.llm.baseUrl, settings.llm.token]);

  useEffect(() => {
    if (!isOpen) return;
    refreshModels();
  }, [isOpen, refreshModels]);

  const modelOptions = useMemo(() => {
    const ids = new Set<string>();
    const options = [];
    for (const m of availableModels) {
      if (!m?.id || ids.has(m.id)) continue;
      ids.add(m.id);
      options.push(m);
    }
    return options;
  }, [availableModels]);

  const handleReorder = (newOrder: string[]) => {
    onUpdateSettings({ ...settings, categoryOrder: newOrder });
  };

  const handleCodeThemeChange = (theme: 'dark' | 'light' | 'match-app') => {
    onUpdateSettings({ ...settings, codeTheme: theme });
  };

  const handleFontSizeChange = (size: FontSize) => {
    onUpdateSettings({ ...settings, fontSize: size });
  };

  const handleLanguageChange = (lang: Language) => {
    onUpdateSettings({ ...settings, language: lang });
  };

  const updateLlm = (partial: Partial<UserSettings['llm']>) => {
    onUpdateSettings({ ...settings, llm: { ...settings.llm, ...partial } });
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center p-4">
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm" 
            onClick={onClose} 
          />
          <motion.div 
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="relative w-full max-w-md bg-background border border-border shadow-2xl rounded-3xl overflow-hidden"
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-border">
              <h2 className="text-lg font-bold">{t.settings_title}</h2>
              <button onClick={onClose} className="p-2 hover:bg-muted rounded-full text-muted-foreground transition-colors">
                <X size={20} />
              </button>
            </div>
            
            <div className="p-6 space-y-8 max-h-[60vh] overflow-y-auto custom-scrollbar">
              <section>
                <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-4">{t.settings_llm_title}</h3>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-[11px] font-bold text-muted-foreground">{t.settings_api_base_url}</label>
                    <input
                      value={settings.llm.baseUrl}
                      onChange={(e) => updateLlm({ baseUrl: e.target.value })}
                      placeholder="http://localhost:8000"
                      className="w-full px-4 py-3 rounded-2xl bg-muted/30 border border-border text-foreground text-sm font-medium outline-none focus:border-primary/60 focus:ring-[6px] focus:ring-primary/5 transition-all"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-[11px] font-bold text-muted-foreground">{t.settings_api_token}</label>
                    <input
                      value={settings.llm.token || ''}
                      onChange={(e) => updateLlm({ token: e.target.value })}
                      placeholder={t.settings_api_token_placeholder}
                      type="password"
                      autoComplete="off"
                      className="w-full px-4 py-3 rounded-2xl bg-muted/30 border border-border text-foreground text-sm font-medium outline-none focus:border-primary/60 focus:ring-[6px] focus:ring-primary/5 transition-all"
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between gap-3">
                      <label className="text-[11px] font-bold text-muted-foreground">{t.settings_model}</label>
                      <button
                        type="button"
                        onClick={refreshModels}
                        disabled={modelsLoading}
                        className="text-[10px] font-black uppercase tracking-widest text-primary disabled:opacity-40"
                      >
                        {modelsLoading ? t.settings_models_loading : t.settings_refresh_models}
                      </button>
                    </div>
                    <select
                      value={settings.llm.model}
                      onChange={(e) => updateLlm({ model: e.target.value })}
                      className="w-full px-4 py-3 rounded-2xl bg-muted/30 border border-border text-foreground text-sm font-medium outline-none focus:border-primary/60 focus:ring-[6px] focus:ring-primary/5 transition-all"
                    >
                      {modelOptions.length === 0 ? (
                        <option value={settings.llm.model}>{settings.llm.model || 'core'}</option>
                      ) : (
                        modelOptions.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.id}
                          </option>
                        ))
                      )}
                    </select>
                    {modelsError && <p className="text-[11px] text-red-500 font-bold">{t.settings_models_error}: {modelsError}</p>}
                  </div>

                  <div className="grid grid-cols-3 gap-3">
                    <div className="space-y-2">
                      <label className="text-[11px] font-bold text-muted-foreground">{t.settings_temperature}</label>
                      <input
                        type="number"
                        step="0.1"
                        min="0"
                        max="2"
                        value={settings.llm.temperature}
                        onChange={(e) => updateLlm({ temperature: Number(e.target.value) })}
                        className="w-full px-4 py-3 rounded-2xl bg-muted/30 border border-border text-foreground text-sm font-bold outline-none focus:border-primary/60"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-[11px] font-bold text-muted-foreground">{t.settings_top_p}</label>
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        value={settings.llm.topP}
                        onChange={(e) => updateLlm({ topP: Number(e.target.value) })}
                        className="w-full px-4 py-3 rounded-2xl bg-muted/30 border border-border text-foreground text-sm font-bold outline-none focus:border-primary/60"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-[11px] font-bold text-muted-foreground">{t.settings_max_tokens}</label>
                      <input
                        type="number"
                        step="1"
                        min="1"
                        max="32768"
                        value={settings.llm.maxTokens}
                        onChange={(e) => updateLlm({ maxTokens: Math.max(1, Math.floor(Number(e.target.value))) })}
                        className="w-full px-4 py-3 rounded-2xl bg-muted/30 border border-border text-foreground text-sm font-bold outline-none focus:border-primary/60"
                      />
                    </div>
                  </div>
                </div>
              </section>

              <section>
                <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-4">{t.language}</h3>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { id: 'es', label: t.lang_es, flagUrl: 'https://flagcdn.com/w80/es.png' },
                    { id: 'en', label: t.lang_en, flagUrl: 'https://flagcdn.com/w80/us.png' },
                  ].map((lang) => (
                    <button
                      key={lang.id}
                      onClick={() => handleLanguageChange(lang.id as Language)}
                      className={`flex flex-col items-center justify-center gap-3 p-5 rounded-2xl border transition-all ${
                        settings.language === lang.id 
                          ? 'bg-primary/10 border-primary text-primary shadow-inner' 
                          : 'bg-muted/30 border-border text-muted-foreground hover:bg-muted/50 hover:border-border/80'
                      }`}
                    >
                      <img 
                        src={lang.flagUrl} 
                        alt={lang.label} 
                        className="w-10 h-auto object-contain filter drop-shadow-sm select-none rounded-sm"
                      />
                      <span className="text-[10px] font-black uppercase tracking-widest opacity-80">{lang.label}</span>
                    </button>
                  ))}
                </div>
              </section>

              <section>
                <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-4">{t.settings_code_theme}</h3>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { id: 'light', label: 'Claro', icon: <Sun size={14} /> },
                    { id: 'dark', label: 'Oscuro', icon: <Moon size={14} /> },
                    { id: 'match-app', label: 'Sistema', icon: <Monitor size={14} /> },
                  ].map((theme) => (
                    <button
                      key={theme.id}
                      onClick={() => handleCodeThemeChange(theme.id as any)}
                      className={`flex flex-col items-center justify-center gap-2 p-3 rounded-xl border transition-all ${
                        settings.codeTheme === theme.id 
                          ? 'bg-primary/10 border-primary text-primary shadow-sm' 
                          : 'bg-muted/30 border-border text-muted-foreground hover:bg-muted/50'
                      }`}
                    >
                      {theme.icon}
                      <span className="text-[10px] font-bold uppercase tracking-wide">{theme.label}</span>
                    </button>
                  ))}
                </div>
              </section>

              <section>
                <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-4">{t.settings_font_size}</h3>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { id: 'small', label: t.font_small, icon: <Type size={12} /> },
                    { id: 'medium', label: t.font_medium, icon: <Type size={16} /> },
                    { id: 'large', label: t.font_large, icon: <Type size={20} /> },
                  ].map((size) => (
                    <button
                      key={size.id}
                      onClick={() => handleFontSizeChange(size.id as FontSize)}
                      className={`flex flex-col items-center justify-center gap-2 p-3 rounded-xl border transition-all ${
                        settings.fontSize === size.id 
                          ? 'bg-primary/10 border-primary text-primary shadow-sm' 
                          : 'bg-muted/30 border-border text-muted-foreground hover:bg-muted/50'
                      }`}
                    >
                      {size.icon}
                      <span className="text-[10px] font-bold uppercase tracking-wide">{size.label}</span>
                    </button>
                  ))}
                </div>
              </section>

              <section>
                <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-2">{t.settings_section_order}</h3>
                <p className="text-[11px] text-muted-foreground mb-4">{t.settings_section_desc}</p>
                
                <Reorder.Group 
                  axis="y" 
                  values={settings.categoryOrder} 
                  onReorder={handleReorder}
                  className="space-y-2"
                >
                  {settings.categoryOrder.map((category) => (
                    <Reorder.Item 
                      key={category} 
                      value={category}
                      className="relative"
                    >
                      <motion.div 
                        className="flex items-center gap-3 p-3 bg-muted/30 border border-border rounded-xl group hover:border-primary/40 transition-colors cursor-grab active:cursor-grabbing bg-background/50 backdrop-blur-sm shadow-sm"
                        whileDrag={{ 
                          scale: 1.02, 
                          boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)",
                          zIndex: 1
                        }}
                      >
                        <GripVertical size={14} className="text-muted-foreground/30 group-hover:text-muted-foreground transition-colors shrink-0" />
                        <span className="flex-1 text-[13px] font-semibold select-none">{category}</span>
                      </motion.div>
                    </Reorder.Item>
                  ))}
                </Reorder.Group>
              </section>
            </div>

            <div className="p-6 bg-muted/20 border-t border-border flex justify-end">
              <button 
                onClick={onClose}
                className="px-6 py-2.5 bg-primary text-primary-foreground rounded-xl text-sm font-bold shadow-sm hover:opacity-90 active:scale-95 transition-all"
              >
                {t.settings_close}
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default SettingsModal;
