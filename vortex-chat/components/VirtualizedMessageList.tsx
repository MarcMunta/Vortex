
import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { AnimatePresence } from 'framer-motion';
import MessageBubble from './MessageBubble';
import AgentMessageWrapper from './agent/AgentMessageWrapper';
import { Message, Role, FontSize, Language, AppMode } from '../types';

interface VirtualizedMessageListProps {
  messages: Message[];
  fontSize: FontSize;
  codeTheme: 'dark' | 'light' | 'match-app';
  onShowReasoning: (messageId: string) => void;
  onOpenModificationExplorer: (fileChanges: { path: string, diff: string }[]) => void;
  onSuggestPatch: (messageId: string) => void;
  onContinueResponse?: (messageId: string) => void;
  isLoading: boolean;
  language: Language;
  containerRef: React.RefObject<HTMLDivElement | null>;
  mode?: AppMode;
}

const BUFFER_COUNT = 5; 
const ESTIMATED_HEIGHT = 124;

const VirtualizedMessageList: React.FC<VirtualizedMessageListProps> = ({
  messages,
  fontSize,
  codeTheme,
  onShowReasoning,
  onOpenModificationExplorer,
  onSuggestPatch,
  onContinueResponse,
  isLoading,
  language,
  containerRef,
  mode
}) => {
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);
  const heightsMap = useRef<Map<string, number>>(new Map());
  const [totalHeight, setTotalHeight] = useState(0);
  const observerRef = useRef<ResizeObserver | null>(null);
  const outerRef = useRef<HTMLDivElement | null>(null);
  const innerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    observerRef.current = new ResizeObserver((entries) => {
      let heightChanged = false;
      for (const entry of entries) {
        const id = (entry.target as HTMLElement).dataset.id;
        if (id) {
          const newHeight = entry.contentRect.height;
          if (heightsMap.current.get(id) !== newHeight) {
            heightsMap.current.set(id, newHeight);
            heightChanged = true;
          }
        }
      }
      if (heightChanged) {
        let acc = 0;
        messages.forEach(m => { acc += heightsMap.current.get(m.id) || ESTIMATED_HEIGHT; });
        setTotalHeight(acc);
      }
    });
    return () => observerRef.current?.disconnect();
  }, [messages]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const handleScroll = () => setScrollTop(container.scrollTop);
    const updateSize = () => setContainerHeight(container.clientHeight);
    container.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('resize', updateSize);
    updateSize();
    setScrollTop(container.scrollTop);
    return () => {
      container.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', updateSize);
    };
  }, [containerRef]);

  const registerItem = useCallback((id: string, el: HTMLDivElement | null) => {
    if (el) {
      el.dataset.id = id;
      observerRef.current?.observe(el);
      if (!heightsMap.current.has(id)) {
        const h = el.offsetHeight;
        heightsMap.current.set(id, h);
        let acc = 0;
        messages.forEach(m => { acc += heightsMap.current.get(m.id) || ESTIMATED_HEIGHT; });
        setTotalHeight(acc);
      }
    }
  }, [messages]);

  const { visibleMessages, offsetTop, startIndex } = useMemo(() => {
    let currentOffset = 0;
    let startIdx = 0;
    let endIdx = messages.length;
    for (let i = 0; i < messages.length; i++) {
      const h = heightsMap.current.get(messages[i].id) || ESTIMATED_HEIGHT;
      if (currentOffset + h > scrollTop) {
        startIdx = Math.max(0, i - BUFFER_COUNT);
        break;
      }
      currentOffset += h;
    }
    let accOffset = 0;
    for (let j = 0; j < startIdx; j++) { accOffset += heightsMap.current.get(messages[j].id) || ESTIMATED_HEIGHT; }
    currentOffset = accOffset;
    for (let i = startIdx; i < messages.length; i++) {
      const h = heightsMap.current.get(messages[i].id) || ESTIMATED_HEIGHT;
      currentOffset += h;
      if (currentOffset > scrollTop + containerHeight + (ESTIMATED_HEIGHT * BUFFER_COUNT)) {
        endIdx = Math.min(messages.length, i + 1);
        break;
      }
    }
    return { visibleMessages: messages.slice(startIdx, endIdx), offsetTop: accOffset, startIndex: startIdx };
  }, [messages, scrollTop, containerHeight, totalHeight]);

  useEffect(() => {
    if (!outerRef.current) return;
    const nextHeight = Math.max(totalHeight, messages.length * ESTIMATED_HEIGHT);
    outerRef.current.style.height = `${nextHeight}px`;
  }, [totalHeight, messages.length]);

  useEffect(() => {
    if (!innerRef.current) return;
    innerRef.current.style.transform = `translateY(${offsetTop}px)`;
  }, [offsetTop]);

  return (
    <div ref={outerRef} className="relative w-full accelerated">
      <div ref={innerRef} className="absolute top-0 left-0 w-full flex flex-col will-change-transform">
        <AnimatePresence mode="popLayout" initial={false}>
          {visibleMessages.map((msg, index) => {
            const absoluteIndex = startIndex + index;
            const isStreaming = isLoading && absoluteIndex === messages.length - 1 && msg.role === Role.AI;
            return (
              <div key={msg.id} ref={(el) => registerItem(msg.id, el)} className="w-full" data-app-mode={msg.mode || 'ask'}>
                {msg.mode === 'agent' && msg.role === Role.AI ? (
                  <AgentMessageWrapper
                    message={msg}
                    isStreaming={isStreaming}
                    language={language}
                    fontSize={fontSize}
                  />
                ) : (
                  <MessageBubble 
                    message={msg} 
                    fontSize={fontSize} 
                    codeTheme={codeTheme}
                    onShowReasoning={onShowReasoning} 
                    onOpenModificationExplorer={onOpenModificationExplorer} 
                    onSuggestPatch={onSuggestPatch}
                    onContinueResponse={onContinueResponse}
                    isStreaming={isStreaming} 
                    language={language} 
                  />
                )}
              </div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default VirtualizedMessageList;
