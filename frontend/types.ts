
export enum Role {
  USER = 'user',
  AI = 'ai'
}

export type ViewType = 'chat' | 'analysis' | 'terminal';
export type AppMode = 'ask' | 'agent';
export type FontSize = 'small' | 'medium' | 'large';
export type Language = 'es' | 'en';

export interface Source {
  title: string;
  url: string;
  domain: string;
  index: number;
}

export interface GroundingSupport {
  segmentText: string;
  startIndex: number;
  endIndex: number;
  sourceIndices: number[];
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  thought?: string;
  sources?: Source[];
  groundingSupports?: GroundingSupport[];
  timestamp: number;
  fileChanges?: { path: string; diff: string }[];
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
}

export interface LlmSettings {
  baseUrl: string;
  token?: string;
  model: string;
  temperature: number;
  topP: number;
  maxTokens: number;
}

export interface UserSettings {
  categoryOrder: string[];
  codeTheme: 'dark' | 'light' | 'match-app';
  fontSize: FontSize;
  language: Language;
  llm: LlmSettings;
}

export interface LogEntry {
  id: string;
  timestamp: number;
  level: 'INFO' | 'LEARN' | 'SEARCH' | 'SYSTEM';
  message: string;
}
