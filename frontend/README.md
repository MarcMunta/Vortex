# Vortex UI (Frontend)

This is the interface for the Vortex AI System. It is built with **React**, **TypeScript**, and **Vite**.

## Configuration

The frontend talks to the local **Vortex** backend (OpenAI-compatible).

- Copy `frontend/.env.example` to `frontend/.env` (optional in dev).
- Default endpoints used:
  - `GET /v1/models`
  - `POST /v1/chat/completions` (SSE streaming)

## Commands

### Install Dependencies
```bash
npm install
```

### Run Development Server
```bash
npm run dev
```

### Build for Production
```bash
npm run build
```

## Structure

- **`services/vortexClient.ts`**: Typed HTTP client + SSE parser.
- **`services/vortexService.ts`**: UI-friendly stream adapter (thinking tags, sources, file diffs).
- **`components/`**: UI components for the chat interface.
- **`types.ts`**: TypeScript definitions for messages and API responses.
