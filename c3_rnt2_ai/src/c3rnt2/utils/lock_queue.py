import sqlite3
import time
import logging
import os
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger("vortex.lock_queue")

class LockQueueManager:
    """
    Gestor robusto de bloqueos y colas para procesos concurrentes de Vortex
    (serve, train, self_patch). Utiliza SQLite como backend centralizado
    y seguro para múltiples procesos (FastAPI, Control Plane, Workers).
    """
    
    CONFLICTS = {
        "serve": {"serve_fallback", "train", "self_patch"},
        "serve_fallback": {"serve", "self_patch"},
        "train": {"serve", "self_patch"},
        "self_patch": {"serve", "serve_fallback", "train"},
    }

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid = os.getpid()
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS active_locks (
                    role TEXT PRIMARY KEY,
                    pid INTEGER,
                    acquired_at REAL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS lock_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    pid INTEGER,
                    queued_at REAL
                )
            ''')
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            yield conn
        finally:
            conn.close()

    def _is_conflicting_lock_held(self, conn, role: str) -> bool:
        conflicts = self.CONFLICTS.get(role, set())
        if not conflicts:
            return False
            
        placeholders = ','.join('?' for _ in conflicts)
        cursor = conn.execute(
            f'SELECT role FROM active_locks WHERE role IN ({placeholders})',
            tuple(conflicts)
        )
        return cursor.fetchone() is not None

    def acquire(self, role: str, timeout_s: float = 300.0, poll_interval_s: float = 2.0) -> bool:
        """
        Intenta adquirir un lock. Si hay conflicto, se encola y espera hasta timeout_s.
        """
        role = role.lower()
        if role not in self.CONFLICTS:
            raise ValueError(f"Rol '{role}' no reconocido.")

        logger.info(f"[PID {self.pid}] Solicitando lock para '{role}'...")
        
        # Encolar la solicitud
        with self._connect() as conn:
            cursor = conn.execute(
                'INSERT INTO lock_queue (role, pid, queued_at) VALUES (?, ?, ?)',
                (role, self.pid, time.time())
            )
            queue_id = cursor.lastrowid
            conn.commit()

        start_time = time.time()
        
        try:
            while True:
                with self._connect() as conn:
                    # Verificar si es nuestro turno (somos el más antiguo en la cola para nuestro rol o roles sin conflicto)
                    # En una implementación real más compleja podríamos permitir concurrencia parcial, 
                    # aquí usamos un modelo seguro FIFO general para roles que colisionan.
                    
                    cursor = conn.execute('SELECT id, role FROM lock_queue ORDER BY queued_at ASC LIMIT 1')
                    first_in_queue = cursor.fetchone()
                    
                    if first_in_queue and first_in_queue[0] == queue_id:
                        # Es nuestro turno en la cola. Veamos si podemos adquirir (no hay locks activos conflictivos)
                        if not self._is_conflicting_lock_held(conn, role):
                            # Adquirir lock
                            conn.execute(
                                'INSERT OR REPLACE INTO active_locks (role, pid, acquired_at) VALUES (?, ?, ?)',
                                (role, self.pid, time.time())
                            )
                            # Salir de la cola
                            conn.execute('DELETE FROM lock_queue WHERE id = ?', (queue_id,))
                            conn.commit()
                            logger.info(f"[PID {self.pid}] Lock '{role}' adquirido con éxito.")
                            return True

                if (time.time() - start_time) > timeout_s:
                    logger.warning(f"[PID {self.pid}] Timeout de {timeout_s}s esperando lock '{role}'.")
                    break
                    
                time.sleep(poll_interval_s)
                
        finally:
            # Limpieza de la cola por si falló o hizo timeout
            with self._connect() as conn:
                conn.execute('DELETE FROM lock_queue WHERE id = ?', (queue_id,))
                conn.commit()

        return False

    def release(self, role: str):
        """Libera el lock adquirido."""
        role = role.lower()
        with self._connect() as conn:
            conn.execute('DELETE FROM active_locks WHERE role = ? AND pid = ?', (role, self.pid))
            conn.commit()
        logger.info(f"[PID {self.pid}] Lock '{role}' liberado.")

    @contextmanager
    def lock_scope(self, role: str, timeout_s: float = 300.0):
        """Context manager para usar con la declaración 'with'."""
        acquired = self.acquire(role, timeout_s)
        if not acquired:
            raise RuntimeError(f"No se pudo adquirir el lock '{role}' después de {timeout_s}s.")
        try:
            yield
        finally:
            self.release(role)
