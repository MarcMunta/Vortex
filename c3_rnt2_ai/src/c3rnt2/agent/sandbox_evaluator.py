import os
import shutil
import tempfile
import logging
from pathlib import Path
import docker
from docker.errors import DockerException

logger = logging.getLogger("vortex.sandbox_evaluator")

class SandboxEvaluator:
    """
    Evaluador en entorno Sandbox (Docker) para verificar auto-ediciones de Vortex.
    Crea un contenedor efímero, copia el código con el parche aplicado, 
    y ejecuta tests antes de notificar al usuario para su aprobación.
    """
    def __init__(self, image: str = "python:3.11-slim"):
        self.image = image
        try:
            self.client = docker.from_env()
        except DockerException as e:
            logger.error(f"Error conectando a Docker. Asegúrate de que el daemon esté corriendo: {e}")
            self.client = None

    def evaluate_patch(self, repo_root: Path, diff_text: str, test_cmd: str = "pytest") -> dict:
        """
        1. Crea un directorio temporal.
        2. Copia el repo_root ignorando carpetas pesadas/cachés.
        3. Aplica el parche (.diff) en el directorio temporal usando `patch`.
        4. Monta el directorio en el contenedor efímero y corre `test_cmd`.
        5. Retorna el resultado.
        """
        if not self.client:
            return {"ok": False, "error": "Docker no está disponible."}

        # Ignorar directorios pesados
        ignore_patterns = shutil.ignore_patterns(".git", ".venv", "__pycache__", "node_modules", "data", ".pytest_cache")

        with tempfile.TemporaryDirectory(prefix="vortex_sandbox_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            sandbox_repo = tmp_path / "repo"
            
            logger.info(f"Copiando repositorio a sandbox temporal: {sandbox_repo}")
            shutil.copytree(repo_root, sandbox_repo, ignore=ignore_patterns)

            # Escribir el parche y aplicarlo
            patch_file = tmp_path / "proposal.diff"
            patch_file.write_text(diff_text, encoding="utf-8")
            
            # Aplicar parche localmente en la copia
            apply_res = os.system(f"patch -p1 -d {sandbox_repo} < {patch_file}")
            if apply_res != 0:
                logger.error("No se pudo aplicar el parche en el entorno de pruebas.")
                return {"ok": False, "error": "Fallo al aplicar diff con `patch`.", "stdout": "", "stderr": "Patch apply failed"}

            # Ejecutar el contenedor
            container_cmd = f"sh -c '{test_cmd}'"
            logger.info(f"Ejecutando '{test_cmd}' en contenedor Docker ({self.image})...")
            
            try:
                # Opcional: Podrías necesitar instalar dependencias aquí si no usas una imagen ya preparada.
                # Para un pipeline real, 'self.image' debería ser una imagen pre-buildeada del proyecto.
                container = self.client.containers.run(
                    self.image,
                    container_cmd,
                    volumes={str(sandbox_repo.absolute()): {'bind': '/app', 'mode': 'rw'}},
                    working_dir='/app',
                    detach=True,
                    remove=False, # Lo mantenemos para leer logs y status, luego lo borramos
                    network_disabled=True # Sandbox sin internet para seguridad
                )
                
                result = container.wait(timeout=120)
                exit_code = result.get('StatusCode', -1)
                logs = container.logs().decode('utf-8')
                container.remove()

                if exit_code == 0:
                    logger.info("¡Tests pasados exitosamente en el Sandbox!")
                    return {"ok": True, "returncode": exit_code, "stdout": logs, "stderr": ""}
                else:
                    logger.warning(f"Tests fallaron en el Sandbox (Exit code: {exit_code}).")
                    return {"ok": False, "returncode": exit_code, "stdout": logs, "stderr": f"Failed with code {exit_code}"}

            except Exception as e:
                logger.error(f"Error ejecutando contenedor Sandbox: {e}")
                return {"ok": False, "error": str(e), "stdout": "", "stderr": str(e)}

