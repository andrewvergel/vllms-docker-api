#!/bin/bash
set -e

echo "🔍 Checking runtime dependencies..."

# Verificar que 'vllm' esté disponible
if ! command -v vllm >/dev/null 2>&1; then
    echo "⚠️ 'vllm' binary not found, creating wrapper..."
    echo '#!/bin/sh' > /usr/local/bin/vllm
    echo 'exec python -m vllm.entrypoints.api_server "$@"' >> /usr/local/bin/vllm
    chmod +x /usr/local/bin/vllm
else
    echo "✅ vllm binary found."
fi

# Verificar versión de vllm
echo "ℹ️ vllm version:"
python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "vllm version check failed"

# Lanzar tu API FastAPI
echo "🚀 Starting OlmOCR API..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
