#!/bin/bash

# =============================================================================
# Ejemplos de curl para validar el modelo OlmOCR
# =============================================================================
# Archivo: curl_examples.sh
# Uso: ./curl_examples.sh [PORT]
#
# Ejemplos para probar diferentes funcionalidades del modelo OlmOCR
# con RTX 4090 optimizada

VLLM_PORT=${1:-8001}
BASE_URL="http://localhost:$VLLM_PORT"

echo "🚀 Ejecutando ejemplos de curl para OlmOCR en $BASE_URL"
echo "================================================================="

# Función para ejecutar curl con manejo de errores
run_curl_example() {
    local title="$1"
    local curl_cmd="$2"

    echo ""
    echo "📋 $title"
    echo "----------------------------------------"

    # Ejecutar el comando curl
    eval "$curl_cmd"
    local status=$?

    if [ $status -eq 0 ]; then
        echo "✅ Request completado exitosamente"
    else
        echo "❌ Error en el request (código: $status)"
    fi

    echo "----------------------------------------"
    sleep 2
}

# =============================================================================
# EJEMPLOS DE CURL
# =============================================================================

# 1. Health Check básico
run_curl_example "1. Health Check" "
curl -s \"$BASE_URL/health\" \
  | jq . 2>/dev/null || echo 'Health check responded (JSON parsing optional)'
"

# 2. Listar modelos disponibles
run_curl_example "2. Listar Modelos" "
curl -s \"$BASE_URL/v1/models\" \
  | jq '.data[]? | {id: .id, object: .object}' 2>/dev/null
"

# 3. Información detallada del modelo
run_curl_example "3. Información del Modelo OlmOCR" "
curl -s \"$BASE_URL/v1/models/olmocr\" \
  | jq . 2>/dev/null
"

# 4. Ejemplo básico de completion
run_curl_example "4. Completion Básico" "
curl -s -X POST \"$BASE_URL/v1/completions\" \
  -H 'Content-Type: application/json' \
  -d '{
    \"model\": \"olmocr\",
    \"prompt\": \"The future of artificial intelligence is\",
    \"max_tokens\": 30,
    \"temperature\": 0.7
  }' \
  | jq '.choices[]?.text' 2>/dev/null
"

# 5. Ejemplo de chat completion (texto simple)
run_curl_example "5. Chat Completion - Texto Simple" "
curl -s -X POST \"$BASE_URL/v1/chat/completions\" \
  -H 'Content-Type: application/json' \
  -d '{
    \"model\": \"olmocr\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"Hola, ¿cómo estás?\"
      }
    ],
    \"max_tokens\": 50,
    \"temperature\": 0.5
  }' \
  | jq '.choices[]?.message.content' 2>/dev/null
"

# 6. Ejemplo de análisis de imagen (Estatua de la Libertad)
run_curl_example "6. Análisis de Imagen - Estatua de la Libertad" "
curl -s -X POST \"$BASE_URL/v1/chat/completions\" \
  -H 'Content-Type: application/json' \
  -d '{
    \"model\": \"olmocr\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"text\",
            \"text\": \"Describe this image in one sentence.\"
          },
          {
            \"type\": \"image_url\",
            \"image_url\": {
              \"url\": \"https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg\"
            }
          }
        ]
      }
    ],
    \"max_tokens\": 100,
    \"temperature\": 0.1
  }' \
  | jq '.choices[]?.message.content' 2>/dev/null
"

# 7. Ejemplo de análisis de imagen complejo
run_curl_example "7. Análisis Complejo de Imagen" "
curl -s -X POST \"$BASE_URL/v1/chat/completions\" \
  -H 'Content-Type: application/json' \
  -d '{
    \"model\": \"olmocr\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"text\",
            \"text\": \"Extrae toda la información importante de esta imagen: nombres, fechas, números y conceptos clave. Organiza la información de manera estructurada.\"
          },
          {
            \"type\": \"image_url\",
            \"image_url\": {
              \"url\": \"https://images.unsplash.com/photo-1586953208448-b95a79798f07?w=800\"
            }
          }
        ]
      }
    ],
    \"max_tokens\": 300,
    \"temperature\": 0.1
  }' \
  | jq '.choices[]?.message.content' 2>/dev/null
"

# 8. Ejemplo en español
run_curl_example "8. Análisis en Español" "
curl -s -X POST \"$BASE_URL/v1/chat/completions\" \
  -H 'Content-Type: application/json' \
  -d '{
    \"model\": \"olmocr\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"text\",
            \"text\": \"¿Qué ves en esta imagen? Describe detalladamente todos los elementos que puedes identificar.\"
          },
          {
            \"type\": \"image_url\",
            \"image_url\": {
              \"url\": \"https://images.unsplash.com/photo-1544568100-847a948585b9?w=800\"
            }
          }
        ]
      }
    ],
    \"max_tokens\": 200,
    \"temperature\": 0.2
  }' \
  | jq '.choices[]?.message.content' 2>/dev/null
"

echo ""
echo "🎉 ¡Ejemplos de curl completados!"
echo "================================================================="
echo ""
echo "💡 Consejos:"
echo "   - Si algún ejemplo falla, verifica que el modelo esté completamente cargado"
echo "   - Usa 'docker-compose logs vllm-olmocr' para ver logs detallados"
echo "   - Asegúrate de que el puerto $VLLM_PORT esté disponible"
echo ""
echo "🚀 Para ejecutar ejemplos individuales:"
echo "   curl -X POST \"$BASE_URL/v1/completions\" -H 'Content-Type: application/json' -d '{\"model\":\"olmocr\",\"prompt\":\"Hello world\",\"max_tokens\":10}'"