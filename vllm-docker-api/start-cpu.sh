#!/bin/bash

# =====================================
# OlmOCR API Startup Script for CPU-Only Mode
# =====================================

set -e

echo "🚀 Starting OlmOCR API (CPU-Only Mode)..."
echo "========================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✅ Docker is running"

# Check if ports are available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ Port 8000 is already in use"
    echo "   Please free up port 8000 or modify the port mapping in docker-compose.cpu.yml"
    exit 1
fi

if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ Port 8001 is already in use"
    echo "   Please free up port 8001 or modify the port mapping in docker-compose.cpu.yml"
    exit 1
fi

echo "✅ Required ports are available"

# Build and start the services
echo ""
echo "🏗️  Building and starting services..."
echo "   This may take several minutes on first run due to model download"
echo ""

docker-compose -f docker-compose.cpu.yml up -d

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to start..."
echo "   VLLM Server (port 8001)..."
sleep 10

# Check if services are healthy
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "   ✅ VLLM Server is ready!"
else
    echo "   ⚠️  VLLM Server may still be starting (this can take several minutes)"
fi

echo ""
echo "🎉 OlmOCR API (CPU-Only Mode) is starting up!"
echo ""
echo "📋 Service URLs:"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • Health Check: http://localhost:8000/health"
echo "   • VLLM Server: http://localhost:8001/health"
echo ""
echo "📝 Usage Example:"
echo "   curl -X POST 'http://localhost:8000/process' \\"
echo "     -F 'file=@document.pdf' \\"
echo "     -F 'output_format=markdown'"
echo ""
echo "🔍 View logs:"
echo "   docker-compose -f docker-compose.cpu.yml logs -f"
echo ""
echo "🛑 To stop:"
echo "   docker-compose -f docker-compose.cpu.yml down"