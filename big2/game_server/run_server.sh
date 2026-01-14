#!/bin/bash
# Start the Big 2 Game Server

set -e

PORT="${BIG2_PORT:-8000}"
HOST="${BIG2_HOST:-0.0.0.0}"
WORKERS="${BIG2_WORKERS:-1}"
API_KEY="${BIG2_API_KEY:-dev-api-key-changeme}"

echo "🎮 Starting Big 2 Game Server"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Workers: $WORKERS"
echo "   API Key: ${API_KEY:0:10}..."
echo ""
echo "📚 Documentation available at:"
echo "   Swagger UI: http://localhost:$PORT/docs"
echo "   ReDoc:      http://localhost:$PORT/redoc"
echo "   OpenAPI:    http://localhost:$PORT/openapi.json"
echo ""

export BIG2_API_KEY="$API_KEY"

if [ "$WORKERS" = "1" ]; then
    # Development mode with reload
    exec uvicorn big2.game_server.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload
else
    # Production mode
    exec uvicorn big2.game_server.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS"
fi

