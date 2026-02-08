#!/bin/bash

# Configuration
BACKEND_PORT=8848
FRONTEND_DIR="investment-ui"

echo "🚀 Starting Investment Strategy Builder Setup..."

# 1. Install Backend Dependencies
if [ -f "backend/requirements.txt" ]; then
    echo "📦 Installing Python dependencies using python3..."
    python3 -m pip install -r backend/requirements.txt
else
    echo "⚠️ Warning: backend/requirements.txt not found."
fi

# 2. Install Frontend Dependencies
if [ -d "$FRONTEND_DIR" ]; then
    echo "📦 Installing Node.js dependencies..."
    cd $FRONTEND_DIR
    npm install
    cd ..
else
    echo "⚠️ Warning: $FRONTEND_DIR directory not found."
fi

# 3. Launch Backend
echo "🐍 Starting Flask Backend on port $BACKEND_PORT..."
# Using nohup to keep it running or just & for interactive use
python3 backend/main.py &
BACKEND_PID=$!

# 4. Launch Frontend
if [ -d "$FRONTEND_DIR" ]; then
    echo "⚛️ Starting React Frontend..."
    cd $FRONTEND_DIR
    npm start
else
    echo "❌ Error: Cannot start frontend. Directory missing."
    kill $BACKEND_PID
    exit 1
fi
