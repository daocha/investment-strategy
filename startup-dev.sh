#!/bin/bash

# Configuration
BACKEND_PORT=8848
FRONTEND_PORT=3848
FRONTEND_DIR="investment-ui"

# Cleanup function to kill background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    [ ! -z "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null
    [ ! -z "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null
    [ ! -z "$WORKER_PID" ] && kill $WORKER_PID 2>/dev/null
    
    # Surgical port cleanup (optional but helpful)
    lsof -ti :$BACKEND_PORT | xargs kill -9 2>/dev/null
    lsof -ti :$FRONTEND_PORT | xargs kill -9 2>/dev/null
    echo "✅ Done."
    exit
}

# Trap Ctrl+C (SIGINT) and terminal close (SIGTERM)
trap cleanup SIGINT SIGTERM

echo "🚀 Starting Investment Strategy Builder Setup..."

# 0. Pre-start: ensure ports are clear
echo "🧹 Clearing existing processes on ports $BACKEND_PORT and $FRONTEND_PORT..."
lsof -ti :$BACKEND_PORT | xargs kill -9 2>/dev/null
lsof -ti :$FRONTEND_PORT | xargs kill -9 2>/dev/null

# 1. Install Backend Dependencies
if [ -f "backend/requirements.txt" ]; then
    echo "📦 Installing Python dependencies using python3..."
    python3 -m pip install -r backend/requirements.txt
else
    echo "⚠️ Warning: backend/requirements.txt not found."
fi

# 1.5. Ensure XGBoost Model exists
MODEL_JSON="backend/xgboost_model.json"
if [ ! -f "$MODEL_JSON" ]; then
    echo "🧠 XGBoost model not found. Generating initial 'brain'..."
    python3 backend/train_model.py
else
    echo "✅ XGBoost model found. Skipping training."
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
python3 backend/main.py &
BACKEND_PID=$!

# 3.5. Launch Maintenance Worker
echo "⚙️ Starting Maintenance Worker (Schedules: HKT 05:30, 16:30)..."
python3 backend/maintenance_worker.py &
WORKER_PID=$!

# 4. Launch Frontend
if [ -d "$FRONTEND_DIR" ]; then
    echo "⚛️ Starting React Frontend on port $FRONTEND_PORT..."
    cd $FRONTEND_DIR
    # Run in subshell and capture PID
    PORT=$FRONTEND_PORT npm start &
    FRONTEND_PID=$!
    # Wait for child processes (this keeps the script alive to catch the trap)
    wait $FRONTEND_PID
else
    echo "❌ Error: Cannot start frontend. Directory missing."
    kill $BACKEND_PID
    exit 1
fi
