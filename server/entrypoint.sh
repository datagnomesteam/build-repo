#!/bin/bash
echo "Starting FastAPI server..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!
echo "FastAPI started with PID $FASTAPI_PID"

echo "Starting Streamlit server..."
python3 model_objects/run_algos.py
streamlit run Home.py --server.port=8501 --server.address=0.0.0.0

# If Streamlit exits, kill FastAPI too
kill $FASTAPI_PID 