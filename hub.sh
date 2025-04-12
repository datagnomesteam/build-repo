#!/bin/bash

echo "Stopping and removing all containers, images, and volumes..."
docker compose -f docker-compose-hub.yml down --rmi all --volumes

echo "Pruning unused Docker resources..."
docker system prune -af
docker system prune -af --volumes

echo "Remove named volume if it exists..."
docker volume rm postgres_data 2>/dev/null || echo "Data Volume not found or already removed."
docker volume rm postgres_logs 2>/dev/null || echo "Logs Volume not found or already removed."

echo "Starting services..."
docker compose -f ./docker-compose-hub.yml up 