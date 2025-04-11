# Data Analytics Dashboard

This project provides a comprehensive data analytics platform with a PostgreSQL database backend, FastAPI server, and Streamlit frontend for data visualization and analysis.

## System Architecture

The application consists of two main containers:

- **Database Container**: PostgreSQL database for storing data
- **Server Container**: Runs both the FastAPI backend (port 8000) and Streamlit frontend (port 8501)

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) [Use the Desktop version if you can]

## Getting Started

### Running the Application

1. Clone the repository:

   ```bash
   git clone https://github.com/zibdie/CSE6242-Group-Project
   cd CSE6242-Group-Project
   ```

2. Start the containers:

   ```bash
   docker-compose up -d
   ```

3. Access the application:
   - Streamlit dashboard: [http://localhost:8501](http://localhost:8501)
   - FastAPI endpoints: [http://localhost:8000](http://localhost:8000)
   - API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### Stopping the Application

```bash
docker-compose down
```

To remove volumes and completely clean up:

```bash
docker-compose down -v
```

## Features

The dashboard provides several features:

- **Data Explorer**: View and explore data with visualizations
- **Data Analysis**:
  - Spark Analysis: Analyze data by category using PySpark
  - Clustering Analysis: Perform K-means clustering using scikit-learn
- **Add Data**: Add new data to the database

## Development

### Project Structure

```
CSE6242-Group-Project/
├── db/
│ ├── Dockerfile - PostgreSQL database configuration
│ └── init-scripts/ - Database initialization scripts
├── server/
│ ├── Dockerfile - FastAPI and Streamlit server configuration
│ ├── app.py - Streamlit dashboard application
│ └── ... (other server files)
└── docker-compose.yml - Container orchestration configuration
```

### Customizing Environment Variables

You can modify the following environment variables in the `docker-compose.yml` file:

- Database credentials
- Connection parameters
- API URLs

### Viewing Logs

```bash
# View all logs
docker-compose logs

# Follow logs in real time
docker-compose logs -f

# View logs for a specific service
docker-compose logs server
```

## Troubleshooting

- If you encounter connection issues, ensure all containers are running: `docker-compose ps`
- For database connection problems, check that the PostgreSQL container is healthy: `docker ps`
- If the API doesn't respond, verify that the server container can communicate with the database

## Data Persistence

Database data is stored in a Docker volume named `postgres_data` and persists between container restarts.
