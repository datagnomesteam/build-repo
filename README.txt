# Getting Started

Our project is packed into two Docker images you can pull and the provided `docker-compose-hub.yml` sets all variables, ports, and images for you.

## Requirements

You will need:
* Docker Desktop (or Docker Engine with Docker Compose)
* An x86-64 (Intel) or ARM (M1 devices) based device 
* Atleast 6GB of storage (may be larger depending how Docker unpacks data on your system or atleast 50GB+ if you want to run the full database)
* A fast, uninterruptable internet connection
* Ability to access DockerHub (https://registry-1.docker.io, https://hub.docker.com should be whitelisted)

## Steps:

1. Ensure your user account can pull & run Docker images (If you can run `docker run hello-world` without issues, your system is ready)
2. Ensure ports 8000, 8501, and 5432 are open for local access only.
3. Place the provided `docker-compose-hub.yml` in an empty accessible directory.
4. Open a Command Prompt/Powershell (for Windows) or Terminal (for Mac/Linux) and `cd` into the directory containing your `docker-compose-hub.yml`
5. Run `docker compose -f ./docker-compose-hub.yml up` 
6. Wait for the images to download and the containers to be built
7. Once both the `db` and `data_server` containers are ready, navigate to `http://localhost:8501` to view the page

If you need assistance, contact our team leader or email `mzibdie3@gatech.edu`