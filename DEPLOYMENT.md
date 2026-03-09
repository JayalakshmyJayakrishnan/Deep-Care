# Deployment Guide

The project is now configured for deployment. It uses a **Dockerized** approach, which is compatible with almost all cloud providers (Google Cloud Run, AWS App Runner, Azure Container Apps, Render, Railway, etc.).

## Option A: Deploy using Docker (Recommended)

1.  **Build the Image**:
    ```bash
    docker build -t pillcam-platform .
    ```
2.  **Run Locally (to test)**:
    ```bash
    docker run -p 8080:8080 pillcam-platform
    ```
3.  **Deploy to Cloud**:
    -   If using **Google Cloud Run**:
        ```bash
        gcloud run deploy pillcam-platform --source .
        ```
    -   If using **Render/Railway**:
        -   Connect your GitHub repository.
        -   It will automatically detect the `Dockerfile` and deploy.

## Option B: Manual / ZIP Upload

Since the `Dockerfile` handles the build, you can simply:
1.  Zip the entire `pillcam_platform` folder.
2.  Upload it to your Cloud Console (e.g., AWS Elastic Beanstalk, Google App Engine).
3.  Ensure the entry point is set to `gunicorn --bind 0.0.0.0:8080 backend.app:app`.

## Configuration Details
-   **Port**: The app listens on port `8080` (standard for cloud runtimes).
-   **Static Files**: The Flask backend serves the React frontend from the `frontend/dist` folder automatically.
