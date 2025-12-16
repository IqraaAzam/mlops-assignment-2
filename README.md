# MLOps Assignment - Complete Implementation

This repository contains the complete implementation of the MLOps assignment covering all 7 tasks including DVC, CI/CD, Docker, Airflow, FastAPI, and AWS deployment.

## Project Structure

```
mlops-assignment/
├── .github/workflows/ci.yml    # CI/CD pipeline
├── api/main.py                 # FastAPI application
├── dags/train_pipeline.py      # Airflow DAG
├── src/train.py               # Training script
├── tests/test_train.py        # Unit tests
├── data/dataset.csv           # Dataset (tracked by DVC)
├── models/                    # Model outputs
├── requirements.txt           # Dependencies
├── Dockerfile.api            # API containerization
├── docker-compose.yaml       # Airflow services
├── dvc.yaml                  # DVC pipeline
└── README.md                 # This file
```

## Assignment Deliverables

### 1. GitHub Repository
- **Link**: [Your GitHub Repository URL]
- Clean project structure with essential files only
- Complete source code and configuration files

### 2. Docker Hub
- **Link**: https://hub.docker.com/r/iqraaazam/mlops-api
- **Image**: `iqraaazam/mlops-api:v1`
- Production-ready containerized API

### 3. Airflow DAG
- 5-task ML pipeline: health_check → load_data → train_model → save_model → log_results
- Automated workflow orchestration
- Error handling and monitoring

### 4. EC2 Public API URL
- **URL**: `http://[your-ec2-public-ip]:8000`
- Deployed on AWS EC2 free tier
- Accessible endpoints: `/health`, `/predict`, `/docs`

### 5. DVC Pipeline
- Data versioning with `dvc.yaml`
- Reproducible ML workflow
- Remote storage configuration

### 6. CI/CD Workflow
- GitHub Actions automation
- Automated testing, linting, and validation
- Multi-environment support

## Quick Start

### Local Development
```bash
git clone <repository-url>
cd mlops-assignment
pip install -r requirements.txt
python src/train.py
uvicorn api.main:app --reload
```

### Docker Deployment
```bash
docker pull iqraaazam/mlops-api:v1
docker run -p 8000:8000 iqraaazam/mlops-api:v1
```

### Airflow Setup
```bash
docker-compose up -d
# Access UI at http://localhost:8080
```

## Task Completion Status

- ✅ **Task 1**: Project Setup + Version Control (Git + DVC)
- ✅ **Task 2**: CI/CD Pipeline (GitHub Actions)
- ✅ **Task 3**: Docker Containerization
- ✅ **Task 4**: Airflow Pipeline
- ✅ **Task 5**: RESTful API (FastAPI)
- ✅ **Task 6**: AWS EC2 + S3 Deployment
- ✅ **Task 7**: Final Deliverables

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /docs` - Interactive API documentation

## Technologies Used

- **ML Framework**: scikit-learn, pandas, numpy
- **API**: FastAPI, uvicorn
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Apache Airflow
- **Version Control**: Git, DVC
- **CI/CD**: GitHub Actions
- **Cloud**: AWS EC2, S3
- **Testing**: pytest, flake8

## Problems Faced & Solutions

1. **Airflow Windows Compatibility**: Used Docker Compose approach
2. **Docker Permissions on EC2**: Added user to docker group
3. **Model Version Compatibility**: Pinned specific versions in requirements
4. **Port Conflicts**: Proper port mapping and container management

## Learning Summary

This project provided hands-on experience with:
- End-to-end MLOps pipeline development
- Container orchestration and cloud deployment
- Automated testing and continuous integration
- Version control for machine learning projects
- API development and documentation
- Infrastructure as Code practices