# Google Cloud Deployment Guide

## Overview

This guide provides a comprehensive plan to deploy the Interview Agent FastAPI server on Google Cloud with HTTPS/WSS support using Cloud Run, Load Balancer, and proper SSL termination.

## Architecture

```
Internet → Load Balancer (HTTPS/WSS) → Cloud Run (HTTP) → FastAPI Application
```

### Components
- **Cloud Run**: Serverless container hosting
- **Load Balancer**: HTTPS termination and WebSocket proxy
- **SSL Certificate**: Managed SSL for api.devbm.site
- **DNS**: Google Cloud DNS for domain management
- **Secret Manager**: Secure storage for API keys

## Prerequisites

### 1. Google Cloud Setup
```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project project-polaris-465312
```

### 2. Required Tools
```bash
# Install required tools
pip install terraform
pip install websockets requests
```

### 3. Domain Configuration
- Domain: `api.devbm.site`
- Ensure domain is accessible for DNS configuration

## Deployment Steps

### Step 1: Quick Deployment (Automated)

```bash
# Make deployment script executable
chmod +x deploy-to-gcp.sh

# Run automated deployment
./deploy-to-gcp.sh
```

### Step 2: Manual Deployment (Step-by-Step)

#### 2.1 Enable APIs
```bash
gcloud services enable \
  run.googleapis.com \
  compute.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  secretmanager.googleapis.com \
  dns.googleapis.com
```

#### 2.2 Create Service Account
```bash
gcloud iam service-accounts create interview-agent-sa \
  --display-name="Interview Agent Service Account"

gcloud projects add-iam-policy-binding project-polaris-465312 \
  --member="serviceAccount:interview-agent-sa@project-polaris-465312.iam.gserviceaccount.com" \
  --role="roles/run.invoker"

gcloud projects add-iam-policy-binding project-polaris-465312 \
  --member="serviceAccount:interview-agent-sa@project-polaris-465312.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

#### 2.3 Store Secrets
```bash
# Store Gemini API key
echo -n "your_gemini_api_key_here" | \
  gcloud secrets create gemini-api-key --data-file=-

# Store Supabase URL
echo -n "your_supabase_url_here" | \
  gcloud secrets create supabase-url --data-file=-

# Store Supabase Key
echo -n "your_supabase_key_here" | \
  gcloud secrets create supabase-key --data-file=-
```

#### 2.4 Build and Push Docker Image
```bash
# Configure Docker
gcloud auth configure-docker

# Tag and push image
docker tag new-server:latest \
  gcr.io/project-polaris-465312/interview-agent:latest

docker push gcr.io/project-polaris-465312/interview-agent:latest
```

#### 2.5 Deploy to Cloud Run
```bash
gcloud run services replace cloud-run-service.yaml \
  --region=us-central1 \
  --project=project-polaris-465312
```

#### 2.6 Deploy Load Balancer (Terraform)
```bash
# Initialize Terraform
terraform init

# Apply configuration
terraform apply -auto-approve
```

#### 2.7 Configure DNS
```bash
# Create DNS zone
gcloud dns managed-zones create interview-agent-zone \
  --dns-name="devbm.site." \
  --description="DNS zone for Interview Agent API"

# Get load balancer IP
LOAD_BALANCER_IP=$(terraform output -raw load_balancer_ip)

# Add A record
gcloud dns record-sets create api.devbm.site. \
  --zone=interview-agent-zone \
  --type=A \
  --ttl=300 \
  --rrdatas=$LOAD_BALANCER_IP
```

## Configuration Files

### 1. Cloud Run Service (`cloud-run-service.yaml`)
- Resource allocation: 2 CPU, 4GB RAM
- Timeout: 3600 seconds (1 hour)
- Health checks with proper startup delays
- Secret integration for API keys

### 2. Load Balancer (`terraform-load-balancer.tf`)
- Global load balancer with SSL termination
- WebSocket support
- HTTP to HTTPS redirect
- Managed SSL certificate

### 3. Docker Configuration
- Multi-stage build for optimization
- Health check with curl
- Proper resource limits

## HTTPS/WSS Support

### Load Balancer Configuration
The load balancer handles:
- **HTTPS termination**: SSL certificates managed by Google
- **WebSocket proxy**: Automatic WebSocket upgrade
- **HTTP redirect**: All HTTP traffic redirected to HTTPS

### WebSocket Support
- **WSS URL**: `wss://api.devbm.site/ws/{session_id}`
- **CORS**: Configured for WebSocket connections
- **Health check**: `/ws-health` endpoint for monitoring

## Monitoring and Testing

### 1. Health Checks
```bash
# Check service health
curl https://api.devbm.site/health

# Check WebSocket health
curl https://api.devbm.site/ws-health
```

### 2. Comprehensive Testing
```bash
# Run full test suite
python test-deployment.py https://api.devbm.site
```

### 3. Logs and Monitoring
```bash
# View Cloud Run logs
gcloud logs tail --service=interview-agent

# Monitor service
gcloud run services describe interview-agent --region=us-central1
```

## Performance Optimization

### 1. Model Preloading
- Models loaded at startup in background threads
- Global model cache prevents reloading
- Health check shows model readiness

### 2. Resource Allocation
- **CPU**: 2 vCPUs for model inference
- **Memory**: 4GB for large ML models
- **Concurrency**: 80 concurrent requests
- **Instances**: 1-10 auto-scaling

### 3. Caching Strategy
- Model cache persisted across requests
- Session data cached in memory
- Recording storage in Supabase

## Security Considerations

### 1. Secret Management
- API keys stored in Google Secret Manager
- No hardcoded secrets in code
- IAM-based access control

### 2. Network Security
- HTTPS/WSS enforced
- CORS properly configured
- Service account with minimal permissions

### 3. Data Protection
- Session data encrypted in transit
- Recording files stored securely
- User data handled according to privacy policies

## Troubleshooting

### 1. Common Issues

#### Service Not Starting
```bash
# Check logs
gcloud logs tail --service=interview-agent

# Check resource limits
gcloud run services describe interview-agent --region=us-central1
```

#### WebSocket Connection Issues
```bash
# Test WebSocket health
curl https://api.devbm.site/ws-health

# Check CORS configuration
# Verify domain in allowed origins
```

#### SSL Certificate Issues
```bash
# Check certificate status
gcloud compute ssl-certificates describe interview-agent-cert --global

# Verify DNS propagation
nslookup api.devbm.site
```

### 2. Performance Issues

#### Slow Session Creation
```bash
# Check model loading status
curl https://api.devbm.site/health

# Monitor resource usage
gcloud run services describe interview-agent --region=us-central1
```

#### Memory Issues
```bash
# Increase memory limit in cloud-run-service.yaml
# Restart service
gcloud run services replace cloud-run-service.yaml --region=us-central1
```

## Cost Optimization

### 1. Resource Management
- **Min instances**: 1 (keeps service warm)
- **Max instances**: 10 (prevents runaway costs)
- **CPU throttling**: Disabled for better performance

### 2. Monitoring Costs
```bash
# View billing
gcloud billing accounts list

# Set up billing alerts in Google Cloud Console
```

## Maintenance

### 1. Updates
```bash
# Update Docker image
docker build -t new-server:latest .
docker tag new-server:latest gcr.io/project-polaris-465312/interview-agent:latest
docker push gcr.io/project-polaris-465312/interview-agent:latest

# Update service
gcloud run services update interview-agent --image=gcr.io/project-polaris-465312/interview-agent:latest
```

### 2. Scaling
```bash
# Scale up for high traffic
gcloud run services update interview-agent \
  --max-instances=20 \
  --concurrency=100

# Scale down for cost optimization
gcloud run services update interview-agent \
  --max-instances=5 \
  --concurrency=50
```

## API Endpoints

### Production URLs
- **Health Check**: `https://api.devbm.site/health`
- **Session Creation**: `https://api.devbm.site/sessions/create`
- **WebSocket**: `wss://api.devbm.site/ws/{session_id}`
- **Job Templates**: `https://api.devbm.site/job-templates`
- **Models**: `https://api.devbm.site/models`

### Development URLs
- **Local**: `http://localhost:8000`
- **Cloud Run**: `https://interview-agent-xxxxx-uc.a.run.app`

## Support and Documentation

### 1. Google Cloud Documentation
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Load Balancer Documentation](https://cloud.google.com/load-balancing/docs)
- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)

### 2. Application Documentation
- [Docker Deployment Guide](DOCKER_DEPLOYMENT.md)
- [API Documentation](API_DOCUMENTATION.md)
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)

## Next Steps

1. **Deploy**: Run the deployment script
2. **Test**: Use the comprehensive test suite
3. **Monitor**: Set up monitoring and alerting
4. **Scale**: Adjust resources based on usage
5. **Optimize**: Fine-tune performance and costs
