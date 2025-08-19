#!/bin/bash

# Google Cloud Deployment Script for Interview Agent
# Project: project-polaris-465312
# Domain: api.devbm.site

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="project-polaris-465312"
REGION="us-central1"
SERVICE_NAME="interview-agent"
DOMAIN="api.devbm.site"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Resolve script directory to make paths robust regardless of current working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}🚀 Starting Google Cloud Deployment for Interview Agent${NC}"
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Domain: ${DOMAIN}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service account to be ready
wait_for_service_account() {
    local service_account=$1
    local max_attempts=10
    local attempt=1
    
    echo -e "${YELLOW}⏳ Waiting for service account to be ready...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if gcloud iam service-accounts describe ${service_account} >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Service account is ready${NC}"
            return 0
        else
            echo -e "${YELLOW}Attempt ${attempt}/${max_attempts}: Service account not ready yet...${NC}"
            sleep 5
            attempt=$((attempt + 1))
        fi
    done
    
    echo -e "${RED}❌ Service account failed to become ready after ${max_attempts} attempts${NC}"
    return 1
}

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command_exists gcloud; then
    echo -e "${RED}❌ gcloud CLI not found. Please install Google Cloud SDK.${NC}"
    exit 1
fi

if ! command_exists terraform; then
    echo -e "${YELLOW}⚠️  Terraform not found. Will use gcloud commands instead.${NC}"
    USE_TERRAFORM=false
else
    USE_TERRAFORM=true
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"
echo ""

# Set project
echo -e "${YELLOW}🔧 Setting up Google Cloud project...${NC}"
gcloud config set project ${PROJECT_ID}
gcloud config set run/region ${REGION}

# Enable required APIs
echo -e "${YELLOW}🔌 Enabling required APIs...${NC}"
gcloud services enable \
    run.googleapis.com \
    compute.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    secretmanager.googleapis.com \
    dns.googleapis.com \
    --quiet

echo -e "${GREEN}✅ APIs enabled${NC}"
echo ""

# Create service account if it doesn't exist
echo -e "${YELLOW}👤 Setting up service account...${NC}"
if ! gcloud iam service-accounts describe interview-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com >/dev/null 2>&1; then
    gcloud iam service-accounts create interview-agent-sa \
        --display-name="Interview Agent Service Account"
    
    # Wait for service account to be ready
    if ! wait_for_service_account "interview-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com"; then
        echo -e "${RED}❌ Failed to create service account${NC}"
        exit 1
    fi
    
    # Retry IAM policy binding with multiple attempts
    for attempt in {1..3}; do
        echo -e "${YELLOW}Attempt ${attempt}/3: Adding IAM policies...${NC}"
        
        if gcloud projects add-iam-policy-binding ${PROJECT_ID} \
            --member="serviceAccount:interview-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
            --role="roles/run.invoker" 2>/dev/null; then
            echo -e "${GREEN}✅ Run invoker role added${NC}"
            break
        else
            echo -e "${YELLOW}⚠️  Attempt ${attempt} failed, retrying...${NC}"
            sleep 5
        fi
    done
    
    for attempt in {1..3}; do
        echo -e "${YELLOW}Attempt ${attempt}/3: Adding secret manager access...${NC}"
        
        if gcloud projects add-iam-policy-binding ${PROJECT_ID} \
            --member="serviceAccount:interview-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
            --role="roles/secretmanager.secretAccessor" 2>/dev/null; then
            echo -e "${GREEN}✅ Secret manager access added${NC}"
            break
        else
            echo -e "${YELLOW}⚠️  Attempt ${attempt} failed, retrying...${NC}"
            sleep 5
        fi
    done
    
    echo -e "${GREEN}✅ Service account created and configured${NC}"
else
    echo -e "${GREEN}✅ Service account already exists${NC}"
    
    # Ensure IAM policies are set even if service account exists
    echo -e "${YELLOW}🔧 Ensuring IAM policies are set...${NC}"
    
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:interview-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/run.invoker" 2>/dev/null || echo -e "${YELLOW}⚠️  Run invoker role already set${NC}"
    
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:interview-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/secretmanager.secretAccessor" 2>/dev/null || echo -e "${YELLOW}⚠️  Secret manager access already set${NC}"
fi
echo ""

# Build and push container image with Cloud Build
echo -e "${YELLOW}🏗️  Building container image with Cloud Build...${NC}"
echo -e "Using build context: ${SCRIPT_DIR}"
gcloud builds submit "${SCRIPT_DIR}" \
    --tag ${IMAGE_NAME}:latest \
    --project ${PROJECT_ID} \
    --quiet
echo -e "${GREEN}✅ Image built and pushed: ${IMAGE_NAME}:latest${NC}"
echo ""

# Store secrets in Secret Manager
echo -e "${YELLOW}🔐 Setting up secrets...${NC}"

# Check if secrets exist, if not prompt for values
if ! gcloud secrets describe gemini-api-key >/dev/null 2>&1; then
    echo -n "Enter Gemini API Key: "
    read -s GEMINI_API_KEY
    echo ""
    if echo -n "${GEMINI_API_KEY}" | gcloud secrets create gemini-api-key --data-file=- 2>/dev/null; then
        echo -e "${GREEN}✅ Gemini API key stored${NC}"
    else
        echo -e "${RED}❌ Failed to store Gemini API key${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Gemini API key already exists${NC}"
fi

if ! gcloud secrets describe supabase-url >/dev/null 2>&1; then
    echo -n "Enter Supabase URL: "
    read SUPABASE_URL
    if echo -n "${SUPABASE_URL}" | gcloud secrets create supabase-url --data-file=- 2>/dev/null; then
        echo -e "${GREEN}✅ Supabase URL stored${NC}"
    else
        echo -e "${RED}❌ Failed to store Supabase URL${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Supabase URL already exists${NC}"
fi

if ! gcloud secrets describe supabase-key >/dev/null 2>&1; then
    echo -n "Enter Supabase Key: "
    read -s SUPABASE_KEY
    echo ""
    if echo -n "${SUPABASE_KEY}" | gcloud secrets create supabase-key --data-file=- 2>/dev/null; then
        echo -e "${GREEN}✅ Supabase key stored${NC}"
    else
        echo -e "${RED}❌ Failed to store Supabase key${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Supabase key already exists${NC}"
fi
echo ""

# Deploy to Cloud Run
echo -e "${YELLOW}🚀 Deploying to Cloud Run...${NC}"

# Try YAML deployment first
if gcloud run services replace "${SCRIPT_DIR}/cloud-run-service-minimal.yaml" \
    --region=${REGION} \
    --project=${PROJECT_ID} 2>/dev/null; then
    echo -e "${GREEN}✅ Cloud Run service deployed successfully using YAML${NC}"
else
    echo -e "${YELLOW}⚠️  YAML deployment failed, trying direct deployment...${NC}"
    
    # Fallback to direct deployment
    gcloud run deploy interview-agent \
        --image gcr.io/${PROJECT_ID}/interview-agent:latest \
        --region=${REGION} \
        --platform managed \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --timeout 3600 \
        --concurrency 80 \
        --max-instances 10 \
        --min-instances 1 \
        --set-env-vars "FASTAPI_HOST=0.0.0.0,FASTAPI_RELOAD=false" \
        --set-secrets "GEMINI_API_KEY=gemini-api-key:latest,SUPABASE_URL=supabase-url:latest,SUPABASE_KEY=supabase-key:latest" \
        --service-account interview-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Cloud Run service deployed successfully using direct deployment${NC}"
    else
        echo -e "${RED}❌ Cloud Run deployment failed${NC}"
        exit 1
    fi
fi

# Get the Cloud Run URL
CLOUD_RUN_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region=${REGION} \
    --format="value(status.url)")

echo -e "${GREEN}✅ Cloud Run service deployed: ${CLOUD_RUN_URL}${NC}"
echo ""

# Deploy load balancer
if [ "$USE_TERRAFORM" = true ]; then
    echo -e "${YELLOW}🌐 Deploying load balancer with Terraform...${NC}"
    
    # Update the Cloud Run URL in Terraform configuration
    sed -i "s|interview-agent-xxxxx-uc.a.run.app|${CLOUD_RUN_URL#https://}|g" terraform-load-balancer.tf
    
    # Initialize and apply Terraform
    terraform init
    terraform apply -auto-approve
    
    LOAD_BALANCER_IP=$(terraform output -raw load_balancer_ip)
    echo -e "${GREEN}✅ Load balancer deployed: ${LOAD_BALANCER_IP}${NC}"
else
    echo -e "${YELLOW}🌐 Setting up load balancer with gcloud...${NC}"
    
    # Reserve static IP
    if ! gcloud compute addresses describe interview-agent-ip --global >/dev/null 2>&1; then
        gcloud compute addresses create interview-agent-ip \
            --global \
            --description="Static IP for Interview Agent Load Balancer"
        echo -e "${GREEN}✅ Static IP created${NC}"
    else
        echo -e "${GREEN}✅ Static IP already exists${NC}"
    fi
    
    LOAD_BALANCER_IP=$(gcloud compute addresses describe interview-agent-ip \
        --global \
        --format="value(address)")
    
    echo -e "${GREEN}✅ Static IP: ${LOAD_BALANCER_IP}${NC}"
    
    # Create SSL certificate
    if ! gcloud compute ssl-certificates describe interview-agent-cert --global >/dev/null 2>&1; then
        gcloud compute ssl-certificates create interview-agent-cert \
            --domains=${DOMAIN} \
            --global
        echo -e "${GREEN}✅ SSL certificate created${NC}"
    else
        echo -e "${GREEN}✅ SSL certificate already exists${NC}"
    fi
    
    echo -e "${YELLOW}⚠️  Note: Manual load balancer configuration required${NC}"
    echo -e "${YELLOW}   Please configure the load balancer manually in Google Cloud Console${NC}"
    echo -e "${YELLOW}   or install Terraform for automated setup${NC}"
fi
echo ""

# Configure DNS
echo -e "${YELLOW}📡 Configuring DNS...${NC}"

# Check if DNS zone exists
if ! gcloud dns managed-zones describe interview-agent-zone >/dev/null 2>&1; then
    if gcloud dns managed-zones create interview-agent-zone \
        --dns-name="devbm.site." \
        --description="DNS zone for Interview Agent API" 2>/dev/null; then
        echo -e "${GREEN}✅ DNS zone created${NC}"
    else
        echo -e "${YELLOW}⚠️  DNS zone creation failed (may already exist)${NC}"
    fi
else
    echo -e "${GREEN}✅ DNS zone already exists${NC}"
fi

# Add A record
if gcloud dns record-sets create ${DOMAIN}. \
    --zone=interview-agent-zone \
    --type=A \
    --ttl=300 \
    --rrdatas=${LOAD_BALANCER_IP} 2>/dev/null; then
    echo -e "${GREEN}✅ DNS A record created for ${DOMAIN}${NC}"
else
    echo -e "${YELLOW}⚠️  DNS A record creation failed (may already exist)${NC}"
    echo -e "${YELLOW}   Please manually add A record for ${DOMAIN} pointing to ${LOAD_BALANCER_IP}${NC}"
fi
echo ""

# Wait for DNS propagation
echo -e "${YELLOW}⏳ Waiting for DNS propagation (this may take a few minutes)...${NC}"
sleep 60

# Test the deployment
echo -e "${YELLOW}🧪 Testing deployment...${NC}"

# Test health endpoint
echo "Testing health endpoint..."
curl -f https://${DOMAIN}/health || echo -e "${RED}❌ Health check failed${NC}"

# Test WebSocket health endpoint
echo "Testing WebSocket health endpoint..."
curl -f https://${DOMAIN}/ws-health || echo -e "${RED}❌ WebSocket health check failed${NC}"

echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo -e "${GREEN}📋 Deployment Summary:${NC}"
echo "  • Cloud Run URL: ${CLOUD_RUN_URL}"
echo "  • Load Balancer IP: ${LOAD_BALANCER_IP}"
echo "  • Domain: https://${DOMAIN}"
echo "  • WebSocket URL: wss://${DOMAIN}/ws/{session_id}"
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo "  1. Wait for SSL certificate to be provisioned (may take 10-15 minutes)"
echo "  2. Test WebSocket connections"
echo "  3. Monitor the service with: gcloud run services describe ${SERVICE_NAME} --region=${REGION}"
echo "  4. View logs with: gcloud logs tail --service=${SERVICE_NAME}"
echo ""
echo -e "${GREEN}✅ Deployment script completed${NC}"
