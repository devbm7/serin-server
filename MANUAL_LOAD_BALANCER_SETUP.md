# Manual Load Balancer Setup Guide

Since Terraform is not available, you'll need to manually configure the load balancer in Google Cloud Console.

## Prerequisites

1. Cloud Run service deployed and running
2. Static IP address reserved
3. SSL certificate created
4. DNS zone configured

## Step-by-Step Manual Setup

### 1. Access Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Select your project: `project-polaris-465312`
3. Navigate to **Network Services** > **Load Balancing**

### 2. Create Load Balancer

1. Click **Create Load Balancer**
2. Select **HTTP(S) Load Balancing**
3. Click **Start configuration**

### 3. Configure Backend

1. **Backend configuration**:
   - Click **Backend configuration**
   - Click **Create a backend service**
   - **Name**: `interview-agent-backend`
   - **Backend type**: Select **Serverless network endpoint group**
   - **Region**: `us-central1`
   - Click **Create serverless NEG**
     - **Name**: `interview-agent-neg`
     - **Cloud Run service**: Select your deployed service
     - Click **Create**
   - **Port numbers**: `80`
   - **Protocol**: `HTTP`
   - Click **Create**

2. **Health check**:
   - Click **Health check**
   - **Name**: `interview-agent-health`
   - **Protocol**: `HTTP`
   - **Port**: `8000`
   - **Request path**: `/health`
   - Click **Create**

### 4. Configure Host and Path Rules

1. **Host and path rules**:
   - **Backend**: Select `interview-agent-backend`
   - Click **Continue**

### 5. Configure Frontend

1. **Frontend configuration**:
   - **Name**: `interview-agent-frontend`
   - **Protocol**: `HTTPS (includes HTTP/2)`
   - **IP address**: Select the reserved static IP (`interview-agent-ip`)
   - **Port**: `443`
   - **Certificate**: Select the created SSL certificate (`interview-agent-cert`)
   - Click **Done**

2. **HTTP to HTTPS redirect**:
   - Check **Enable HTTP to HTTPS redirect**
   - Click **Done**

### 6. Review and Create

1. Review the configuration
2. Click **Create**

## Verification

### 1. Test Load Balancer

```bash
# Test HTTPS endpoint
curl -I https://api.devbm.site/health

# Test HTTP redirect
curl -I http://api.devbm.site/health
```

### 2. Test WebSocket

```bash
# Test WebSocket health endpoint
curl https://api.devbm.site/ws-health
```

## Troubleshooting

### Common Issues

1. **SSL Certificate Not Provisioned**
   - Wait 10-15 minutes for certificate provisioning
   - Verify DNS is pointing to the load balancer IP

2. **Backend Service Unhealthy**
   - Check Cloud Run service is running
   - Verify health check path `/health` is accessible

3. **WebSocket Connection Issues**
   - Ensure load balancer supports WebSocket upgrade
   - Check CORS configuration in FastAPI

### Manual DNS Configuration

If DNS zone creation fails, manually configure DNS:

1. Go to your domain registrar
2. Add A record:
   - **Name**: `api`
   - **Value**: `[LOAD_BALANCER_IP]` (from the deployment script output)
   - **TTL**: `300`

### Alternative: Use Cloud Run Direct URL

If load balancer setup is complex, you can use the Cloud Run URL directly:

```bash
# Get the Cloud Run URL
gcloud run services describe interview-agent --region=us-central1 --format="value(status.url)"
```

The Cloud Run URL will be something like:
`https://interview-agent-xxxxx-uc.a.run.app`

## Next Steps

1. **Test the deployment**:
   ```bash
   python test-deployment.py https://api.devbm.site
   ```

2. **Monitor the service**:
   ```bash
   gcloud logs tail --service=interview-agent
   ```

3. **Set up monitoring** in Google Cloud Console

## Cost Considerations

- Load balancer: ~$18/month
- Static IP: ~$7/month (if not used)
- SSL certificate: Free (managed)
- Cloud Run: Pay per use

## Security Notes

1. **HTTPS enforcement**: All HTTP traffic redirected to HTTPS
2. **SSL certificate**: Managed by Google (auto-renewal)
3. **CORS**: Configured in FastAPI for WebSocket support
4. **IAM**: Service account with minimal permissions
