# First-time setup script for MLOps_Project
# Usage (PowerShell):
#   Set-ExecutionPolicy -Scope Process Bypass; ./scripts/first_run.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info([string]$msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Success([string]$msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg) { Write-Host "[ERR]  $msg" -ForegroundColor Red }

function Require-Cli([string]$cmd, [string]$friendly) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        throw "Missing prerequisite: $friendly ('$cmd') is not available in PATH."
    }
}

function Get-ComposeCmd() {
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        # Prefer modern 'docker compose'
        try {
            docker compose version | Out-Null
            return @('docker', 'compose')
        } catch {
            # Fall back to legacy 'docker-compose'
        }
    }
    if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
        return @('docker-compose')
    }
    throw "Neither 'docker compose' nor 'docker-compose' is available. Install Docker Desktop."
}

function Ensure-EnvFile([string]$path) {
    if (-not (Test-Path $path)) {
        Write-Info "Creating .env with AWS credential placeholders at $path"
        @(
            "# Populate your AWS credentials for the API to load model/data from S3",
            "AWS_ACCESS_KEY_ID=",
            "AWS_SECRET_ACCESS_KEY="
        ) | Set-Content -Encoding UTF8 $path
        Write-Warn ".env created with empty values. Update AWS credentials if you want the API to load from S3."
    } else {
        Write-Info ".env already exists."
    }
}

function Ensure-DataDirs() {
    $dirs = @(
        "data",
        "data/raw",
        "monitoring/evidently/workspace",
        "monitoring/evidently/reports",
        "mlruns",
        "mlflow-artifacts"
    )
    foreach ($d in $dirs) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }
}

function Start-Services() {
    $compose = Get-ComposeCmd
    Write-Info "Building and starting Docker services (this may take several minutes on first run)..."
    & $compose build
    & $compose up -d
    Write-Success "Docker services are starting in the background."
}

function Wait-For-Http([string]$url, [int]$timeoutSec = 420, [int]$intervalSec = 6) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $timeoutSec) {
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
                return $true
            }
        } catch { }
        Start-Sleep -Seconds $intervalSec
    }
    return $false
}

function Ensure-Python() {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Warn "Python not found. Skipping Evidently workspace initialization."
        return $false
    }
    return $true
}

function Init-EvidentlyWorkspace() {
    Write-Info "Initializing Evidently workspace (local files)..."
    # Use a lightweight ephemeral venv to avoid polluting global Python
    $venvPath = ".first-run-venv"
    python -m venv $venvPath
    $pip = Join-Path $venvPath "Scripts/pip.exe"
    $py  = Join-Path $venvPath "Scripts/python.exe"
    & $pip install --upgrade pip | Out-Null
    & $pip install evidently==0.4.33 pandas -q
    & $py scripts/setup_evidently_workspace.py
    Write-Success "Evidently workspace initialized in 'monitoring/evidently/workspace'."
}

try {
    Write-Info "Checking prerequisites..."
    Require-Cli -cmd docker -friendly "Docker Desktop"
    # Compose cmd is resolved later; this call also validates availability
    [void](Get-ComposeCmd)
    Write-Success "Docker is available."

    Ensure-EnvFile -path ".env"
    Ensure-DataDirs

    Start-Services

    Write-Info "Waiting for API health at http://localhost:8000/health ..."
    if (Wait-For-Http -url "http://localhost:8000/health") {
        Write-Success "API is responding. Docs: http://localhost:8000/docs"
    } else {
        Write-Warn "API did not become healthy within the expected time. It may still be downloading models."
    }

    if (Ensure-Python) {
        try { Init-EvidentlyWorkspace } catch { Write-Warn "Evidently init failed: $($_.Exception.Message)" }
    }

    Write-Host ""
    Write-Success "First-time setup complete."
    Write-Host "Access your services:" -ForegroundColor Cyan
    Write-Host "  UI (Streamlit):           http://localhost:8501"
    Write-Host "  Resume Matcher API Docs:  http://localhost:8000/docs"
    Write-Host "  MLflow UI:                http://localhost:5000"
    Write-Host "  Prometheus UI:            http://localhost:9090"
    Write-Host "  Grafana UI:               http://localhost:3000 (admin/admin)"
    Write-Host ""
    Write-Host "Note: Update .env with AWS credentials so the API can load the model and embeddings from S3." -ForegroundColor Yellow
}
catch {
    Write-Err $_.Exception.Message
    exit 1
}


