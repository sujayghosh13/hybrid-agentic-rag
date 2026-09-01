$ErrorActionPreference = "Stop"

# Hybrid Agentic RAG - Phase 1 dataset downloader
# Downloads official Docker and Kubernetes documentation as HTML
# into data/raw for the ingestion pipeline.

$projectRoot = $PSScriptRoot
$rawDir = Join-Path $projectRoot "data\raw"

New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

$docs = @(
    @{
        Name = "docker-networking.html"
        Url  = "https://docs.docker.com/engine/network/"
    },
    @{
        Name = "docker-bridge-network.html"
        Url  = "https://docs.docker.com/engine/network/drivers/bridge/"
    },
    @{
        Name = "kubernetes-pods.html"
        Url  = "https://kubernetes.io/docs/concepts/workloads/pods/"
    },
    @{
        Name = "kubernetes-deployments.html"
        Url  = "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/"
    },
    @{
        Name = "kubernetes-workloads.html"
        Url  = "https://kubernetes.io/docs/concepts/workloads/"
    }
)

foreach ($doc in $docs) {
    $destination = Join-Path $rawDir $doc.Name
    Write-Host "Downloading $($doc.Name)..."

    Invoke-WebRequest `
        -Uri $doc.Url `
        -OutFile $destination `
        -UseBasicParsing

    Write-Host "Saved: $destination"
}

Write-Host ""
Write-Host "Done. Dataset files are in:"
Write-Host $rawDir
Write-Host ""
Get-ChildItem $rawDir | Select-Object Name, Length
