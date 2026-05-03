param(
    [Parameter(Mandatory = $true)]
    [string]$SessionName,

    [string]$InboxRoot = "<PROJECT_ROOT>\inbox",
    [string]$SessionsRoot = "<PROJECT_ROOT>\sessions"
)

$ErrorActionPreference = "Stop"

$sourceSessionDir = Join-Path $InboxRoot $SessionName
if (-not (Test-Path $sourceSessionDir)) {
    throw "Source session directory not found: $sourceSessionDir"
}

$targetSessionDir = Join-Path $SessionsRoot $SessionName
New-Item -ItemType Directory -Force -Path $targetSessionDir | Out-Null

# Required files for a captured session.
$pcapPath = Join-Path $sourceSessionDir "capture.pcapng"
if (-not (Test-Path $pcapPath)) {
    throw "capture.pcapng not found in: $sourceSessionDir"
}

$manifestPath = Join-Path $sourceSessionDir "session_manifest.json"
if (-not (Test-Path $manifestPath)) {
    throw "session_manifest.json not found in: $sourceSessionDir"
}

# Copy top-level session artefacts produced by the Ubuntu capture workflow.
$topLevelFiles = @(
    "capture.pcapng",
    "summary.txt",
    "summary.json",
    "session_manifest.json",
    "capture_meta.txt",
    "cbf_by_peer.csv"
)

foreach ($name in $topLevelFiles) {
    $src = Join-Path $sourceSessionDir $name
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $targetSessionDir $name) -Force
    }
}

# Copy per-peer CBF CSV files, for example capture_rtax52_a.csv.
Get-ChildItem -Path $sourceSessionDir -Filter "capture_*.csv" | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $targetSessionDir $_.Name) -Force
}

# Read the manifest to discover peer IDs and create a consistent session layout.
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
if (-not $manifest.peer_mapping) {
    throw "peer_mapping missing in session_manifest.json"
}

# Create canonical subdirectories for subsequent MATLAB and Python processing.
$dirs = @(
    (Join-Path $targetSessionDir "peers"),
    (Join-Path $targetSessionDir "fused"),
    (Join-Path $targetSessionDir "reports"),
    (Join-Path $targetSessionDir "logs")
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

# Create per-peer work directories and keep a small peer manifest for debugging.
$peerIds = @($manifest.peer_mapping.PSObject.Properties.Name)

foreach ($peerId in $peerIds) {
    $peerDir = Join-Path (Join-Path $targetSessionDir "peers") $peerId
    New-Item -ItemType Directory -Force -Path $peerDir | Out-Null

    $csvName = "capture_{0}.csv" -f $peerId
    $csvPath = Join-Path $targetSessionDir $csvName

    $active = $true
    try {
        $active = [bool]$manifest.peer_mapping.$peerId.active_in_this_session
    } catch {
        $active = $true
    }

    if ($active -and -not (Test-Path $csvPath)) {
        Write-Warning "Active peer '$peerId' has no CSV at $csvPath"
    }

    $peerInfo = $manifest.peer_mapping.$peerId | ConvertTo-Json -Depth 8
    $peerMetaPath = Join-Path $peerDir "peer_info.json"
    Set-Content -Path $peerMetaPath -Value $peerInfo -Encoding UTF8
}

Write-Host "[OK] prepared session directory: $targetSessionDir"
Write-Host "[INFO] peers found: $($peerIds -join ', ')"
