param(
    [Parameter(Mandatory = $true)]
    [string]$SessionName,

    [string]$BSRoot = "<PROJECT_ROOT>",
    [string]$MatlabExe = "<MATLAB_EXE>",
    [string]$PythonExe = "python",
    [string]$ModelPath = "",
    [switch]$SkipPrepare,
    [switch]$SkipExtract,
    [switch]$SkipBatch,
    [switch]$SkipFuse,
    [switch]$SkipInfer
)

$ErrorActionPreference = "Stop"

function Convert-ToMatlabPath {
    param([string]$PathText)
    return ($PathText -replace "\\", "/")
}

function Invoke-MatlabBatch {
    param(
        [string]$MatlabExePath,
        [string]$CommandText
    )
    Write-Host "[MATLAB] $CommandText"
    & $MatlabExePath -batch $CommandText
    if ($LASTEXITCODE -ne 0) {
        throw "MATLAB command failed with exit code $LASTEXITCODE"
    }
}

$InboxRoot = Join-Path $BSRoot "inbox"
$SessionsRoot = Join-Path $BSRoot "sessions"
$MatlabRoot = Join-Path $BSRoot "matlab"
$PythonRoot = Join-Path $BSRoot "python"

$PrepareScript = Join-Path $BSRoot "prepare_session_v2.ps1"
$SessionDir = Join-Path $SessionsRoot $SessionName
$ManifestPath = Join-Path $SessionDir "session_manifest.json"
$FuseScript = Join-Path $PythonRoot "fuse_batches_multipeer.py"
$InferScript = Join-Path $PythonRoot "infer_multipeer_session.py"

if (-not (Test-Path $MatlabExe)) {
    throw "MATLAB executable not found: $MatlabExe"
}
if (-not (Test-Path $PrepareScript)) {
    throw "prepare_session_v2.ps1 not found: $PrepareScript"
}
if (-not (Test-Path $FuseScript)) {
    throw "fuse_batches_multipeer.py not found: $FuseScript"
}
if (-not (Test-Path $InferScript)) {
    throw "infer_multipeer_session.py not found: $InferScript"
}

# Step 1: prepare the Windows session directory from the inbox copy.
if (-not $SkipPrepare) {
    Write-Host "[STEP] prepare session"
    & powershell -ExecutionPolicy Bypass -File $PrepareScript -SessionName $SessionName -InboxRoot $InboxRoot -SessionsRoot $SessionsRoot
    if ($LASTEXITCODE -ne 0) {
        throw "prepare_session_v2.ps1 failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "[STEP] prepare session skipped"
}

if (-not (Test-Path $ManifestPath)) {
    throw "session_manifest.json not found after prepare: $ManifestPath"
}

$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$PeerIds = @($Manifest.peer_mapping.PSObject.Properties.Name)
if ($PeerIds.Count -eq 0) {
    throw "No peer_mapping entries found in manifest"
}

$ActivePeerIds = @()
foreach ($peerId in $PeerIds) {
    try {
        if ([bool]$Manifest.peer_mapping.$peerId.active_in_this_session) {
            $ActivePeerIds += $peerId
        }
    }
    catch {
        $ActivePeerIds += $peerId
    }
}
if ($ActivePeerIds.Count -eq 0) {
    $ActivePeerIds = $PeerIds
}

Write-Host "[INFO] session_dir: $SessionDir"
Write-Host "[INFO] active_peers: $($ActivePeerIds -join ', ')"

$SessionDirMat = Convert-ToMatlabPath $SessionDir
$MatlabRootMat = Convert-ToMatlabPath $MatlabRoot

# Step 2: extract BFA features for each active peer.
if (-not $SkipExtract) {
    foreach ($peerId in $ActivePeerIds) {
        Write-Host "[STEP] extract BFA for $peerId"
        $cmd = @"
addpath('$MatlabRootMat');
out = extract_bfa_session('$SessionDirMat', '$peerId');
disp(out);
"@
        Invoke-MatlabBatch -MatlabExePath $MatlabExe -CommandText $cmd
    }
}
else {
    Write-Host "[STEP] extract BFA skipped"
}

# Step 3: convert per-peer BFA features into fixed windows and batches.
if (-not $SkipBatch) {
    foreach ($peerId in $ActivePeerIds) {
        Write-Host "[STEP] build batches for $peerId"
        $cmd = @"
addpath('$MatlabRootMat');
out = bfa_to_batches_session('$SessionDirMat', '$peerId');
disp(out);
"@
        Invoke-MatlabBatch -MatlabExePath $MatlabExe -CommandText $cmd
    }
}
else {
    Write-Host "[STEP] build batches skipped"
}

# Step 4: fuse per-peer batches into a multi-peer tensor representation.
if (-not $SkipFuse) {
    Write-Host "[STEP] fuse multipeer batches"
    & $PythonExe $FuseScript --session-dir $SessionDir
    if ($LASTEXITCODE -ne 0) {
        throw "fuse_batches_multipeer.py failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "[STEP] fuse multipeer batches skipped"
}

# Step 5: optionally run session-level inference.
if (-not $SkipInfer) {
    Write-Host "[STEP] infer multipeer session"
    if ([string]::IsNullOrWhiteSpace($ModelPath)) {
        Write-Host "[INFO] ModelPath is empty; infer script will run in summary-only mode"
        & $PythonExe $InferScript --session-dir $SessionDir
    }
    else {
        & $PythonExe $InferScript --session-dir $SessionDir --model-path $ModelPath
    }

    if ($LASTEXITCODE -ne 0) {
        throw "infer_multipeer_session.py failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "[STEP] infer skipped"
}

Write-Host "[OK] run_session_pipeline_v2 completed for $SessionName"
