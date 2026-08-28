param(
    [int]$TimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$NotebookPath = Join-Path $ProjectRoot "notebooks\project_2_cores_mde_kaggle.ipynb"
$MetadataPath = Join-Path $ProjectRoot "kaggle\kernel-metadata.json"
$PublishDirectory = Join-Path $ProjectRoot ".kaggle-sync\publish"
$PublishNotebook = Join-Path $PublishDirectory "project2-cv.ipynb"
$PublishMetadata = Join-Path $PublishDirectory "kernel-metadata.json"

if (-not (Test-Path -LiteralPath $NotebookPath -PathType Leaf)) {
    throw "Notebook not found: $NotebookPath"
}
if (-not (Test-Path -LiteralPath $MetadataPath -PathType Leaf)) {
    throw "Kaggle metadata not found: $MetadataPath"
}
if (-not (Test-Path -LiteralPath "$env:USERPROFILE\.kaggle\access_token" -PathType Leaf)) {
    throw "Kaggle access token not found. Expected %USERPROFILE%\.kaggle\access_token"
}

New-Item -ItemType Directory -Force -Path $PublishDirectory | Out-Null
Copy-Item -LiteralPath $NotebookPath -Destination $PublishNotebook -Force
Copy-Item -LiteralPath $MetadataPath -Destination $PublishMetadata -Force

$PushOutput = python -m kaggle kernels push --path $PublishDirectory --timeout $TimeoutSeconds 2>&1
$PushExitCode = $LASTEXITCODE
$PushOutput | ForEach-Object { Write-Host $_ }
$PushText = $PushOutput | Out-String

if ($PushExitCode -ne 0 -or $PushText -match "(?im)kernel push error|^error:") {
    throw "Kaggle notebook push failed."
}

Write-Host "Published notebook to https://www.kaggle.com/code/plomo02/project2-cv"
