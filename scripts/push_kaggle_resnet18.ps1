param(
    [ValidateRange(60, 43200)]
    [int]$TimeoutSeconds = 43200
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$NotebookPath = Join-Path $ProjectRoot "notebooks\resnet18_architecture_ablation_kaggle.ipynb"
$MetadataPath = Join-Path $ProjectRoot "kaggle\resnet18-kernel-metadata.json"
$PublishDirectory = Join-Path $ProjectRoot ".kaggle-sync\publish-resnet18"
$PublishNotebook = Join-Path $PublishDirectory "resnet18-ablation.ipynb"
$PublishMetadata = Join-Path $PublishDirectory "kernel-metadata.json"

if (-not (Test-Path -LiteralPath $NotebookPath -PathType Leaf)) {
    throw "Notebook not found: $NotebookPath"
}
if (-not (Test-Path -LiteralPath $MetadataPath -PathType Leaf)) {
    throw "Kaggle metadata not found: $MetadataPath"
}
if (-not (Test-Path -LiteralPath "$env:USERPROFILE\.kaggle\access_token" -PathType Leaf)) {
    throw "Kaggle access token not found."
}

New-Item -ItemType Directory -Force -Path $PublishDirectory | Out-Null
Copy-Item -LiteralPath $NotebookPath -Destination $PublishNotebook -Force
Copy-Item -LiteralPath $MetadataPath -Destination $PublishMetadata -Force

$PushOutput = python -m kaggle kernels push --path $PublishDirectory --timeout $TimeoutSeconds 2>&1
$PushExitCode = $LASTEXITCODE
$PushOutput | ForEach-Object { Write-Host $_ }
$PushText = $PushOutput | Out-String
if ($PushExitCode -ne 0 -or $PushText -match "(?im)kernel push error|^error:") {
    throw "Kaggle ResNet18 notebook push failed."
}

Write-Host "Published to https://www.kaggle.com/code/plomo02/project2-cv-resnet18-ablation"
