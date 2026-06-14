param(
    [string]$DataRoot = "data/polyps",
    [switch]$KeepDownloads
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedDataRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $DataRoot))
$ExpectedPrefix = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot "data"))
if (-not $ResolvedDataRoot.StartsWith($ExpectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "DataRoot must resolve inside $ExpectedPrefix"
}

$ArchiveRoot = Join-Path $ResolvedDataRoot "archives"
$RawRoot = Join-Path $ResolvedDataRoot "raw"
$PreparedRoot = Join-Path $ResolvedDataRoot "prepared"
$TrainArchive = Join-Path $ArchiveRoot "PraNet-TrainDataset.zip"
$TestArchive = Join-Path $ArchiveRoot "PraNet-TestDataset.zip"

New-Item -ItemType Directory -Force -Path $ArchiveRoot | Out-Null

function Download-GoogleDriveFile {
    param([string]$Id, [string]$Output)
    if (Test-Path -LiteralPath $Output) {
        Write-Host "Using existing archive: $Output"
        return
    }
    $Uri = "https://drive.usercontent.google.com/download?id=$Id&export=download&confirm=t"
    Write-Host "Downloading $Output"
    Invoke-WebRequest -Uri $Uri -OutFile $Output -UseBasicParsing
}

Download-GoogleDriveFile "1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb" $TrainArchive
Download-GoogleDriveFile "1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao" $TestArchive

if (Test-Path -LiteralPath $RawRoot) {
    $ResolvedRawRoot = [System.IO.Path]::GetFullPath($RawRoot)
    if (-not $ResolvedRawRoot.StartsWith($ResolvedDataRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove raw directory outside DataRoot: $ResolvedRawRoot"
    }
    Remove-Item -LiteralPath $ResolvedRawRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $RawRoot | Out-Null
Expand-Archive -LiteralPath $TrainArchive -DestinationPath $RawRoot -Force
Expand-Archive -LiteralPath $TestArchive -DestinationPath $RawRoot -Force

if (Test-Path -LiteralPath $PreparedRoot) {
    $ResolvedPreparedRoot = [System.IO.Path]::GetFullPath($PreparedRoot)
    if (-not $ResolvedPreparedRoot.StartsWith($ResolvedDataRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove prepared directory outside DataRoot: $ResolvedPreparedRoot"
    }
    Remove-Item -LiteralPath $ResolvedPreparedRoot -Recurse -Force
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
python -m sam2unet.data prepare `
    (Join-Path $RawRoot "TrainDataset") `
    (Join-Path $RawRoot "TestDataset") `
    $PreparedRoot
if ($LASTEXITCODE -ne 0) {
    throw "Dataset preparation failed"
}

python -m sam2unet.data validate $PreparedRoot
if ($LASTEXITCODE -ne 0) {
    throw "Dataset validation failed"
}

if (-not $KeepDownloads) {
    Remove-Item -LiteralPath $RawRoot -Recurse -Force
    Remove-Item -LiteralPath $ArchiveRoot -Recurse -Force
}

Write-Host "Prepared polyp data at $PreparedRoot"
