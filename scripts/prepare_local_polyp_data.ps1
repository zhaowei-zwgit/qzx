param(
    [string]$DataRoot = "data/polyps",
    [switch]$KeepRaw
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
$Archives = @{
    Pranet = Join-Path $ArchiveRoot "PraNet-TrainDataset.zip"
    Kvasir = Join-Path $ArchiveRoot "kvasir-seg.zip"
    Clinic = Join-Path $ArchiveRoot "CVC-ClinicDB.zip"
}

foreach ($Archive in $Archives.Values) {
    if (-not (Test-Path -LiteralPath $Archive)) {
        throw "Missing required archive: $Archive"
    }
}

function Remove-DataDirectory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $Resolved = [System.IO.Path]::GetFullPath($Path)
    if (-not $Resolved.StartsWith($ResolvedDataRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove directory outside DataRoot: $Resolved"
    }
    Remove-Item -LiteralPath $Resolved -Recurse -Force
}

Remove-DataDirectory $RawRoot
Remove-DataDirectory $PreparedRoot

$PranetRaw = Join-Path $RawRoot "pranet"
$KvasirRaw = Join-Path $RawRoot "kvasir"
$ClinicRaw = Join-Path $RawRoot "clinic"
New-Item -ItemType Directory -Force -Path $PranetRaw, $KvasirRaw, $ClinicRaw | Out-Null

Expand-Archive -LiteralPath $Archives.Pranet -DestinationPath $PranetRaw -Force
Expand-Archive -LiteralPath $Archives.Kvasir -DestinationPath $KvasirRaw -Force
Expand-Archive -LiteralPath $Archives.Clinic -DestinationPath $ClinicRaw -Force

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
python -m sam2unet.data prepare-from-full `
    (Join-Path $PranetRaw "TrainDataset") `
    (Join-Path $KvasirRaw "Kvasir-SEG") `
    (Join-Path $ClinicRaw "PNG") `
    $PreparedRoot
if ($LASTEXITCODE -ne 0) {
    throw "Dataset preparation failed"
}

python -m sam2unet.data validate $PreparedRoot --basic-only
if ($LASTEXITCODE -ne 0) {
    throw "Dataset validation failed"
}

if (-not $KeepRaw) {
    Remove-DataDirectory $RawRoot
}

Write-Host "Prepared polyp data at $PreparedRoot"
