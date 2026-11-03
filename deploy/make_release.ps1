# Packs the code for the cloud server into release.zip (no secrets, no call data).
# Run in PowerShell from the project folder:   powershell -ExecutionPolicy Bypass -File deploy\make_release.ps1
$ErrorActionPreference = "Stop"
$items = @("app", "scripts", "config", "tests", "deploy", "Dockerfile", "docker-compose.yml", "requirements-server.txt")
foreach ($i in $items) { if (-not (Test-Path $i)) { Write-Host "Missing: $i  (run this from the project folder)"; exit 1 } }
$tmp = Join-Path $env:TEMP "voice-release"
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory $tmp | Out-Null
foreach ($i in $items) { Copy-Item $i -Destination $tmp -Recurse }
# never ship caches or backups
Get-ChildItem $tmp -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem $tmp -Recurse -File -Include "*.bak", "*.pyc" | Remove-Item -Force
if (Test-Path release.zip) { Remove-Item release.zip }
Compress-Archive -Path (Join-Path $tmp "*") -DestinationPath release.zip
Remove-Item $tmp -Recurse -Force
$mb = [math]::Round((Get-Item release.zip).Length / 1MB, 1)
Write-Host "release.zip ready ($mb MB). Next: scp release.zip voice@SERVER_IP:"
