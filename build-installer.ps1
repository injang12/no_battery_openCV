param(
    [string]$RuntimeIdentifier = "win-x64",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$localDotnet = Join-Path $root ".tools\dotnet\dotnet.exe"
$dotnet = if (Test-Path $localDotnet) { $localDotnet } else { "dotnet" }
$project = Join-Path $root "src\NoBatteryOpenCV.Wpf\NoBatteryOpenCV.Wpf.csproj"
$publishRoot = Join-Path $root "artifacts\publish\$RuntimeIdentifier"
$publishDir = Join-Path $publishRoot ("build-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
$installerRoot = Join-Path $root "artifacts\installer"
$payloadDir = Join-Path $installerRoot "payload"
$appExe = "CameraCalibration.exe"
$appArchive = "CameraCalibration.zip"
$setupPath = Join-Path $installerRoot "CameraCalibrationSetup-$RuntimeIdentifier.exe"
$sedPath = Join-Path $installerRoot "CameraCalibration-$RuntimeIdentifier.sed"

New-Item -ItemType Directory -Force -Path $publishRoot, $publishDir, $installerRoot, $payloadDir | Out-Null

$payloadArchive = Join-Path $payloadDir $appArchive
Remove-Item -LiteralPath $payloadArchive -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $setupPath -Force -ErrorAction SilentlyContinue

& $dotnet publish $project `
    -c $Configuration `
    -r $RuntimeIdentifier `
    --self-contained true `
    -p:PublishSingleFile=false `
    -p:DebugType=None `
    -p:DebugSymbols=false `
    -o $publishDir
if ($LASTEXITCODE -ne 0) {
    throw "dotnet publish failed with exit code $LASTEXITCODE"
}

Compress-Archive -Path (Join-Path $publishDir "*") -DestinationPath $payloadArchive -Force
Copy-Item -Path (Join-Path $root "installer\install.cmd") -Destination (Join-Path $payloadDir "install.cmd") -Force
Copy-Item -Path (Join-Path $root "installer\install.ps1") -Destination (Join-Path $payloadDir "install.ps1") -Force

$setupPathForSed = [System.IO.Path]::GetFullPath($setupPath)
$payloadDirForSed = [System.IO.Path]::GetFullPath($payloadDir).TrimEnd("\") + "\"

@"
[Version]
Class=IEXPRESS
SEDVersion=3
[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=0
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=%InstallPrompt%
DisplayLicense=%DisplayLicense%
FinishMessage=%FinishMessage%
TargetName=%TargetName%
FriendlyName=%FriendlyName%
AppLaunched=%AppLaunched%
PostInstallCmd=%PostInstallCmd%
AdminQuietInstCmd=%AdminQuietInstCmd%
UserQuietInstCmd=%UserQuietInstCmd%
SourceFiles=SourceFiles
[Strings]
InstallPrompt=
DisplayLicense=
FinishMessage=Camera Calibration installation finished.
TargetName=$setupPathForSed
FriendlyName=Camera Calibration
AppLaunched=install.cmd
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
FILE0=$appArchive
FILE1=install.cmd
FILE2=install.ps1
[SourceFiles]
SourceFiles0=$payloadDirForSed
[SourceFiles0]
%FILE0%=
%FILE1%=
%FILE2%=
"@ | Set-Content -LiteralPath $sedPath -Encoding ASCII

& "$env:WINDIR\System32\iexpress.exe" /N $sedPath
if ($LASTEXITCODE -ne 0) {
    throw "IExpress failed with exit code $LASTEXITCODE"
}

while (Get-Process -Name makecab -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 1
}

while (Get-Process -Name iexpress -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 1
}

if (-not (Test-Path $setupPath)) {
    throw "Installer was not created: $setupPath"
}

Write-Host "Installer created: $setupPath"
