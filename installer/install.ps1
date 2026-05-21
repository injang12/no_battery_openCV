param(
    [string]$SourceDir = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"

$appName = "CameraCalibration"
$displayName = "Camera Calibration"
$programFiles = [Environment]::GetFolderPath("ProgramFiles")
$commonPrograms = [Environment]::GetFolderPath("CommonPrograms")
$commonDesktop = [Environment]::GetFolderPath("CommonDesktopDirectory")
if ([string]::IsNullOrWhiteSpace($commonDesktop)) {
    $commonDesktop = Join-Path $env:PUBLIC "Desktop"
}

$installDir = Join-Path $programFiles $displayName
$targetExe = Join-Path $installDir "$appName.exe"
$sourceArchive = Join-Path $SourceDir "$appName.zip"
$dataDir = Join-Path ([Environment]::GetFolderPath("CommonApplicationData")) $appName
$startMenuDir = Join-Path $commonPrograms $displayName
$desktopShortcut = Join-Path $commonDesktop "$displayName.lnk"
$startMenuShortcut = Join-Path $startMenuDir "$displayName.lnk"
$uninstallScript = Join-Path $installDir "uninstall.ps1"
$uninstallShortcut = Join-Path $startMenuDir "Uninstall $displayName.lnk"

$legacyUserInstallDir = Join-Path $env:LOCALAPPDATA "Programs\CameraCalibration"
$legacyNoBatteryInstallDir = Join-Path $env:LOCALAPPDATA "Programs\NoBatteryOpenCV"
$legacyUserStartMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Camera Calibration"
$legacyNoBatteryStartMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\NoBattery OpenCV"
$legacyUserDesktopShortcut = Join-Path ([Environment]::GetFolderPath("DesktopDirectory")) "Camera Calibration.lnk"
$legacyNoBatteryDesktopShortcut = Join-Path ([Environment]::GetFolderPath("DesktopDirectory")) "NoBattery OpenCV.lnk"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Copy-InstallerPayloadForElevation {
    $stagingDir = Join-Path ([System.IO.Path]::GetTempPath()) ("CameraCalibrationInstaller-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null
    Copy-Item -LiteralPath (Join-Path $SourceDir "$appName.zip") -Destination (Join-Path $stagingDir "$appName.zip") -Force
    Copy-Item -LiteralPath (Join-Path $SourceDir "install.ps1") -Destination (Join-Path $stagingDir "install.ps1") -Force
    Copy-Item -LiteralPath (Join-Path $SourceDir "install.cmd") -Destination (Join-Path $stagingDir "install.cmd") -Force
    return $stagingDir
}

if (-not (Test-Path $sourceArchive)) {
    throw "Application archive was not found: $sourceArchive"
}

if (-not (Test-IsAdministrator)) {
    $stagingDir = Copy-InstallerPayloadForElevation
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$(Join-Path $stagingDir "install.ps1")`"",
        "-SourceDir", "`"$stagingDir`""
    )
    if ($NoLaunch) {
        $arguments += "-NoLaunch"
    }

    $process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -Verb RunAs -Wait -PassThru
    Remove-Item -LiteralPath $stagingDir -Recurse -Force -ErrorAction SilentlyContinue
    exit $process.ExitCode
}

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
New-Item -ItemType Directory -Force -Path $startMenuDir | Out-Null
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

& icacls.exe $dataDir /grant "*S-1-5-32-545:(OI)(CI)M" /T | Out-Null

Remove-Item -LiteralPath $legacyUserDesktopShortcut -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $legacyNoBatteryDesktopShortcut -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $legacyUserStartMenuDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $legacyNoBatteryStartMenuDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $legacyUserInstallDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $legacyNoBatteryInstallDir -Recurse -Force -ErrorAction SilentlyContinue

Get-ChildItem -LiteralPath $installDir -Force | Remove-Item -Recurse -Force
Expand-Archive -LiteralPath $sourceArchive -DestinationPath $installDir -Force

if (-not (Test-Path $targetExe)) {
    throw "Installed executable was not found: $targetExe"
}

@"
`$ErrorActionPreference = "Stop"
`$displayName = "$displayName"
`$installDir = "$installDir"
`$startMenuDir = "$startMenuDir"
`$desktopShortcut = "$desktopShortcut"
Remove-Item -LiteralPath `$desktopShortcut -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath `$startMenuDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath `$installDir -Recurse -Force -ErrorAction SilentlyContinue
"@ | Set-Content -LiteralPath $uninstallScript -Encoding UTF8

function New-AppShortcut {
    param(
        [string]$ShortcutPath,
        [string]$TargetPath,
        [string]$WorkingDirectory,
        [string]$Arguments = ""
    )

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.IconLocation = $TargetPath
    $shortcut.Arguments = $Arguments
    $shortcut.Save()
}

New-AppShortcut -ShortcutPath $startMenuShortcut -TargetPath $targetExe -WorkingDirectory $installDir
New-AppShortcut -ShortcutPath $desktopShortcut -TargetPath $targetExe -WorkingDirectory $installDir
New-AppShortcut -ShortcutPath $uninstallShortcut -TargetPath "powershell.exe" -WorkingDirectory $installDir -Arguments "-NoProfile -ExecutionPolicy Bypass -File `"$uninstallScript`""

if (-not $NoLaunch) {
    Start-Process -FilePath $targetExe -WorkingDirectory $installDir
}
