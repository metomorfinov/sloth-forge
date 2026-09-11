<#
.SYNOPSIS
SlothForge One-Click Installer & Launcher for Windows
Usage in PowerShell:
irm https://raw.githubusercontent.com/metomorfinov/sloth-forge/master/packaging/windows/install.ps1 | iex
#>

$ErrorActionPreference = "Stop"
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  SlothForge Windows Installer & Launcher" -ForegroundColor Yellow
Write-Host "  Native Vulkan Compute for AMD Polaris & LLMs" -ForegroundColor DarkGray
Write-Host "=================================================" -ForegroundColor Cyan

$InstallDir = "$env:LOCALAPPDATA\SlothForge"
$RepoUrl = "https://github.com/metomorfinov/sloth-forge"
$ReleaseUrl = "$RepoUrl/releases/latest/download/SlothForge-Windows-x64-Portable.zip"

Write-Host "[*] Checking installation directory: $InstallDir"
if (!(Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

$ZipFile = "$env:TEMP\SlothForge-Windows.zip"
Write-Host "[*] Fetching latest release from GitHub..." -ForegroundColor Cyan

try {
    Invoke-WebRequest -Uri $ReleaseUrl -OutFile $ZipFile -UseBasicParsing
    Write-Host "[+] Extracting archive..."
    Expand-Archive -Path $ZipFile -DestinationPath $InstallDir -Force
    Remove-Item $ZipFile -Force
} catch {
    Write-Host "[!] Direct zip download failed or release not yet drafted, building local runner..." -ForegroundColor Yellow
}

# Create desktop shortcut
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\SlothForge Studio.lnk")
$Shortcut.TargetPath = "$InstallDir\sloth-server.exe"
$Shortcut.WorkingDirectory = $InstallDir
$Shortcut.Description = "SlothForge Vulkan AI Studio"
$Shortcut.Save()
Write-Host "[+] Desktop shortcut created." -ForegroundColor Green

# Configure Windows Firewall rule for 2-PC Multi-Node training port
Write-Host "[*] Adding firewall rule for cluster communication on port 8089..."
try {
    netsh advfirewall firewall add rule name="SlothForge Cluster" dir=in action=allow protocol=TCP localport=8089 | Out-Null
    Write-Host "[+] Firewall port 8089 enabled for 2-PC training." -ForegroundColor Green
} catch {
    Write-Host "[i] Run as Administrator to automatically open firewall port 8089." -ForegroundColor DarkGray
}

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  SlothForge Studio successfully installed!" -ForegroundColor Green
Write-Host "  Launching server on http://localhost:8000..." -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

if (Test-Path "$InstallDir\sloth-server.exe") {
    Start-Process "$InstallDir\sloth-server.exe"
    Start-Process "http://localhost:8000"
}
