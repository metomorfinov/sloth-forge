; SlothForge Windows Inno Setup Script
; Generates SlothForge-Setup-Windows-x64.exe

#define MyAppName "SlothForge Studio"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "metomorfinov"
#define MyAppURL "https://github.com/metomorfinov/sloth-forge"
#define MyAppExeName "sloth-server.exe"

[Setup]
AppId={{D37E6F4A-98C1-42C4-A7E9-8F79658E1B93}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\SlothForge
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
OutputDir=bin
OutputBaseFilename=SlothForge-Setup-Windows-x64
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "Запускать SlothForge при старте Windows"; Flags: unchecked

[Files]
; Main executable and libraries
Source: "target\x86_64-pc-windows-msvc\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "crates\sloth-vulkan-sys\bin\libsloth_vulkan.dll"; DestDir: "{app}"; Flags: ignoreversion external; Check: FileExists('crates\sloth-vulkan-sys\bin\libsloth_vulkan.dll')
Source: "frontend\dist\*"; DestDir: "{app}\frontend\dist"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "models\README.txt"; DestDir: "{app}\models"; Flags: ignoreversion createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\target"
Type: filesandordirs; Name: "{app}\logs"
