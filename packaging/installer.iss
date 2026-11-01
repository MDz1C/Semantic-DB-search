; Inno Setup 6.7 script for the Semantic DB Search installer.
;
; packaging\build.ps1 compiles it after PyInstaller has produced dist\SemanticDBSearch:
;   ISCC.exe /DAppVersion=1.0.0 packaging\installer.iss
; -> dist\SemanticDBSearch-Setup-<version>.exe
;
; Installs per user without administrator rights (an all-users install can be chosen in the
; privileges dialog). User data lives in %LOCALAPPDATA%\SemanticDBSearch, never in the program
; folder; the uninstaller offers to delete it.

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "Semantic DB Search"
#define AppPublisher "Manuel Dzimah"
#define AppExeName "SemanticDBSearch.exe"
#define DataFolderName "SemanticDBSearch"

[Setup]
; The AppId identifies the app across upgrades: never change it.
AppId={{F8BF84DF-0C4B-4BEC-99C3-0CE5BF936B8C}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppCopyright=Copyright (c) 2026 {#AppPublisher}
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup
; Every path below is relative to the repository root.
SourceDir=..
OutputDir=dist
OutputBaseFilename=SemanticDBSearch-Setup-{#AppVersion}
DefaultDirName={autopf}\Semantic DB Search
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern dynamic
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
LicenseFile=LICENSE
Compression=lzma2/ultra64
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; An upgrade replaces the whole bundle, so no library of an older version is left behind.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "dist\SemanticDBSearch\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Comment: "Search your spreadsheets offline"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Comment: "Search your spreadsheets offline"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  DataDir := ExpandConstant('{localappdata}\{#DataFolderName}');
  if UninstallSilent() or not DirExists(DataDir) then
    Exit;
  if MsgBox('Also delete your saved database, results and settings?' + #13#10#13#10 +
            'They are stored in ' + DataDir + '. If you keep them, a later installation finds them again.',
            mbConfirmation, MB_YESNO or MB_DEFBUTTON2) <> IDYES then
    Exit;
  if not DelTree(DataDir, True, True, True) then
    MsgBox('Some files could not be deleted. You can remove this folder yourself:' + #13#10 + DataDir,
           mbInformation, MB_OK);
end;
