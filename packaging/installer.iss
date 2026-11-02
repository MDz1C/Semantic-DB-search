; Inno Setup 6.7 script for the Fundscope installer.
;
; packaging\build.ps1 compiles it after PyInstaller has produced dist\Fundscope:
;   ISCC.exe /DAppVersion=1.0.0 packaging\installer.iss
; -> dist\Fundscope-Setup-<version>.exe
;
; Installs per user without administrator rights (an all-users install can be chosen in the
; privileges dialog). User data lives in %LOCALAPPDATA%\Fundscope, never in the program
; folder; the uninstaller offers to delete it.

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "Fundscope"
#define AppPublisher "Manuel Dzimah"
#define AppExeName "Fundscope.exe"
#define DataFolderName "Fundscope"

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
OutputBaseFilename=Fundscope-Setup-{#AppVersion}
DefaultDirName={autopf}\Fundscope
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
; Builds released before the Fundscope name used another executable and shortcut name.
Type: files; Name: "{app}\SemanticDBSearch.exe"
Type: files; Name: "{autoprograms}\Semantic DB Search.lnk"
Type: files; Name: "{autodesktop}\Semantic DB Search.lnk"

[Files]
Source: "dist\Fundscope\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Comment: "Find any fund in your spreadsheet database, offline"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Comment: "Find any fund in your spreadsheet database, offline"; Tasks: desktopicon

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
