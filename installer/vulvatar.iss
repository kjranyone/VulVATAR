; Inno Setup script for VulVATAR.
;
; Build prerequisites (run from the repo root):
;   1. Build the main app:
;        cargo build --release
;   2. Build the MediaFoundation virtual camera DLL with a static CRT
;      so it has no VC runtime dependency when svchost loads it:
;        $env:RUSTFLAGS = "-C target-feature=+crt-static"
;        cargo build --release -p vulvatar-mf-camera
;        Remove-Item Env:RUSTFLAGS
;   3. Make sure assets\ and models\ are populated:
;        .\dev.ps1 -Select 1     # download fonts + ONNX models
;   4. Compile this script with Inno Setup 6 (iscc.exe):
;        iscc installer\vulvatar.iss
;
; Output: installer\output\VulVATAR-Setup-<version>.exe
;
; The installer registers the MediaFoundation virtual camera under HKLM
; (admin required) — see docs\mf-virtual-camera.md for the contract.

#define MyAppName            "VulVATAR"
#define MyAppVersion         "0.1.0"
#define MyAppPublisher       "kjranyone"
#define MyAppExeName         "vulvatar.exe"
#define MfCameraDllName      "vulvatar_mf_camera.dll"
#define MfCameraClsid        "{B5F1C320-2B8F-4A9C-9BDC-43B0E8E6B2E1}"
#define MfCameraFriendlyName "VulVATAR Virtual Camera"

[Setup]
; AppId pins the install identity across upgrades. Use the same CLSID
; the camera DLL uses — guaranteed unique to this product.
AppId={#MfCameraClsid}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=output
OutputBaseFilename=VulVATAR-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
; Windows 11 only — the MediaFoundation virtual camera contract relies on
; Frame Server features that did not ship in earlier releases.
MinVersion=10.0.22000

[Languages]
Name: "english";  MessagesFile: "compiler:Default.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Dirs]
; Pre-create directories that need explicit ACL adjustment so the [Run]
; icacls steps have something to act on. {commonappdata}\VulVATAR holds
; the file-backed shared-memory frame buffer; FrameServer's svchost (NT
; AUTHORITY\LocalService) needs inherited read access here so files
; vulvatar.exe creates inside automatically pick up the grant.
Name: "{app}\VirtualCamera"
Name: "{commonappdata}\VulVATAR"

[InstallDelete]
; Sweep timestamped DLLs left over from previous dev.ps1 installs.
; The shipped installer uses a fixed file name (no timestamp) because
; we stop FrameServer before [Files] runs, so there's no held DLL to
; install alongside.
Type: files; Name: "{app}\VirtualCamera\vulvatar_mf_camera_*.dll"

[Files]
Source: "..\target\release\{#MyAppExeName}";    DestDir: "{app}";                Flags: ignoreversion
Source: "..\target\release\DirectML.dll";       DestDir: "{app}";                Flags: ignoreversion
Source: "..\target\release\{#MfCameraDllName}"; DestDir: "{app}\VirtualCamera";  Flags: ignoreversion restartreplace uninsrestartdelete
Source: "..\LICENSE";                           DestDir: "{app}";                Flags: ignoreversion
Source: "..\README.md";                         DestDir: "{app}";                Flags: ignoreversion
Source: "..\assets\NotoSansJP-Regular.otf";     DestDir: "{app}\assets";         Flags: ignoreversion
Source: "..\assets\NotoSansKR-Regular.otf";     DestDir: "{app}\assets";         Flags: ignoreversion
Source: "..\assets\NotoSansSC-Regular.otf";     DestDir: "{app}\assets";         Flags: ignoreversion
Source: "..\models\rtmw3d.onnx";                DestDir: "{app}\models";         Flags: ignoreversion
Source: "..\models\yolox.onnx";                 DestDir: "{app}\models";         Flags: ignoreversion
Source: "..\models\face_landmark.onnx";         DestDir: "{app}\models";         Flags: ignoreversion
Source: "..\models\face_blendshapes.onnx";      DestDir: "{app}\models";         Flags: ignoreversion

[Registry]
; HKLM CLSID registration — mirrors what dev.ps1 Register-MfCameraSystem
; writes. Frame Server resolves the virtual camera CLSID through HKLM,
; so this is mandatory; per-user HKCU is not enough.
Root: HKLM; Subkey: "Software\Classes\CLSID\{#MfCameraClsid}";                 ValueType: string; ValueData: "{#MfCameraFriendlyName}";                                Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\Classes\CLSID\{#MfCameraClsid}\InprocServer32";  ValueType: string; ValueData: "{app}\VirtualCamera\{#MfCameraDllName}"
Root: HKLM; Subkey: "Software\Classes\CLSID\{#MfCameraClsid}\InprocServer32";  ValueType: string; ValueName: "ThreadingModel"; ValueData: "Both"

[Icons]
; WorkingDir matters: vulvatar.exe loads models\ and assets\ relative
; to CWD, not exe-dir. Setting WorkingDir to the install dir on the
; shortcut keeps that working for normal Start Menu launches.
Name: "{group}\{#MyAppName}";          Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}";    Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Grant LocalService (S-1-5-19) read+execute on the installed DLL so
; FrameServer's svchost can LoadLibrary it. Default Program Files ACL
; only grants BUILTIN\Users which excludes LocalService, so without
; this step ReadSample returns ENDOFSTREAM immediately and clients
; (Meet, Teams, Camera app) see a black frame.
Filename: "{sys}\icacls.exe"; \
  Parameters: """{app}\VirtualCamera\{#MfCameraDllName}"" /grant ""*S-1-5-19:(RX)"""; \
  Flags: runhidden waituntilterminated; \
  StatusMsg: "Granting LocalService read access on the camera DLL..."

; Same idea on the frame-buffer directory under ProgramData. (OI)(CI)
; makes the grant inherit so files vulvatar.exe creates inside pick up
; LocalService:(RX) automatically. SID literal is used so the icacls
; call works regardless of UI language.
Filename: "{sys}\icacls.exe"; \
  Parameters: """{commonappdata}\VulVATAR"" /grant ""*S-1-5-19:(OI)(CI)(RX)"""; \
  Flags: runhidden waituntilterminated; \
  StatusMsg: "Granting LocalService read access on the frame-buffer directory..."

; Start FrameServer so the freshly-installed DLL is loaded on the next
; ActivateObject. PrepareToInstall stopped it before [Files] ran.
Filename: "{sys}\net.exe"; Parameters: "start FrameServer"; \
  Flags: runhidden waituntilterminated; \
  StatusMsg: "Starting Windows Camera Frame Server..."

Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; \
  Flags: nowait postinstall skipifsilent; WorkingDir: "{app}"

[UninstallDelete]
; ProgramData\VulVATAR is not user-authored data — only the ephemeral
; camera_frame_buffer.bin lives there — so it's safe to remove on
; uninstall.
Type: filesandordirs; Name: "{commonappdata}\VulVATAR"

[Code]
function StopFrameServer: Integer;
var
  ResultCode: Integer;
begin
  // `net stop ... /Y` waits for the service to actually stop AND
  // auto-confirms any "stop dependent services" prompt. `sc stop`
  // returns immediately without waiting, which would race the file
  // copy in [Files].
  Exec(ExpandConstant('{sys}\net.exe'), 'stop FrameServer /Y', '',
       SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := ResultCode;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  // Stop FrameServer before [Files] so the installed DLL isn't pinned
  // by svchost when Inno Setup overwrites it during upgrade. A
  // non-zero return is non-fatal: the service may already be stopped,
  // and `restartreplace` on the DLL [Files] entry covers any residual
  // file lock by deferring the swap to the next reboot.
  StopFrameServer;
  Result := '';
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ResultCode: Integer;
begin
  if CurUninstallStep = usUninstall then
  begin
    // Stop the service before [UninstallDelete] removes the DLL,
    // otherwise svchost keeps it `LoadLibrary`-held and the file
    // delete fails silently, leaving an orphan in Program Files.
    StopFrameServer;
  end
  else if CurUninstallStep = usPostUninstall then
  begin
    // Bring FrameServer back so the OS camera stack works again
    // (built-in webcams etc. depend on it).
    Exec(ExpandConstant('{sys}\net.exe'), 'start FrameServer', '',
         SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;
