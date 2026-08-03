; uvr-lite 全量安装程序（Inno Setup 7）
; 打包：scripts/build_installer.py 准备 _bundle（代码快照+内置 Python+CPU[/CUDA] torch+模型）后调 ISCC 编译
; 设计（全量安装：用户拿到安装包即装即用，无需联网下载任何组件）：
;   - 安装 = 纯文件复制（app 代码 / python 绿色 Python+依赖 / torch_cpu[+torch_cuda] /
;     models 模型权重），Inno 原生进度与取消（Inno 6+ 支持 >2GB 安装包；
;     NSIS 有 ~2GB 硬限制，full 变体无法打包）
;   - 变体（/DVARIANT=cpu|full）：cpu 不含 torch_cuda（省 3.3GB，无独立显卡用户）
;   - 推理引擎（CPU/CUDA）由应用内选择（torch.ini），启动时切换
;   - 快捷方式：桌面 + 开始菜单（♪），指向 pythonw -m uvr_lite.ui
;   - 卸载：控制面板/卸载器，删除快捷方式 + 注册表（含 QSettings 残留）+ 安装目录

#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif
#ifndef BundleDir
  #define BundleDir "..\dist\_bundle"
#endif
#ifndef Variant
  #define Variant "full"
#endif

[Setup]
AppId={{5A3B9C41-7E2D-4F8A-B6C1-9D0E4F2A8C37}
AppName=uvr-lite
AppVersion={#MyAppVersion}
AppVerName=uvr-lite {#MyAppVersion}
AppPublisher=KeS1Ke
DefaultDirName={userpf}\uvr-lite
DefaultGroupName=uvr-lite
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist
OutputBaseFilename=uvr-lite-setup-{#Variant}_v{#MyAppVersion}
SetupIconFile=..\uvr_lite\ui\resources\uvr-lite.ico
UninstallDisplayIcon={app}\app\uvr_lite\ui\resources\uvr-lite.ico
UninstallDisplayName=uvr-lite（人声/伴奏分离）
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
VersionInfoVersion={#MyAppVersion}.0
VersionInfoProductName=uvr-lite
VersionInfoProductVersion={#MyAppVersion}
VersionInfoCompany=KeS1Ke
VersionInfoDescription=uvr-lite 安装程序

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Files]
; 代码快照（uvr_lite + msst + pyproject.toml + README）→ {app}\app
Source: "{#BundleDir}\app\*"; DestDir: "{app}\app"; Flags: recursesubdirs createallsubdirs ignoreversion
; 内置绿色 Python + 全部依赖 → {app}\python
Source: "{#BundleDir}\python\*"; DestDir: "{app}\python"; Flags: recursesubdirs createallsubdirs ignoreversion
; CPU torch（应用内选择，启动时切换）→ torch_cpu/
Source: "{#BundleDir}\torch_cpu\*"; DestDir: "{app}\torch_cpu"; Flags: recursesubdirs createallsubdirs ignoreversion
; CUDA torch（full 变体）→ torch_cuda/
#if Variant == "full"
Source: "{#BundleDir}\torch_cuda\*"; DestDir: "{app}\torch_cuda"; Flags: recursesubdirs createallsubdirs ignoreversion
#endif
; 模型权重 → {app}\models
Source: "{#BundleDir}\models\*"; DestDir: "{app}\models"; Flags: recursesubdirs createallsubdirs ignoreversion

; 快捷方式：桌面 + 开始菜单（♪），WorkingDir=app 使 -m uvr_lite.ui 命中快照代码
[Icons]
Name: "{autodesktop}\uvr-lite"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m uvr_lite.ui --model-dir ""{app}\models"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\uvr_lite\ui\resources\uvr-lite.ico"
Name: "{autoprograms}\uvr-lite\uvr-lite"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m uvr_lite.ui --model-dir ""{app}\models"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\uvr_lite\ui\resources\uvr-lite.ico"
Name: "{autoprograms}\uvr-lite\卸载 uvr-lite"; Filename: "{uninstallexe}"

[Code]
{ 卸载：清理 QSettings 残留（UI 参数记忆存于注册表） }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RegDeleteKeyIncludingSubkeys(HKCU, 'Software\uvr-lite');
end;
