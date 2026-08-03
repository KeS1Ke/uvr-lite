; uvr-lite 半在线安装程序（Inno Setup 7）
; 打包：scripts/build_installer.py 准备 _bundle（代码快照+内置 Python+CPU torch+模型）后调 ISCC 编译
; 设计（半在线：CPU 引擎离线内置即装即用，CUDA 引擎安装时按需联网下载）：
;   - 安装 = 纯文件复制（app 代码 / python 绿色 Python+依赖 / torch_cpu /
;     models 模型权重），Inno 原生进度与取消（Inno 6+ 支持 >2GB 安装包；
;     NSIS 有 ~2GB 硬限制无法打包）
;   - 附加任务「下载 CUDA 引擎」（默认不勾）：勾选时安装中联网下载 torch
;     cu128 wheel（约 3.3 GB，DownloadTemporaryFile 自带进度页 + SHA256 校验），
;     [Files] extractarchive 原生解压到 {app}\torch_cuda，装完裁剪
;     .lib/include/bin（省约 900 MB）；镜像源详见 CUDA_URL 注释。
;     取消勾选/下载失败不影响安装（可在应用内「推理引擎」区或 CLI
;     `uvr-lite install-cuda` 随时补装——复用多段并发+断点续传+镜像回退下载器）
;   - 推理引擎（CPU/CUDA）由应用内选择（torch.ini），启动时切换
;   - 快捷方式：桌面 + 开始菜单（♪），指向 pythonw -m uvr_lite.ui
;   - 卸载：控制面板/卸载器，删除快捷方式 + 注册表（含 QSettings 残留）+ 安装目录

#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif
#ifndef BundleDir
  #define BundleDir "..\dist\_bundle"
#endif

[Setup]
AppId={{5A3B9C41-7E2D-4F8A-B6C1-9D0E4F2A8C37}
AppName=uvr-lite
AppVersion={#MyAppVersion}
AppVerName=uvr-lite {#MyAppVersion}
AppPublisher=KeS1Ke
DefaultDirName={userpf}\uvr-lite
DefaultGroupName=uvr-lite
; Inno 7 的 DisableDirPage 默认值从 no 改为 auto：检测到同 AppId 已安装
; （升级/重装）会隐藏目录选择页。显式 no 保证每次安装都可选位置。
DisableDirPage=no
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist
OutputBaseFilename=uvr-lite-setup_v{#MyAppVersion}
SetupIconFile=..\uvr_lite\ui\resources\uvr-lite.ico
UninstallDisplayIcon={app}\app\uvr_lite\ui\resources\uvr-lite.ico
UninstallDisplayName=uvr-lite（人声/伴奏分离）
; lzma2/max（8MB 字典）替代 ultra64（64MB）：解压快 2-3 倍，包体积仅 +3%。
; ultra64 时 7.4G 解压耗时过长是安装慢的主因。
Compression=lzma2/max
SolidCompression=yes
; [Files] extractarchive 解压 CUDA wheel（zip）需要 full 引擎（basic 只支持 .7z）
ArchiveExtraction=full
WizardStyle=modern
VersionInfoVersion={#MyAppVersion}.0
VersionInfoProductName=uvr-lite
VersionInfoProductVersion={#MyAppVersion}
VersionInfoCompany=KeS1Ke
VersionInfoDescription=uvr-lite 安装程序

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

; 附加任务：CUDA 引擎（默认不勾——半在线模式，无显卡/不想等下载的用户直接跳过）
[Tasks]
Name: "cuda"; Description: "下载 CUDA 推理引擎（约 3.3 GB，需联网；NVIDIA 显卡推荐，GPU 加速分离）"; Flags: unchecked

[Files]
; 代码快照（uvr_lite + msst + pyproject.toml + README）→ {app}\app
Source: "{#BundleDir}\app\*"; DestDir: "{app}\app"; Flags: recursesubdirs createallsubdirs ignoreversion
; 内置绿色 Python + 全部依赖 → {app}\python
Source: "{#BundleDir}\python\*"; DestDir: "{app}\python"; Flags: recursesubdirs createallsubdirs ignoreversion
; CPU torch（应用内选择，启动时切换）→ torch_cpu/
Source: "{#BundleDir}\torch_cpu\*"; DestDir: "{app}\torch_cpu"; Flags: recursesubdirs createallsubdirs ignoreversion
; CUDA torch（勾选附加任务时由 [Code] 下载到 {tmp}，此处原生解压到 torch_cuda/）
Source: "{tmp}\torch-2.7.1+cu128-cp312-cp312-win_amd64.zip"; DestDir: "{app}\torch_cuda"; ExternalSize: 3273024349; Flags: external extractarchive recursesubdirs createallsubdirs ignoreversion; Check: CudaTaskSelected
; 模型权重 → {app}\models
Source: "{#BundleDir}\models\*"; DestDir: "{app}\models"; Flags: recursesubdirs createallsubdirs ignoreversion

; 快捷方式：桌面 + 开始菜单（♪），WorkingDir=app 使 -m uvr_lite.ui 命中快照代码
[Icons]
Name: "{autodesktop}\uvr-lite"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m uvr_lite.ui --model-dir ""{app}\models"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\uvr_lite\ui\resources\uvr-lite.ico"
Name: "{autoprograms}\uvr-lite\uvr-lite"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m uvr_lite.ui --model-dir ""{app}\models"""; WorkingDir: "{app}\app"; IconFilename: "{app}\app\uvr_lite\ui\resources\uvr-lite.ico"
Name: "{autoprograms}\uvr-lite\卸载 uvr-lite"; Filename: "{uninstallexe}"

; 完成页"运行 uvr-lite"复选框（默认勾选，安装后立即启动界面）
[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-m uvr_lite.ui --model-dir ""{app}\models"""; WorkingDir: "{app}\app"; Flags: nowait postinstall skipifsilent; Description: "{cm:LaunchProgram,uvr-lite}"

[Code]
// ---------- CUDA 引擎下载（半在线） ----------
// wheel 自包含全部 CUDA 运行库（torch/lib 内 cudnn/cublas/cufft DLL，无独立
// nvidia-* 包），SHA256 与 uvr_lite/download.py 的 TORCH_CUDA_SHA256 一致。
// 镜像按实测速度排序（2026-08-04）：SJTU 15-17MB/s > 官方 13-14MB/s >
// 阿里云 3-4MB/s（需浏览器 UA）。单 URL 无镜像回退——下载失败仅影响 CUDA
// 引擎，可在应用内补装（多源回退 + 断点续传）。
const
  // 下载后改名 .zip：wheel 本质是 zip，但 extractarchive 按扩展名识别格式，
  // .whl 会导致 "ArchiveFindFirstFile: Unknown ArchiveFileName extension"
  // （已实测）；内容不变，仅扩展名不同。
  CUDA_WHEEL = 'torch-2.7.1+cu128-cp312-cp312-win_amd64.zip';
  CUDA_SHA = '2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30';
  CUDA_URL = 'https://mirrors.sjtug.sjtu.edu.cn/pytorch-wheels/cu128/torch-2.7.1%2Bcu128-cp312-cp312-win_amd64.whl';

var
  ProgressPage: TOutputProgressWizardPage;

function CudaTaskSelected: Boolean;
begin
  // 勾选且尚未装过 CUDA 引擎（升级/重装场景已存在则跳过，避免重下 3.3GB）
  Result := WizardIsTaskSelected('cuda')
    and not DirExists(ExpandConstant('{app}\torch_cuda\torch'));
end;

// 下载进度回调 → 进度页进度条。注意：Inno 7 的 CreateDownloadPage
// （TDownloadWizardPage）有运行时兼容问题（脚本异常 Type Mismatch，
// 已最小复现），改用老牌的 CreateOutputProgressPage；其进度条为 32 位
// Integer，3.3GB 直接传会溢出，按 MB 缩放（最大 3273 MB，安全）。
function OnCudaDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
var
  P, M: Integer;
begin
  if ProgressMax > 0 then
  begin
    P := Integer(Progress div 1048576);
    M := Integer(ProgressMax div 1048576);
    ProgressPage.SetProgress(P, M);
  end;
  Result := True;
end;

// 卸载临时目录兜底：卸载程序从 [tmp]\is-*-uninstall.tmp\_unins.tmp 副本运行，
// 卸载结束由 Inno 父进程删除该目录；偶发失败（句柄/杀软锁定 _unins.tmp）时
// 目录残留在 %TEMP%（曾实测发现）。注册系统重启时递归删除——对 Inno 已自删
// 的正常路径无副作用，且只针对本应用的 [tmp]，不碰其他软件的 is-* 目录。
const
  MOVEFILE_DELAY_UNTIL_REBOOT = $4;

function MoveFileEx(lpExistingFileName: string; lpNewFileName: string; dwFlags: DWORD): BOOL;
  external 'MoveFileExW@kernel32.dll stdcall';

// ---------- 裁剪 CUDA torch（.lib / include / bin，运行时不需要） ----------
// 与打包脚本 build_installer._prune_torch 同款逻辑（Pascal 版）；实测
// torch_cuda 5.6G→4.7G（-900M）。bin/ 保留 torch_shm_manager.exe。

procedure DelTree(const Dir: String);
var
  FindRec: TFindRec;
begin
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then
  begin
    try
      repeat
        if FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0 then
        begin
          if FindRec.Name <> '.' then
            if FindRec.Name <> '..' then
              DelTree(AddBackslash(Dir) + FindRec.Name);
        end
        else
          DeleteFile(AddBackslash(Dir) + FindRec.Name);
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
    RemoveDir(Dir);
  end;
end;

procedure PruneTorchCuda(const TorchCudaDir: String);
var
  FindRec: TFindRec;
  T: String;
begin
  T := AddBackslash(AddBackslash(TorchCudaDir) + 'torch');
  // .lib 编译期导入库
  if FindFirst(AddBackslash(T + 'lib') + '*.lib', FindRec) then
  begin
    try
      repeat
        DeleteFile(AddBackslash(T + 'lib') + FindRec.Name);
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
  // include/ 头文件
  DelTree(T + 'include');
  // bin/（保留 torch_shm_manager.exe）。注意 Inno 的 FindFirst 会返回 .
  // 和 .. 条目——必须过滤，否则遇到 .. 会递归 DelTree 到 torch/ 父目录
  // 把整个引擎删光（曾实测：bin\.. → torch 被整目录递归删除）
  if FindFirst(AddBackslash(T + 'bin') + '*', FindRec) then
  begin
    try
      repeat
        if FindRec.Name <> 'torch_shm_manager.exe' then
          if FindRec.Name <> '.' then
            if FindRec.Name <> '..' then
              if FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0 then
                DelTree(AddBackslash(T + 'bin') + FindRec.Name)
              else
                DeleteFile(AddBackslash(T + 'bin') + FindRec.Name);
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

procedure InitializeWizard;
begin
  ProgressPage := CreateOutputProgressPage('正在下载 CUDA 推理引擎', '下载完成后自动继续安装。');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
  begin
    // 勾选附加任务且未装过才下载；文件复制阶段开始前 wheel 必须已在 {tmp}
    if CudaTaskSelected then
    begin
      ProgressPage.Show;
      try
        ProgressPage.SetProgress(0, 0);
        if DownloadTemporaryFile(CUDA_URL, CUDA_WHEEL, CUDA_SHA, @OnCudaDownloadProgress) < 0 then
          RaiseException('CUDA 引擎下载失败，可在应用内重试');
      finally
        ProgressPage.Hide;
      end;
    end;
  end;
  if CurStep = ssPostInstall then
  begin
    // [Files] extractarchive 解压完成后裁剪。注意不能用 CudaTaskSelected：
    // 其 DirExists 检查在此时已因刚解压的目录而误判（返回 False 跳过裁剪）。
    // 裁剪幂等——升级场景勾选但已有旧目录时同样安全。
    if WizardIsTaskSelected('cuda') then
      PruneTorchCuda(ExpandConstant('{app}\torch_cuda'));
  end;
end;

// 卸载：清理 QSettings 残留（UI 参数记忆存于注册表）+ 卸载临时目录兜底
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    RegDeleteKeyIncludingSubkeys(HKCU, 'Software\uvr-lite');
    MoveFileEx(ExpandConstant('{tmp}'), '', MOVEFILE_DELAY_UNTIL_REBOOT);
  end;
end;
