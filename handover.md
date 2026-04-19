# VulVATAR MediaFoundation 仮想カメラ — Codex 引き継ぎメモ

作成日: 2026-04-19 / 対象: `vulvatar-mf-camera` クレート

## ゴール

`MFCreateVirtualCamera(MFVirtualCameraType_SoftwareCameraSource, ...)` で登録した仮想カメラが、Google Meet / Chrome / Windows カメラアプリから**フレームを受け取れる**ところまで完成させる。

## 現状サマリー

- **Direct CoCreateInstance path（LoadLibraryW + DllGetClassObject + CreateInstance(IMFActivate) + ActivateObject + MFCreateSourceReaderFromMediaSource + ReadSample）は完全に動く。** `Stream::RequestSample direct frame seq=...` で正常なサンプルが 3 枚返る（`cargo run --bin mfenum` で再現可能）。
- **MFEnumDeviceSources 経由の FS-mediated path は `flags=0x100`（`MF_SOURCE_READERF_ENDOFSTREAM`）を即返す。** 1 枚も流れない。
- 仮想カメラは `MFEnumDeviceSources` で列挙される（symlink = `\\?\swd#vcamdevapi#{hash}#{KSCATEGORY_VIDEO_CAMERA}\{PINNAME_VIDEO_CAPTURE}`）。
- ホストアプリ側（`src/output/mf_virtual_camera.rs` → `MFCreateVirtualCamera` → `camera.Start()`）は `info!("mf-virtual-camera: started ...")` を出して登録成功を報告。HKLM CLSID 登録 + `NT AUTHORITY\LocalService:(RX)` grant 済み（`dev.ps1 → install mf virtual camera`）。

## 決定的な観測事実

### FrameServer event log (`Microsoft-Windows-MF-FrameServer/Camera_FrameServer`)

VulVATAR (vcamdevapi):
```
Event 6 (FsProxy 初期化開始) SymbolicLink: ...#{65e8773d-...}\{fcebba03-...}
Event 11 WatchdogTimer 開始 WatchdogOperation: アクティブ化 TriggerHns=200000000
Event 12 WatchdogTimer 停止 WatchdogOperation: アクティブ化 CompletionHns=31〜41   ← 3〜4 マイクロ秒で完了
Event 7 (FsProxy 初期化停止) hr=0x0
Event 2 (SetOutputType 開始) PinId:0 NV12 1920×1080@30
Event 3 (SetOutputType Stop) hr=0x0
```

RealSense 等の実機カメラ:
```
Event 6 (FsProxy 初期化開始)
Event 11 WatchdogTimer 開始 WatchdogOperation: 初期化
Event 7 (FsProxy 初期化停止) hr=0x0
Event 12 WatchdogTimer 停止 WatchdogOperation: 初期化 CompletionHns=18477〜32466   ← 2〜3 ms
```

- 実機は `WatchdogOperation: 初期化` で ms オーダーの実作業。VulVATAR だけが `WatchdogOperation: アクティブ化` を使い、4 マイクロ秒で「完了」する → 内部ショートサーキットの疑い濃厚。
- `SetOutputType` で VulVATAR の NV12/1920×1080/30 メタデータは FS 側でも見えている（= 何らかの QI は成立している）。

### トレース (`vulvatar-mf-camera/src/trace.rs`)

- Direct path: `Source::Start` → `MENewStream/MESourceStarted/MEStreamStarted` → `Stream::RequestSample direct frame seq=...` の系列が流れる。
- FS-mediated path: **svchost pid からのトレース出力が 0 行**。`SetDefaultAllocator` も `GetAllocatorUsage` も**一度も呼ばれていない**。
- トレース先を `%TEMP%\vulvatar_mf_camera.log` から `C:\Users\Public\vulvatar_mf_camera.log`（LocalService からも書ける場所）に切り替えたが、現時点で再インストール＋再起動後の検証は未完。

## 既に試した（が解消していない）対策

`IMFMediaSource` 側で実装／公開済みのインターフェース：

- `IMFMediaSource` / `IMFMediaSourceEx` / `IMFGetService`
- `IMFAttributes`（`GetSourceAttributes` と同じ store を QI で返す）
- `IMFExtendedCameraController` + `IMFExtendedCameraControl`
- `IKsControl`（PRIVACY property / event）
- `IMFSampleAllocatorControl`（`SetDefaultAllocator` 保存 / `GetAllocatorUsage=UsesProvidedAllocator(0)`）

Source 属性にセット済み：
- `MFT_TRANSFORM_CLSID_Attribute` = CLSID
- `MF_VIRTUALCAMERA_PROVIDE_ASSOCIATED_CAMERA_SOURCES` = 1
- `MF_DEVICESTREAM_FRAMESERVER_SHARED` = 1（※ 後述の [6] で source 側には付けない方がよい可能性あり）
- sensor profile collection（`KSCAMERAPROFILE_Legacy`）

Activator (`VulvatarActivate`) 側：
- `MFT_TRANSFORM_CLSID_Attribute`、`MF_VIRTUALCAMERA_PROVIDE_ASSOCIATED_CAMERA_SOURCES` を自身の IMFAttributes にも set
- `ActivateObject` 内で `source.GetSourceAttributes().CopyAllItems(activator_attrs)` 実行

その他：
- `Win32NamedSharedMemorySink` の ZST ポインタバグ（`*mut ()` で `.add(N)` が no-op 化）は修正済み。ホスト側はシェアドメモリに正しいフレームを書いている。
- stream は `IMFMediaStream2` を `#[implement]`、`IMFMediaStream_Impl` は手書き。

## リサーチエージェントによる診断（優先順）

### 1. [最重要] Activator と MediaSource を **1 つの COM オブジェクト / 1 つの IMFAttributes store** に統合

smourier/VCamSample の本質的な構造は:

- `IClassFactory::CreateInstance` が返すのは **Activator** のみ。
- `Activator::ActivateObject` は `MediaSource::Initialize(this)` を呼び、その中で **source の `CBaseAttributes`（= source 自身が IMFAttributes 実体）に activator の全属性を `CopyAllItems` する**。
- 結果として、**activator と source は同一の IMFAttributes identity を共有**。FrameServer は activation 後も activator 側の IMFAttributes を読み書きし続けるが、それが source に届く（同一オブジェクトだから）。

VulVATAR は `VulvatarActivate` が独自の `MFCreateAttributes` を保持し、activation 時に一度だけ CopyAllItems する **スナップショット方式**。FsProxy が activation 後に activator の属性ストアに書き込んだ値が source に届かず、QI ID の不一致で FsProxy が 3〜4 マイクロ秒で tear down → `ENDOFSTREAM` という仮説。

**修正方針**: Rust の単一 #[implement] で両立しないので、どちらかを選ぶ：

- **案 A（推奨）**: `VulvatarActivate` を薄いラッパにし、**`IMFAttributes_Impl` のメソッドは自前 store ではなく `source.attributes` に forward**。`ActivateObject` は `source.QueryInterface(riid, ppv)` だけ。これで activator と source が実質同一 attributes store を見る。
- **案 B**: Activator を廃止し、`VulvatarMediaSource` が直接 `IMFActivate` も `#[implement]`。`ActivateObject` は `self.QueryInterface(riid, ppv)`。smourier に最も近い構造。

### 2. [高] `IMFRealTimeClientEx` を実装

FrameServer の marshaling 層は `IMFRealTimeClientEx` を QI し、「FrameServerPool MTA worker で扱ってよい source」と判定する。windows-rs の `#[implement]` は列挙されていないインターフェースをサポートしないので、`IMFRealTimeClientEx` を実装リストに追加し、`RegisterThreads` / `UnregisterThreads` / `SetWorkQueueEx` は全て `S_OK` を返せば OK。

### 3. [中] Stream も IMFAttributes 実体にする

smourier の `MediaSource::GetStreamAttributes` は **stream 自身を `IMFAttributes` として返す**（stream が `CBaseAttributes`）。VulVATAR は `GetStreamAttributes` で `MFCreateAttributes` した別オブジェクトを返しているので、FS が QI → `IKsControl` に戻そうとして失敗する可能性。`VulvatarMediaStream` の `#[implement]` に `IMFAttributes` を足して `GetStreamAttributes` は `stream.cast::<IMFAttributes>()` を返す。

### 4. [中] `SoftwareCameraSource` のプロセスモデル

`MFVirtualCameraType_SoftwareCameraSource` では source は **client プロセス内**（mfenum.exe やブラウザ内）で activation される可能性が高く、svchost/FrameServer はそれを *marshal 経由で proxy* する。svchost のトレースが空なのと整合。4 マイクロ秒の Watchdog は FrameServer 側の marshal-proxy が QI 契約を満たさず即失敗しているタイミング。

### 5. [低] KSCATEGORY GUID は red herring

`{65e8773d-8f56-11d0-a3b9-00a0c9223196}` は `KSCATEGORY_VIDEO`（`KSCATEGORY_CAPTURE` ではない）。FS 内部の正規化なので無害。

### 6. [低] Source 属性のクリーンアップ

- `MF_DEVICESTREAM_FRAMESERVER_SHARED` は **stream-scoped**。source attributes に付けているのは無害かもしれないが、smourier は stream 属性にのみ付けているので合わせた方が安全。
- `IMFMediaSource2`（`SetMediaType` スタブ）を追加。

## 作業中だが未完の変更

- `vulvatar-mf-camera/src/trace.rs`: ログ先を `C:\Users\Public\vulvatar_mf_camera.log` に変更（LocalService svchost からも書ける）。ビルド済み。
- `vulvatar-mf-camera/src/lib.rs`: `BUILD_MARKER = "trace-public-dir"`。
- 未だ Program Files の DLL は**旧版（`source-is-attributes` マーカー）**。再インストール + vulvatar.exe 再起動が必要。
- 再インストールコマンド: `.\dev.ps1` を管理者 PowerShell で起動 → `install mf virtual camera (HKLM)` を選択。`icacls` で `LOCAL SERVICE:(RX)` 付与 + HKLM 更新まで自動。

## 具体的な Next Step（優先順）

1. **最優先: 案 A か案 B（上記 [1]）を実装して Activator と MediaSource の IMFAttributes identity を統一**。案 B（VulvatarMediaSource に直接 IMFActivate を実装）が smourier と一番近く、差分も少ない。
2. `IMFRealTimeClientEx` を追加（[2]）。
3. 新ビルドを `.\dev.ps1 → install` で Program Files に反映 → vulvatar.exe 再起動 → `cargo run --bin mfenum` で検証。
4. `C:\Users\Public\vulvatar_mf_camera.log` に svchost pid からの DllGetClassObject / ActivateObject / Source::Start が現れるか確認。
5. それでもダメなら stream の IMFAttributes 化（[3]）、`IMFMediaSource2`、`MF_DEVICESTREAM_FRAMESERVER_SHARED` の source 側削除を順に試す。

## 参照すべきファイル

- `vulvatar-mf-camera/src/activate.rs` — 現行 Activator 実装（独立 IMFAttributes）
- `vulvatar-mf-camera/src/media_source.rs` — Source 本体、IMFAttributes forwarder 28 メソッド含む
- `vulvatar-mf-camera/src/media_stream.rs` — IMFMediaStream2 実装、allocator path + direct path
- `vulvatar-mf-camera/src/class_factory.rs` — `DllGetClassObject` が返す class factory
- `vulvatar-mf-camera/src/trace.rs` — ログ出力先（今回 Public に変更）
- `vulvatar-mf-camera/src/lib.rs` — DLL エントリポイント、BUILD_MARKER
- `src/output/mf_virtual_camera.rs` — ホスト側 MFCreateVirtualCamera 呼び出し
- `src/bin/mfenum.rs` — 診断ツール（direct probe + FS-mediated enumerate）
- `dev.ps1` — HKLM install（要管理者）
- `docs/mf-virtual-camera-handover.md` — 以前のハンドオーバードキュメント（文脈用）

## 参考リンク

- smourier/VCamSample — https://github.com/smourier/VCamSample
  - `VCamSampleSource/MediaSource.h` — インターフェースリスト
  - `VCamSampleSource/Activator.cpp` — `_source->Initialize(this)` パターン
  - `VCamSampleSource/MediaSource.cpp` — `GetSourceAttributes` は `QueryInterface(IID_PPV_ARGS(...))` で自身を返す
- Frame Server Custom Media Source — https://github.com/MicrosoftDocs/windows-driver-docs/blob/staging/windows-driver-docs-pr/stream/frame-server-custom-media-source.md
- MFSampleAllocatorUsage — https://learn.microsoft.com/en-us/windows/win32/api/mfidl/ne-mfidl-mfsampleallocatorusage
- MFCreateVirtualCamera — https://learn.microsoft.com/en-us/windows/win32/api/mfvirtualcamera/nf-mfvirtualcamera-mfcreatevirtualcamera
- qing-wang/MFVCamSource — https://github.com/qing-wang/MFVCamSource
- Alax.info Virtual Camera blog — https://alax.info/blog/2206 / https://alax.info/blog/2245

## 再現手順メモ

```powershell
# 1. ビルド（すでに済み、新トレース先）
cargo build -p vulvatar-mf-camera

# 2. 再インストール（要管理者 PowerShell）
.\dev.ps1    # → "install mf virtual camera (HKLM)" を選択

# 3. vulvatar.exe 再起動（古い DLL を掴んでいるので）

# 4. 診断ツール
cargo run --bin mfenum
#   direct-dll ReadSample[0..2]: sample=true       ← 直接パスは動く
#   FS-mediated ReadSample: flags=0x100 sample=false ← ここが 0 に戻ればゴール

# 5. トレース確認
Get-Content C:\Users\Public\vulvatar_mf_camera.log -Tail 50
# svchost の pid から DllGetClassObject が出れば FS 経由で DLL がロードされた証拠

# 6. FrameServer イベント確認
Get-WinEvent -LogName 'Microsoft-Windows-MF-FrameServer/Camera_FrameServer' -MaxEvents 30 |
  Format-List TimeCreated, Id, Message
# WatchdogOperation: 初期化 に変われば正常。アクティブ化 で 40hns のままならまだ short-circuit。
```
