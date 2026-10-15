param(
    [Parameter(Mandatory=$true)][string]$ServerDirectory,
    [Parameter(Mandatory=$true)][string]$Output,
    [Parameter(Mandatory=$true)][string]$NativeSha256,
    [string]$ModelPath='C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf',
    [string]$MmprojPath='C:/Works/models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf',
    [ValidateSet('qwen4exp','glm5-next','glm5next')][string]$Architecture='qwen4exp',
    [string]$SerialThreeChatReport,
    [ValidateRange(0,32768)][int]$ServerMaxTokens=0,
    [int]$Port=52176,
    [ValidateRange(1,2048)][int]$MaxTokens=160,
    [ValidateRange(0,2048)][int]$ExtendedMaxTokens=0,
    [int]$ReadyTimeoutSeconds=600,
    [switch]$PrefixChecks,
    [switch]$PrefixDiagnostics,
    [switch]$HttpChecks,
    [switch]$ThreeChatRegression,
    [switch]$FusedTrace,
    [switch]$RequireJointStartupWarmup,
    [switch]$DiagnosticTiming,
    [switch]$TokenTrace,
    [switch]$SkipSerialControls,
    [switch]$SerialOnlyAllowed
)
# Own exactly one explicitly identified server; generated evidence is never a
# substitute for the checker's semantic and overlapping-generation gates.
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$server=[IO.Path]::GetFullPath($ServerDirectory)
$outputPath=[IO.Path]::GetFullPath($Output)
if (!(($outputPath.StartsWith((Join-Path $repo 'artifacts')+[IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) -or
      ($outputPath.StartsWith((Join-Path $repo 'docs/validation')+[IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)))) { throw 'Output must be a fresh ignored evidence directory' }
if (Test-Path -LiteralPath $outputPath) { throw 'Preserve existing evidence; choose a new output directory' }
$null=New-Item -ItemType Directory -Path $outputPath
$exe=Join-Path $server 'TensorSharp.Server.Host.exe'
$checker=Join-Path $repo 'eng/validation/unified-qwen-concurrency-bench.py'
$prefixChecker=Join-Path $repo 'eng/validation/qwen-chat-cache-benchmark.py'
$httpChecker=Join-Path $repo 'eng/validation/unified-server-http-checks.py'
$threeChatChecker=Join-Path $repo 'eng/validation/unified-three-chat-regression.py'
$serialThreeChecker=Join-Path $repo 'eng/validation/unified-three-chat-parity.py'
$fusedVerifier=Join-Path $repo 'eng/validation/verify-fused-batch-evidence.py'
$sampler=Join-Path $repo 'eng/validation/sample-server-process.py'
$image=Join-Path $repo 'artifacts/multimodal-local-preflight/vision-fixtures/card-0.png'
$python=(Get-Command python -ErrorAction Stop).Source
$modelRoot=Split-Path -Parent ([IO.Path]::GetFullPath($ModelPath))
$launchArgs=@('--model', $ModelPath, '--mmproj', $MmprojPath, '--backend', 'ggml_cuda', '--unified-memory', '--host', '127.0.0.1', '--port', "$Port")
if($ThreeChatRegression -and $SerialThreeChatReport) { throw 'Choose concurrent regression or its serial controls for one launch' }
if($ThreeChatRegression) {
    if(!$FusedTrace) { throw 'Three-chat regression requires native fused trace' }
}
if($ThreeChatRegression -or $SerialThreeChatReport) {
    if($ServerMaxTokens -ne 0 -and $ServerMaxTokens -ne 20000) { throw 'Three-chat acceptance requires the reported 20000-token server limit' }
    $ServerMaxTokens=20000
}
if($ServerMaxTokens -gt 0) { $launchArgs+=@('--max-tokens',"$ServerMaxTokens") }
$audit=[ordered]@{
    startedUtc=[DateTime]::UtcNow.ToString('o'); command=@($exe)+$launchArgs; passed=$false; errors=@(); cleanupComplete=$false
    prefixDiagnosticLogging=[ordered]@{
        enabled=[bool]$PrefixDiagnostics
        excludesDefaultLoggingPerformanceComparison=[bool]$PrefixDiagnostics
        scope=$(if($PrefixDiagnostics) {
            'Diagnostic logging configuration: child TENSORSHARP_LOG_LEVEL=Debug. Do not compare its performance with runs without this diagnostic logging.'
        } else {
            'Prefix diagnostics disabled. This flag alone does not qualify performance or exclude overhead from other trace options.'
        })
    }
}
$owned=$null; $ownedStart=$null; $checkerExit=2; $sampling=$null
function Save($value,$name) { ConvertTo-Json -InputObject $value -Depth 32 | Set-Content -LiteralPath (Join-Path $outputPath $name) -Encoding utf8 }
function Hash($path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Snapshot {
    $snapshot=[ordered]@{}
    foreach($file in (Get-ChildItem -LiteralPath $server -File | Where-Object Extension -in '.exe','.dll','.json' | Sort-Object Name)) { $snapshot[$file.Name]=Hash $file.FullName }
    return $snapshot
}
function Owned {
    $p=Get-Process -Id $owned.Id -ErrorAction Stop
    if ([IO.Path]::GetFullPath($p.Path) -ne $exe -or $p.StartTime.ToUniversalTime().Ticks -ne $ownedStart) { throw 'Owned process identity changed' }
    return $p
}
function Listener {
    $rows=@(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    if ($rows.Count -eq 0 -or @($rows | Where-Object OwningProcess -ne $owned.Id).Count -ne 0) { throw 'Listener is not exclusively owned by this run' }
    $null=Owned
    return @($rows | Select-Object LocalAddress,LocalPort,OwningProcess)
}
try {
    if($RequireJointStartupWarmup -and ($Architecture -ne 'qwen4exp' -or !$FusedTrace)) {
        throw 'Joint startup acceptance requires qualified Qwen and native fused trace'
    }
    $audit.requireJointStartupWarmup=[bool]$RequireJointStartupWarmup
    if ((Hash (Join-Path $server 'GgmlOps.dll')) -ne $NativeSha256) { throw 'Native artifact differs from pinned candidate' }
    $audit.binariesBefore=Snapshot
    $audit.checkerSha=Hash $checker; $audit.wrapperSha=Hash $PSCommandPath
    $audit.samplerSha=Hash $sampler
    if($ThreeChatRegression) { $audit.threeChatCheckerSha=Hash $threeChatChecker }
    if($PrefixChecks) { $audit.prefixCheckerSha=Hash $prefixChecker }
    if($HttpChecks) { $audit.httpCheckerSha=Hash $httpChecker }
    if($SerialThreeChatReport) {
        $audit.serialThreeCheckerSha=Hash $serialThreeChecker
        $audit.concurrentReportSha=Hash $SerialThreeChatReport
    }
    $audit.fusedVerifierSha=Hash $fusedVerifier
    if($PrefixChecks -or $HttpChecks) { $audit.imageSha=Hash $image }
    $audit.modelFiles=@(Get-ChildItem -LiteralPath $modelRoot -Filter '*.gguf' -File | Select-Object FullName,Length,LastWriteTimeUtc)
    $audit.modelIdentityScope='File paths, sizes and modification times; not a fresh full-content hash of model shards.'
    $overrides=@(Get-ChildItem Env: | Where-Object { $_.Name -match '^(TS_|TENSORSHARP_|GGML_|MAX_CONTEXT|MAX_TOKENS|KV_CACHE_|CUDA_VISIBLE|OMP_|BACKEND$)' -and $_.Value -ne '' } | Select-Object Name,Value)
    Save $overrides 'environment-overrides.json'
    if ($overrides.Count -ne 0) { throw 'Inherited configuration override would invalidate this default-command acceptance' }
    if (@(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue).Count -ne 0) { throw 'Port already occupied' }
    $clock=[Diagnostics.Stopwatch]::StartNew()
    $processOptions=@{FilePath=$exe;ArgumentList=$launchArgs;WorkingDirectory=$repo;WindowStyle='Hidden';PassThru=$true;RedirectStandardOutput=(Join-Path $outputPath 'stdout.log');RedirectStandardError=(Join-Path $outputPath 'stderr.log')}
    $audit.childEnvironment=@{}
    if($PrefixDiagnostics) { $audit.childEnvironment.TENSORSHARP_LOG_LEVEL='Debug' }
    if($FusedTrace) { $audit.childEnvironment.TS_BATCHED_FUSED_TRACE='1' }
    if($TokenTrace) { $audit.childEnvironment.TS_VALIDATION_TOKEN_TRACE='1' }
    if($DiagnosticTiming) {
        $audit.childEnvironment.TS_HOST_MOE_TIMING='3'
        $audit.childEnvironment.TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS='1'
    }
    if($audit.childEnvironment.Count -gt 0) {
        if(!(Get-Command Start-Process).Parameters.ContainsKey('Environment')) {
            throw 'Child-only trace capture requires Start-Process -Environment; run this validation script with pwsh (PowerShell 7.4 or newer).'
        }
        $processOptions.Environment=$audit.childEnvironment
    }
    $owned=Start-Process @processOptions
    $ownedStart=$owned.StartTime.ToUniversalTime().Ticks
    $audit.owned=@{pid=$owned.Id;startTicks=$ownedStart;executable=$exe}
    Save $audit.owned 'owned-process.json'
    $sampling=Start-Process -FilePath $python -ArgumentList @('-B',$sampler,'--pid',"$($owned.Id)",'--output',(Join-Path $outputPath 'memory-samples.jsonl')) -WindowStyle Hidden -PassThru -RedirectStandardError (Join-Path $outputPath 'memory-sampler-errors.log')
    $ready=$false
    while($clock.Elapsed.TotalSeconds -lt $ReadyTimeoutSeconds) {
        $owned.Refresh()
        if($owned.HasExited) { throw "Server exited during startup: $($owned.ExitCode)" }
        try { $state=Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/models" -TimeoutSec 2; if($state.unifiedMemory -eq $true -and $state.visionReady -eq $true -and $state.architecture -eq $Architecture -and $state.loaded -eq [IO.Path]::GetFileName($ModelPath)) { $ready=$true; break } } catch { }
        Start-Sleep -Milliseconds 500
    }
    if(!$ready) { throw 'Server did not become ready' }
    $audit.readySeconds=$clock.Elapsed.TotalSeconds
    $audit.listenerBefore=Listener
    Save $state 'initial-model.json'
    Save ((Owned) | Select-Object Id,Path,WorkingSet64,PeakWorkingSet64,PrivateMemorySize64) 'ready-process.json'
    $native=@((Owned).Modules | Where-Object ModuleName -eq 'GgmlOps.dll' | Select-Object -ExpandProperty FileName)
    if($native.Count -ne 1 -or [IO.Path]::GetFullPath($native[0]) -ne (Join-Path $server 'GgmlOps.dll') -or (Hash $native[0]) -ne $NativeSha256) { throw 'Unexpected loaded native library' }
    $audit.loadedNative=$native[0]
    if(!$SerialOnlyAllowed -and $state.startupWarmup.completed -ne $true) { throw 'Scheduled startup warmup did not complete both thinking modes' }
    if($RequireJointStartupWarmup) {
        $joint=@($state.startupWarmup.jointStages)
        if($state.startupWarmup.jointWarmupRequested -ne $true -or $state.startupWarmup.jointCompleted -ne $true -or
           $state.startupWarmup.jointCleanupPending -ne $false -or $joint.Count -ne 2) {
            throw 'Startup did not accept and clean both native N2/N3 cohorts; inspect initial-model.json'
        }
        foreach($width in 2,3) {
            $stage=@($joint | Where-Object width -eq $width)
            if($stage.Count -ne 1 -or $stage[0].completed -ne $true -or
               $stage[0].nativeBatchAccepted -ne $true -or $stage[0].nativeAcceptedExecutionId -le 0 -or
               $stage[0].nativeNodeCount -le 0) { throw "Startup native width $width was not qualified" }
        }
    }
    $checkArgs=@('-X','utf8','-B',$checker,'--url',"http://127.0.0.1:$Port",'--output',(Join-Path $outputPath 'concurrency'),'--max-tokens',"$MaxTokens",'--timeout','600')
    if(!$SkipSerialControls) { $checkArgs+='--compare-serial' }
    $audit.serialControlsRequested=!$SkipSerialControls
    $fileLog=Join-Path $server "logs/tensorsharp-server-$([DateTime]::UtcNow.ToString('yyyyMMdd')).jsonl"
    if($ThreeChatRegression) {
        & $python -X utf8 -B $threeChatChecker --url "http://127.0.0.1:$Port" --output (Join-Path $outputPath 'three-chat') --server-log $fileLog --max-tokens 20000 --timeout 900 --join-timeout 90
        $audit.threeChatCheckerExit=$LASTEXITCODE
        if($LASTEXITCODE -ne 0) { throw "Three-chat 20K admission/cancellation checker failed: $LASTEXITCODE" }
        $audit.listenerAfterThreeChat=Listener
    }
    if(!$SerialOnlyAllowed) { $checkArgs+='--expect-interleaved' }
    $checkArgs+=@('--server-log',$fileLog)
    if($FusedTrace) { $checkArgs+='--require-fused-batch' }
    if($SerialThreeChatReport) {
        & $python -X utf8 -B $serialThreeChecker --url "http://127.0.0.1:$Port" --concurrent-report $SerialThreeChatReport --output (Join-Path $outputPath 'serial-three') --max-tokens $MaxTokens --server-log $fileLog
    } else { & $python @checkArgs }
    $checkerExit=$LASTEXITCODE
    $audit.checkerExit=$checkerExit
    $audit.listenerAfter=Listener
    if($checkerExit -ne 0) { throw "Concurrency checker failed: $checkerExit" }
    if($ExtendedMaxTokens -gt 0) {
        # Keep the controlled short run and the longer, warmed run separate.
        # Both use the same owned process and strict graph/serial comparison.
        $extendedArgs=@($checkArgs)
        $extendedArgs[[Array]::IndexOf($extendedArgs,'--output')+1]=Join-Path $outputPath 'extended/concurrency'
        $extendedArgs[[Array]::IndexOf($extendedArgs,'--max-tokens')+1]="$ExtendedMaxTokens"
        $audit.extendedMaxTokens=$ExtendedMaxTokens
        $audit.extendedScope='Same process after the primary bounded test; whole-run resource samples include both tests, startup and any subsequent checks.'
        & $python @extendedArgs
        $audit.extendedCheckerExit=$LASTEXITCODE
        if($LASTEXITCODE -ne 0) { throw "Extended concurrency checker failed: $LASTEXITCODE" }
        $audit.listenerAfterExtended=Listener
    }
    if($PrefixChecks) {
        & $python -X utf8 -B $prefixChecker --url "http://127.0.0.1:$Port" --model-discovery webui --protocol webui --cases text,image --image $image --turns 2 --repeats 1 --max-tokens 64 --timeout 600 --branches --invalidation --require-reuse --output (Join-Path $outputPath 'prefix.json')
        $audit.prefixCheckerExit=$LASTEXITCODE
        if($LASTEXITCODE -ne 0) { throw "Prefix checker failed: $LASTEXITCODE" }
    }
    if($HttpChecks) {
        & $python -X utf8 -B $httpChecker --url "http://127.0.0.1:$Port" --model ([IO.Path]::GetFileName($ModelPath)) --architecture $Architecture --image $image --server-log $fileLog --require-unified-status --default-output-limit-check --output (Join-Path $outputPath 'http')
        $audit.httpCheckerExit=$LASTEXITCODE
        if($LASTEXITCODE -ne 0) { throw "HTTP quality/cancellation/reload checker failed: $LASTEXITCODE" }
    }
} catch { $audit.errors+=$_.ToString() } finally {
    if($null -ne $owned) {
        try {
            $owned.Refresh()
            if(!$owned.HasExited) { Save ((Owned) | Select-Object Id,Path,WorkingSet64,PeakWorkingSet64,PrivateMemorySize64) 'final-process.json'; $null=Owned; $owned.Kill() }
            if(!$owned.WaitForExit(30000)) { throw 'Owned process did not exit within 30 seconds' }
            $audit.serverExit=$owned.ExitCode; $audit.cleanupComplete=$true
            $owned.Dispose()
        } catch { $audit.errors+="Cleanup: $_" }
    } else { $audit.cleanupComplete=$true }
    if($sampling) {
        try {
            if(!$sampling.WaitForExit(5000)) { $sampling.Kill(); $null=$sampling.WaitForExit(5000) }
            $audit.samplerExit=$sampling.ExitCode; $sampling.Dispose()
        } catch { $audit.errors+="Sampler cleanup: $_" }
    }
    try {
        $audit.binariesAfter=Snapshot
        if((ConvertTo-Json -InputObject $audit.binariesBefore -Compress) -cne (ConvertTo-Json -InputObject $audit.binariesAfter -Compress)) { throw 'Application binaries changed during the run' }
        if((Hash $checker) -ne $audit.checkerSha -or (Hash $PSCommandPath) -ne $audit.wrapperSha) { throw 'Validation code changed during the run' }
        if((Hash $sampler) -ne $audit.samplerSha) { throw 'Memory sampler changed' }
        if($PrefixChecks -and (Hash $prefixChecker) -ne $audit.prefixCheckerSha) { throw 'Prefix checker changed' }
        if($HttpChecks -and (Hash $httpChecker) -ne $audit.httpCheckerSha) { throw 'HTTP checker changed' }
        if($ThreeChatRegression -and (Hash $threeChatChecker) -ne $audit.threeChatCheckerSha) { throw 'Three-chat checker changed' }
        if($SerialThreeChatReport -and ((Hash $serialThreeChecker) -ne $audit.serialThreeCheckerSha -or (Hash $SerialThreeChatReport) -ne $audit.concurrentReportSha)) { throw 'Serial three-chat checker or concurrent evidence changed' }
        if((Hash $fusedVerifier) -ne $audit.fusedVerifierSha) { throw 'Fused evidence verifier changed' }
        if(($PrefixChecks -or $HttpChecks) -and (Hash $image) -ne $audit.imageSha) { throw 'Input image changed' }
    } catch { $audit.errors+="Identity: $_" }
    $audit.finishedUtc=[DateTime]::UtcNow.ToString('o')
    $audit.passed=$checkerExit -eq 0 -and $audit.cleanupComplete -and $audit.errors.Count -eq 0
    Save $audit 'launch-audit.json'
    if(Test-Path -LiteralPath (Join-Path $outputPath 'extended/concurrency/report.json')) {
        # Copies retain the explicit whole-run scope; neither is a peak measured
        # only during the extended run. Source reports are never overwritten.
        Copy-Item -LiteralPath (Join-Path $outputPath 'launch-audit.json') -Destination (Join-Path $outputPath 'extended/launch-audit.json')
        if(Test-Path -LiteralPath (Join-Path $outputPath 'memory-samples.jsonl')) {
            Copy-Item -LiteralPath (Join-Path $outputPath 'memory-samples.jsonl') -Destination (Join-Path $outputPath 'extended/memory-samples.jsonl')
        }
    }
}
if(!$audit.passed) { Write-Error ($audit.errors -join [Environment]::NewLine); exit 1 }
