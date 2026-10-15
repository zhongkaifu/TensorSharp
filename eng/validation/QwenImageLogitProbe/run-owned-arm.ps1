param(
 [Parameter(Mandatory)][string]$Application,
 [Parameter(Mandatory)][string]$InputFile,
 [Parameter(Mandatory)][string]$Image,
 [Parameter(Mandatory)][string]$Output,
 [Parameter(Mandatory)][string]$Model,
 [Parameter(Mandatory)][string]$Mmproj,
 [ValidateSet(32,317)][int]$Chunk=32,
 [ValidateSet('legacy','reference')][string]$Merger='legacy',
 [int]$DeadlineSeconds=600,
 [string]$NativeSha256='4c428da9ce42f9a973bfb961928c760f1b1fbc61ef32d4f9ec840369f6269006'
)
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $Output) { throw 'Fresh output required.' }
New-Item -ItemType Directory -Path $Output | Out-Null
$before=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'TensorSharp.Server.Host|QwenImageLogitProbe|GlmUnifiedSessionProbe|QwenUnifiedPrefillProbe|llama-server|llama-cli|maya' })
$before | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Output 'preflight-processes.json') -Encoding utf8
if ($before.Count) { throw 'Existing model/Host processes: leave them untouched.' }
foreach ($name in @(Get-ChildItem Env: | Where-Object { $_.Name -match '^(TS_|TENSORSHARP_)' })) { throw "Inherited runtime override: $($name.Name)" }
$env:CUDA_VISIBLE_DEVICES='0'
$arguments=@((Join-Path $Application 'QwenImageLogitProbe.dll'),'--model',$Model,'--mmproj',$Mmproj,
 '--input',$InputFile,'--image',$Image,'--output',(Join-Path $Output 'run'),'--chunk',"$Chunk",'--merger',$Merger,
 '--timeout',"$DeadlineSeconds",'--native-sha256',$NativeSha256)
$started=[DateTime]::UtcNow
$process=Start-Process -FilePath 'C:/Program Files/dotnet/dotnet.exe' -ArgumentList $arguments -WorkingDirectory $Application -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Output 'stdout.log') -RedirectStandardError (Join-Path $Output 'stderr.log')
$ownedTicks=$process.StartTime.ToUniversalTime().Ticks
$identity=[ordered]@{Pid=$process.Id;StartTicks=$ownedTicks;StartedUtc=$started.ToString('o');Arguments=$arguments;DeadlineSeconds=$DeadlineSeconds}
$identity | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $Output 'owned.json') -Encoding utf8
$timedOut=$false
try {
 if (-not $process.WaitForExit($DeadlineSeconds*1000)) { $timedOut=$true; throw 'Owned process exceeded external hard deadline.' }
 $exit=$process.ExitCode
}
finally {
 $current=Get-Process -Id $process.Id -ErrorAction SilentlyContinue
 if ($current -and $current.StartTime.ToUniversalTime().Ticks -eq $ownedTicks) { Stop-Process -Id $current.Id -Force; $current.WaitForExit() }
 $remaining=Get-Process -Id $process.Id -ErrorAction SilentlyContinue
 $cleanup=-not($remaining -and $remaining.StartTime.ToUniversalTime().Ticks -eq $ownedTicks)
 [ordered]@{Pid=$process.Id;TimedOut=$timedOut;ExitCode=$exit;ElapsedSeconds=([DateTime]::UtcNow-$started).TotalSeconds;OwnedProcessGone=$cleanup;CompletedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content (Join-Path $Output 'execution.json') -Encoding utf8
}
if (-not $cleanup -or $exit -ne 0) { exit 1 }
exit 0
