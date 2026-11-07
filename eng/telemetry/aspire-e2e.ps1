# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#
# TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

<#
.SYNOPSIS
End-to-end check of TensorSharp's OpenTelemetry export against a standalone Aspire dashboard.

.DESCRIPTION
Starts an Aspire dashboard (a Docker container, Aspire.Dashboard.dll, or one that is already
running), starts TensorSharp.Server.Host several times with different telemetry settings, sends
real requests, then reads what the dashboard received through its telemetry HTTP API
(/api/telemetry/resources, /spans, /logs, /traces/{traceId}) and checks it.

Every check ends as PASS, FAIL, SKIPPED (its model or setup was not given) or MANUAL (it needs a
person to look at the dashboard UI). SKIPPED and MANUAL never count as passed. results.json,
results.md, environment.json and the evidence of every request go to -OutDir, which must be under
the git-ignored docs/validation/ or artifacts/ folders. The script exits with 1 when anything FAILED.

eng/telemetry/README.md lists the scenarios and explains how to read the results.

.EXAMPLE
pwsh eng/telemetry/aspire-e2e.ps1 -ChatConfig D:\models\qwen3-4b.json

.EXAMPLE
pwsh eng/telemetry/aspire-e2e.ps1 -ChatConfig "--model D:\models\gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda" -Scenarios llm-openai,llm-cancel

.EXAMPLE
pwsh eng/telemetry/aspire-e2e.ps1 -LtxConfig D:\models\ltx25.json -MiniMaxH3Config D:\models\minimax-h3.json -Scenarios video-ltx,video-minimax-h3,video-cancel,studio-generate
#>
#Requires -Version 7.2
[CmdletBinding()]
param(
    # A built TensorSharp.Server.Host: its .dll (run with dotnet), its executable, or a folder to
    # search. Default: the newest TensorSharp.Server.Host.dll under TensorSharp.Server.Host/bin.
    [string] $ServerHost = '',

    # Chat model for the llm-* and content-capture scenarios: a --config JSON file, a GGUF file
    # (passed as --model), or server arguments in one string ("--model x.gguf --backend ggml_cuda").
    [string] $ChatConfig = '',

    # Embedding model for the embeddings scenario, in the same forms. A GGUF file also gets
    # --embeddings; a --config file or an argument string must turn embeddings on itself.
    [string] $EmbeddingConfig = '',

    # LTX 2.5 video model for video-ltx (and video-cancel / studio-generate), in the same forms.
    [string] $LtxConfig = '',

    # MiniMax-H3 video model for video-minimax-h3 (and video-cancel / studio-generate without LTX).
    [string] $MiniMaxH3Config = '',

    # Run only these scenarios (a list, or one comma-separated string as `pwsh -File` passes it).
    # Default: all. A scenario whose model is not given is SKIPPED.
    [ArgumentCompletions('llm-openai-stream', 'llm-openai', 'llm-ollama-generate', 'llm-ollama-chat', 'llm-webui',
        'llm-client-traceparent', 'llm-cancel', 'llm-bad-request', 'llm-burst', 'queue-status-poll',
        'embeddings', 'video-ltx', 'video-minimax-h3', 'video-cancel', 'studio-generate',
        'negative-off', 'negative-sdk-disabled', 'negative-wrong-key', 'content-capture', 'metrics',
        'engine-step-spans')]
    [string[]] $Scenarios = @(),

    # Port TensorSharp.Server.Host listens on, bound to 127.0.0.1. The server runs one at a time.
    [ValidateRange(1, 65535)] [int] $Port = 5055,

    # Dashboard container image. Keep an exact tag so results name what they ran against.
    [string] $DashboardImage = 'mcr.microsoft.com/aspire/dashboard:13.6.2',

    # Evidence folder. Default: docs/validation/otel-e2e-<yyyyMMdd-HHmmss> under the repository root.
    [string] $OutDir = '',

    # Use a dashboard that is already running at -DashboardUrl instead of starting one.
    [switch] $SkipDocker,

    # With -SkipDocker: the dashboard UI URL (the telemetry API is served on the same port).
    [string] $DashboardUrl = 'http://localhost:18888',

    # With -SkipDocker: the dashboard's Dashboard:Api:PrimaryApiKey, sent as X-API-Key.
    [string] $DashboardApiKey = '',

    # With -SkipDocker: the dashboard's Dashboard:Otlp:PrimaryApiKey, sent as x-otlp-api-key.
    [string] $OtlpApiKey = '',

    # Run the dashboard from an Aspire.Dashboard.dll with dotnet instead of Docker.
    [string] $DashboardDll = '',

    # Leave the dashboard running at the end (its data is lost when it stops).
    [switch] $KeepDashboard,

    # Host ports of the dashboard UI / telemetry API, OTLP/gRPC and OTLP/HTTP.
    [ValidateRange(1, 65535)] [int] $DashboardUiPort = 18888,
    [ValidateRange(1, 65535)] [int] $OtlpGrpcPort = 4317,
    [ValidateRange(1, 65535)] [int] $OtlpHttpPort = 4318,

    # OTLP protocols to run the LLM scenarios over, one server run each: grpc and/or http/protobuf.
    [ArgumentCompletions('grpc', 'http/protobuf')] [string[]] $Protocols = @('grpc', 'http/protobuf'),

    # Extra arguments appended to every server command line, in one string ("--backend ggml_cuda").
    [string] $ExtraServerArgs = '',

    # max_tokens for the chat scenarios.
    [ValidateRange(2, 4096)] [int] $MaxTokens = 32,

    # Small clip sizes: LTX needs sides divisible by 64 and 8k+1 frames; MiniMax-H3 sides divisible
    # by 32 and 17k+5 frames.
    [string] $LtxSize = '512x320',
    [int] $LtxFrames = 17,

    # Stage spans expected under LTX's denoise phase (a single-stage LTX setup shows single-pass).
    [string[]] $LtxExpectStages = @('first-pass', 'upscale', 'refinement'),
    [string] $MiniMaxH3Size = '384x224',
    [int] $MiniMaxH3Frames = 22,

    # video-cancel drops the stream at the first denoise progress event, or after this many seconds.
    [int] $VideoCancelAfterSec = 120,

    [int] $StartupTimeoutSec = 1800,
    [int] $RequestTimeoutSec = 600,
    [int] $VideoTimeoutSec = 3600,

    # How long to wait for a request's telemetry to show up in the dashboard.
    [int] $TelemetryTimeoutSec = 60,

    # negative-wrong-key passes when its median latency is at most baseline * (1 + percent/100) + ms.
    [int] $LatencyMarginPercent = 50,
    [int] $LatencyMarginMs = 500,

    # How far (ms) a child span may stick out of its parent before it counts as outside it.
    [double] $ContainmentToleranceMs = 5,

    # Also turn on engine step spans (TENSORSHARP_OTEL_TRACE_STEPS=1) in the gRPC run and check them.
    [switch] $TraceSteps
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$Invariant = [Globalization.CultureInfo]::InvariantCulture

# The telemetry API returns the newest `limit` items (default 200). A request is looked up right
# after it finishes, so only its own spans and later ones come after its server span: the newest
# thousand is plenty, more with engine step spans on. Whole-run checks ask for everything.
$RecentSpanLimit = if ($TraceSteps) { 5000 } else { 1000 }
$AllLimit = 1000000
$VideoTelemetryTimeoutSec = [math]::Max($TelemetryTimeoutSec, 300)

$AllScenarioNames = @('llm-openai-stream', 'llm-openai', 'llm-ollama-generate', 'llm-ollama-chat', 'llm-webui',
    'llm-client-traceparent', 'llm-cancel', 'llm-bad-request', 'llm-burst', 'queue-status-poll',
    'embeddings', 'video-ltx', 'video-minimax-h3', 'video-cancel', 'studio-generate',
    'negative-off', 'negative-sdk-disabled', 'negative-wrong-key', 'content-capture', 'metrics',
    'engine-step-spans')
$LlmScenarios = @('llm-openai-stream', 'llm-openai', 'llm-ollama-generate', 'llm-ollama-chat', 'llm-webui',
    'llm-client-traceparent', 'llm-cancel', 'llm-bad-request', 'llm-burst')

# List parameters also accept one comma-separated string: `pwsh -File` passes "a,b" as one value.
function Split-ListParameter([string[]] $Values) {
    @($Values | ForEach-Object { "$_" -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -Unique)
}
$Selected = Split-ListParameter $Scenarios
$unknownScenarios = @($Selected | Where-Object { $AllScenarioNames -notcontains $_ })
if ($unknownScenarios) { throw "Unknown scenario(s) $($unknownScenarios -join ', '). Known: $($AllScenarioNames -join ', ')." }
if ($Selected.Count -eq 0) { $Selected = $AllScenarioNames }
$Protocols = Split-ListParameter $Protocols
$unknownProtocols = @($Protocols | Where-Object { $_ -notin 'grpc', 'http/protobuf' })
if ($unknownProtocols) { throw "Unknown protocol(s) $($unknownProtocols -join ', '); use grpc and/or http/protobuf." }
$LtxExpectStages = Split-ListParameter $LtxExpectStages

# Inherited variables that would change what a server run tests. Every run starts without them
# and gets only what the run sets.
$DroppedVariablePrefixes = @('OTEL_', 'TENSORSHARP_OTEL')
$DroppedVariables = @('APPLICATIONINSIGHTS_CONNECTION_STRING', 'TENSORSHARP_AZURE_MONITOR_AAD', 'PORT', 'HOST',
    'ASPNETCORE_URLS', 'TENSORSHARP_LOG_DIR', 'TS_STUDIO_DIRECTORY')

# ---------------------------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------------------------

function Write-Step([string] $Message) {
    Write-Host ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $Message)
}

function New-RandomHex([int] $Bytes) {
    [Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes($Bytes)).ToLowerInvariant()
}

function Save-Text([string] $Path, [string] $Text) {
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    [IO.File]::WriteAllText($Path, [string]$Text, [Text.UTF8Encoding]::new($false))
}

function Save-Json([string] $Path, $Value) {
    Save-Text $Path (ConvertTo-Json -InputObject $Value -Depth 32)
}

# Read a file another process is still writing.
function Read-SharedText([string] $Path) {
    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) { return '' }
    try {
        $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
        try { return [IO.StreamReader]::new($stream).ReadToEnd() } finally { $stream.Dispose() }
    } catch {
        return ''
    }
}

function Get-Median([double[]] $Values) {
    if (-not $Values -or $Values.Count -eq 0) { return $null }
    $sorted = [double[]]($Values | Sort-Object)
    $mid = [int][math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { return $sorted[$mid] }
    return ($sorted[$mid - 1] + $sorted[$mid]) / 2
}

# Split "--model 'C:\my models\x.gguf' --backend ggml_cuda" into arguments, honouring quotes.
function Split-CommandLine([string] $Text) {
    $tokens = [System.Collections.Generic.List[string]]::new()
    $current = [Text.StringBuilder]::new()
    $quote = ''
    $inToken = $false
    foreach ($ch in $Text.ToCharArray()) {
        $c = [string]$ch
        if ($quote) {
            if ($c -eq $quote) { $quote = '' } else { [void]$current.Append($c) }
        } elseif ($c -eq '"' -or $c -eq "'") {
            $quote = $c
            $inToken = $true
        } elseif ([char]::IsWhiteSpace($ch)) {
            if ($inToken) { $tokens.Add($current.ToString()); [void]$current.Clear(); $inToken = $false }
        } else {
            [void]$current.Append($c)
            $inToken = $true
        }
    }
    if ($quote) { throw "Unbalanced quote in: $Text" }
    if ($inToken) { $tokens.Add($current.ToString()) }
    return $tokens.ToArray()
}

# A -ChatConfig style value: a --config JSON file, a model file, or server arguments.
function Resolve-ModelArgs([string] $Spec, [string] $Label, [switch] $Embeddings) {
    if ([string]::IsNullOrWhiteSpace($Spec)) { return $null }
    $text = $Spec.Trim()
    if ($text.StartsWith('-')) { return , @(Split-CommandLine $text) }
    if (-not (Test-Path -LiteralPath $text -PathType Leaf)) {
        throw "-$Label '$Spec' is neither an existing file nor a list of server arguments starting with '-'."
    }
    $full = (Resolve-Path -LiteralPath $text).Path
    if ([IO.Path]::GetExtension($full) -ieq '.json') { return , @('--config', $full) }
    $list = @('--model', $full)
    if ($Embeddings) { $list += '--embeddings' }
    return , $list
}

function ConvertTo-Size([string] $Size) {
    if ($Size -notmatch '^\s*(\d+)\s*x\s*(\d+)\s*$') { throw "Size '$Size' is not WIDTHxHEIGHT." }
    [pscustomobject]@{ Width = [int]$Matches[1]; Height = [int]$Matches[2] }
}

# Poll $Probe until it returns an object whose .Done is true, or the time runs out. Returns the
# last non-null object the probe produced (Done or not), or $null.
function Wait-Until {
    param([Parameter(Mandatory)] [scriptblock] $Probe, [int] $TimeoutSec = 60, [double] $IntervalSec = 1.5)
    $__deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSec)
    $__last = $null
    while ($true) {
        $__result = & $Probe
        if ($null -ne $__result) {
            $__last = $__result
            if ($__result.Done) { return $__result }
        }
        if ([DateTime]::UtcNow -ge $__deadline) { return $__last }
        Start-Sleep -Milliseconds ([int]($IntervalSec * 1000))
    }
}
# How many times the server has logged the end of a request (the request logger's
# "HTTP <method> <path> -> <status>" line, written when the handler returns).
function Get-RequestEndCount($Ctx, [string] $Method, [string] $Path) {
    ([regex]::Matches((Read-SharedText $Ctx.Server.StdoutPath), [regex]::Escape("HTTP $Method $Path -> "))).Count
}

# Wait until the server logs the end of one more such request than $Before.
function Wait-RequestEnded($Ctx, [string] $Method, [string] $Path, [int] $Before, [int] $TimeoutSec) {
    $state = Wait-Until -TimeoutSec $TimeoutSec -IntervalSec 1 -Probe {
        [pscustomobject]@{ Done = ((Get-RequestEndCount $Ctx $Method $Path) -gt $Before) }
    }
    return [bool]($state -and $state.Done)
}

# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------

$script:Results = [System.Collections.Generic.List[object]]::new()

function Add-Result([string] $Run, [string] $Scenario, [string] $Status, [string] $Details = '', [double] $Seconds = 0) {
    if ($Status -notin 'PASS', 'FAIL', 'SKIPPED', 'MANUAL') { throw "Unknown status $Status" }
    $script:Results.Add([pscustomobject]@{
            run      = $Run
            scenario = $Scenario
            status   = $Status
            details  = $Details
            seconds  = [math]::Round($Seconds, 1)
        })
    $color = switch ($Status) { 'PASS' { 'Green' } 'FAIL' { 'Red' } 'SKIPPED' { 'DarkYellow' } default { 'Cyan' } }
    $shown = if ($Details.Length -gt 400) { $Details.Substring(0, 400) + '...' } else { $Details }
    Write-Host ("  {0,-7} {1} / {2}{3}" -f $Status, $Run, $Scenario, $(if ($shown) { ": $shown" } else { '' })) -ForegroundColor $color
}

# A check collects problems (Issues -> FAIL), observations (Notes), and why it could only be
# checked by hand (Manual -> MANUAL) or not at all (Skip -> SKIPPED).
function New-Check {
    [pscustomobject]@{
        Issues = [System.Collections.Generic.List[string]]::new()
        Notes  = [System.Collections.Generic.List[string]]::new()
        Manual = $null
        Skip   = $null
    }
}

function Complete-Check([string] $Run, [string] $Scenario, $Check, [double] $Seconds = 0) {
    $notes = ($Check.Notes -join '; ')
    if ($Check.Skip) {
        Add-Result $Run $Scenario 'SKIPPED' $Check.Skip $Seconds
    } elseif ($Check.Issues.Count -gt 0) {
        $text = $Check.Issues -join '; '
        if ($notes) { $text += " | notes: $notes" }
        Add-Result $Run $Scenario 'FAIL' $text $Seconds
    } elseif ($Check.Manual) {
        $text = $Check.Manual
        if ($notes) { $text += " | notes: $notes" }
        Add-Result $Run $Scenario 'MANUAL' $text $Seconds
    } else {
        Add-Result $Run $Scenario 'PASS' $notes $Seconds
    }
}

# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------

$script:HttpHandler = [System.Net.Http.SocketsHttpHandler]::new()
$script:HttpHandler.UseProxy = $false          # everything here is on this machine
$script:HttpHandler.AllowAutoRedirect = $false # the dashboard UI redirects to its login page
# Disposing an unfinished response must close the connection, so the server sees the client go.
# By default the handler first reads up to 1 MB / 2 s of the rest to reuse the connection,
# which for a short reply means the server streams it all and never sees a disconnect.
$script:HttpHandler.MaxResponseDrainSize = 0
$script:Http = [System.Net.Http.HttpClient]::new($script:HttpHandler)
$script:Http.Timeout = [System.Threading.Timeout]::InfiniteTimeSpan

function New-HttpRequest([string] $Method, [string] $Url, $Body, [hashtable] $Headers) {
    $message = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::new($Method), $Url)
    if ($Headers) {
        foreach ($key in $Headers.Keys) { [void]$message.Headers.TryAddWithoutValidation([string]$key, [string]$Headers[$key]) }
    }
    $text = $null
    if ($null -ne $Body) {
        $text = if ($Body -is [string]) { $Body } else { ConvertTo-Json -InputObject $Body -Depth 32 -Compress }
        $message.Content = [System.Net.Http.StringContent]::new($text, [Text.Encoding]::UTF8, 'application/json')
    }
    [pscustomobject]@{ Message = $message; BodyText = $text }
}

function New-HttpResult([string] $Method, [string] $Url, [string] $BodyText) {
    [pscustomobject]@{
        Method      = $Method
        Url         = $Url
        RequestBody = $BodyText
        Status      = 0
        Headers     = @{}
        Body        = ''
        Json        = $null
        Lines       = [System.Collections.Generic.List[string]]::new()
        RequestId   = $null
        ElapsedMs   = 0.0
        Error       = $null
        Aborted     = $false
    }
}

function Complete-HttpResult($Result, $Response) {
    $Result.Status = [int]$Response.StatusCode
    foreach ($header in $Response.Headers) { $Result.Headers[$header.Key] = ($header.Value -join ',') }
    foreach ($header in $Response.Content.Headers) { $Result.Headers[$header.Key] = ($header.Value -join ',') }
    $Result.RequestId = $Result.Headers['X-TensorSharp-Request-Id']
}

function ConvertFrom-JsonText([string] $Text) {
    if ([string]::IsNullOrWhiteSpace($Text)) { return $null }
    $trimmed = $Text.TrimStart()
    if (-not ($trimmed.StartsWith('{') -or $trimmed.StartsWith('['))) { return $null }
    try {
        $parsed = ConvertFrom-Json -InputObject $Text -NoEnumerate
        return , $parsed
    } catch {
        return $null
    }
}

function Invoke-Http {
    param(
        [string] $Method = 'GET',
        [Parameter(Mandatory)] [string] $Url,
        $Body = $null,
        [hashtable] $Headers = @{},
        [int] $TimeoutSec = 120,
        # Read the body line by line (server-sent events) into .Lines instead of .Body.
        [switch] $Stream,
        # Stream mode: stop reading, and drop the connection (which cancels the request on the
        # server), once this many lines matched -AbortPattern ...
        [int] $AbortAfterMatches = 0,
        [string] $AbortPattern = '^data:',
        # ... or once this many seconds have passed.
        [double] $AbortAfterSec = 0
    )
    $request = New-HttpRequest $Method $Url $Body $Headers
    $result = New-HttpResult $Method $Url $request.BodyText
    $cts = [System.Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($TimeoutSec))
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $response = $null
    try {
        $response = $script:Http.SendAsync($request.Message, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead, $cts.Token).GetAwaiter().GetResult()
        Complete-HttpResult $result $response
        if ($Stream) {
            $reader = [IO.StreamReader]::new($response.Content.ReadAsStreamAsync().GetAwaiter().GetResult(), [Text.Encoding]::UTF8)
            $matched = 0
            $pending = $reader.ReadLineAsync()
            while ($true) {
                # Wait in short slices so the time limits apply even while no line arrives.
                if (-not $pending.Wait(500)) {
                    if ($watch.Elapsed.TotalSeconds -ge $TimeoutSec) { $result.Error = "timed out after $TimeoutSec s"; $result.Aborted = $true; break }
                    if ($AbortAfterSec -gt 0 -and $watch.Elapsed.TotalSeconds -ge $AbortAfterSec) { $result.Aborted = $true; break }
                    continue
                }
                $line = $pending.Result
                if ($null -eq $line) { break }
                $result.Lines.Add($line)
                if ($AbortAfterMatches -gt 0 -and $line -match $AbortPattern) { $matched++ }
                if (($AbortAfterMatches -gt 0 -and $matched -ge $AbortAfterMatches) -or
                    ($AbortAfterSec -gt 0 -and $watch.Elapsed.TotalSeconds -ge $AbortAfterSec)) {
                    $result.Aborted = $true
                    break
                }
                $pending = $reader.ReadLineAsync()
            }
        } else {
            $result.Body = $response.Content.ReadAsStringAsync($cts.Token).GetAwaiter().GetResult()
            $result.Json = ConvertFrom-JsonText $result.Body
        }
    } catch {
        $result.Error = $_.Exception.GetBaseException().Message
    } finally {
        $watch.Stop()
        $result.ElapsedMs = $watch.Elapsed.TotalMilliseconds
        # Disposing an unfinished response closes the connection: the server sees the client go.
        if ($response) { $response.Dispose() }
        $request.Message.Dispose()
        $cts.Dispose()
    }
    return $result
}

# Send several requests at once (each @{ Method; Url; Body; Headers }) and wait for all of them.
# ElapsedMs is approximate: completions are observed in order.
function Invoke-HttpConcurrent([object[]] $Specs, [int] $TimeoutSec) {
    $pending = foreach ($spec in $Specs) {
        $request = New-HttpRequest $spec.Method $spec.Url $spec.Body $spec.Headers
        $cts = [System.Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($TimeoutSec))
        $watch = [Diagnostics.Stopwatch]::StartNew()
        [pscustomobject]@{
            Request = $request
            Cts     = $cts
            Watch   = $watch
            Result  = (New-HttpResult $spec.Method $spec.Url $request.BodyText)
            Task    = $script:Http.SendAsync($request.Message, [System.Net.Http.HttpCompletionOption]::ResponseContentRead, $cts.Token)
        }
    }
    foreach ($item in $pending) {
        try {
            $response = $item.Task.GetAwaiter().GetResult()
            Complete-HttpResult $item.Result $response
            $item.Result.Body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            $item.Result.Json = ConvertFrom-JsonText $item.Result.Body
            $response.Dispose()
        } catch {
            $item.Result.Error = $_.Exception.GetBaseException().Message
        } finally {
            $item.Result.ElapsedMs = $item.Watch.Elapsed.TotalMilliseconds
            $item.Request.Message.Dispose()
            $item.Cts.Dispose()
        }
        $item.Result
    }
}

# The JSON payloads of the "data:" lines of a server-sent event stream.
function ConvertFrom-Sse($Lines) {
    foreach ($line in @($Lines)) {
        if ([string]::IsNullOrEmpty($line) -or -not $line.StartsWith('data:')) { continue }
        $payload = $line.Substring(5).Trim()
        if (-not $payload -or $payload -eq '[DONE]') { continue }
        $event = ConvertFrom-JsonText $payload
        if ($null -ne $event) { $event }
    }
}

function Get-ErrorText($Result) {
    if ($Result.Error) { return [string]$Result.Error }
    $body = [string]$Result.Body
    if ($body.Length -gt 300) { $body = $body.Substring(0, 300) + '...' }
    return $body
}

# One request and its response, for the evidence folder.
function Save-Exchange($Ctx, [string] $Name, $Result) {
    $body = [string]$Result.Body
    if ($body.Length -gt 65536) { $body = $body.Substring(0, 65536) + "... [$($Result.Body.Length) chars]" }
    $lines = @($Result.Lines)
    $Ctx.ExchangeCount++
    Save-Json (Join-Path $Ctx.Dir ("requests/{0:D3}-{1}.json" -f $Ctx.ExchangeCount, $Name)) ([ordered]@{
            method          = $Result.Method
            url             = $Result.Url
            status          = $Result.Status
            elapsedMs       = [math]::Round($Result.ElapsedMs, 1)
            requestId       = $Result.RequestId
            error           = $Result.Error
            aborted         = $Result.Aborted
            requestBody     = $Result.RequestBody
            responseHeaders = $Result.Headers
            responseBody    = $body
            streamLines     = $lines.Count
            streamHead      = @($lines | Select-Object -First 20)
            streamTail      = @($lines | Select-Object -Last 10)
        })
}

function Test-TcpPort([string] $HostName, [int] $PortNumber) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        return ($client.ConnectAsync($HostName, $PortNumber).Wait(500) -and $client.Connected)
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Wait-PortFree([int] $PortNumber, [int] $TimeoutSec = 30) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSec)
    while ((Test-TcpPort '127.0.0.1' $PortNumber) -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 500 }
}

# ---------------------------------------------------------------------------------------------
# Processes: TensorSharp.Server.Host and (optionally) Aspire.Dashboard.dll
# ---------------------------------------------------------------------------------------------

function Resolve-ServerHost([string] $Path) {
    if (-not $Path) { $Path = Join-Path $RepoRoot 'TensorSharp.Server.Host/bin' }
    $full = [IO.Path]::GetFullPath($Path)
    if (Test-Path -LiteralPath $full -PathType Container) {
        $found = Get-ChildItem -LiteralPath $full -Recurse -File -Filter 'TensorSharp.Server.Host.dll' -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
        if (-not $found) { throw "No TensorSharp.Server.Host.dll under $full. Build TensorSharp.Server.Host first, or pass -ServerHost." }
        $full = $found.FullName
    }
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { throw "-ServerHost $full does not exist." }
    if ([IO.Path]::GetExtension($full) -ieq '.dll') {
        $dotnet = Get-Command dotnet -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $dotnet) { throw "dotnet is not on PATH; it is needed to run $full." }
        return [pscustomobject]@{ FileName = $dotnet.Source; Prefix = @($full); Path = $full; Directory = (Split-Path -Parent $full) }
    }
    return [pscustomobject]@{ FileName = $full; Prefix = @(); Path = $full; Directory = (Split-Path -Parent $full) }
}

function Test-DroppedVariable([string] $Name) {
    foreach ($prefix in $DroppedVariablePrefixes) {
        if ($Name.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return ($DroppedVariables -contains $Name)
}

# Start a process with its output streamed straight into files (unbuffered, so the files can be
# read while it runs). Inherited telemetry variables are dropped first when asked.
function Start-LoggedProcess {
    param(
        [string] $FileName, [string[]] $Arguments, [hashtable] $Environment, [string] $WorkingDirectory,
        [string] $StdoutPath, [string] $StderrPath, [switch] $DropTelemetryVariables
    )
    $psi = [System.Diagnostics.ProcessStartInfo]::new($FileName)
    foreach ($argument in $Arguments) { $psi.ArgumentList.Add([string]$argument) }
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.WorkingDirectory = $WorkingDirectory
    if ($DropTelemetryVariables) {
        foreach ($key in @($psi.Environment.Keys)) { if (Test-DroppedVariable $key) { [void]$psi.Environment.Remove($key) } }
    }
    foreach ($key in $Environment.Keys) { $psi.Environment[[string]$key] = [string]$Environment[$key] }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $psi
    if (-not $process.Start()) { throw "Could not start $FileName" }
    $outFile = [IO.FileStream]::new($StdoutPath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite, 1)
    $errFile = [IO.FileStream]::new($StderrPath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite, 1)
    [pscustomobject]@{
        Process    = $process
        Pid        = $process.Id
        StdoutPath = $StdoutPath
        StderrPath = $StderrPath
        OutFile    = $outFile
        ErrFile    = $errFile
        CopyOut    = $process.StandardOutput.BaseStream.CopyToAsync($outFile)
        CopyErr    = $process.StandardError.BaseStream.CopyToAsync($errFile)
        ExitCode   = $null
    }
}

function Stop-LoggedProcess($Handle, [int] $GraceSec = 20) {
    if ($null -eq $Handle -or $null -eq $Handle.Process) { return }
    $process = $Handle.Process
    try {
        if (-not $process.HasExited) {
            # SIGTERM lets the host shut down and flush its logs; Windows has no equivalent for a
            # console child, so it is killed there. Telemetry has been read before this point.
            if (-not $IsWindows -and (Test-Path '/bin/kill')) {
                & '/bin/kill' '-TERM' "$($Handle.Pid)" 2>$null
                [void]$process.WaitForExit($GraceSec * 1000)
            }
            if (-not $process.HasExited) {
                $process.Kill($true)
                [void]$process.WaitForExit(15000)
            }
        }
        if ($process.HasExited) { $Handle.ExitCode = $process.ExitCode }
    } catch {
        Write-Warning "Stopping process $($Handle.Pid): $($_.Exception.Message)"
    }
    try { [void][System.Threading.Tasks.Task]::WaitAll([System.Threading.Tasks.Task[]]@($Handle.CopyOut, $Handle.CopyErr), 10000) } catch { }
    $Handle.OutFile.Dispose()
    $Handle.ErrFile.Dispose()
    $process.Dispose()
    $Handle.Process = $null
}

function Start-TsServer([string] $Dir, [string[]] $ServerArgs, [hashtable] $Environment) {
    $handle = Start-LoggedProcess -FileName $script:ServerHostInfo.FileName `
        -Arguments (@($script:ServerHostInfo.Prefix) + @($ServerArgs)) `
        -Environment $Environment -WorkingDirectory (Get-Location).Path `
        -StdoutPath (Join-Path $Dir 'server.stdout.log') -StderrPath (Join-Path $Dir 'server.stderr.log') `
        -DropTelemetryVariables
    $handle | Add-Member -NotePropertyName Base -NotePropertyValue "http://127.0.0.1:$Port"
    $handle | Add-Member -NotePropertyName LogDir -NotePropertyValue $Environment['TENSORSHARP_LOG_DIR']
    $handle | Add-Member -NotePropertyName HealthPolls -NotePropertyValue 0
    return $handle
}

function Wait-ServerReady($Server) {
    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSec)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($Server.Process.HasExited) {
            throw "the server exited with code $($Server.Process.ExitCode) before /health answered; see $($Server.StderrPath)"
        }
        $Server.HealthPolls++
        $health = Invoke-Http -Url "$($Server.Base)/health" -TimeoutSec 5
        if ($health.Status -eq 200) { return }
        Start-Sleep -Seconds 2
    }
    throw "the server did not answer /health within $StartupTimeoutSec s; see $($Server.StdoutPath)"
}

# The host's one-line telemetry summary, e.g. "Telemetry: traces(otlp) metrics(otlp) logs(otlp) -> OTLP ...".
function Get-TelemetryLine($Server) {
    $sources = @($Server.StdoutPath, $Server.StderrPath)
    if ($Server.LogDir -and (Test-Path -LiteralPath $Server.LogDir)) {
        $sources += @(Get-ChildItem -LiteralPath $Server.LogDir -File -Recurse -ErrorAction SilentlyContinue | ForEach-Object FullName)
    }
    foreach ($path in $sources) {
        $match = [regex]::Match((Read-SharedText $path), 'Telemetry: (?:off \(|traces\()[^\r\n]*')
        if ($match.Success) { return $match.Value.TrimEnd() }
    }
    return $null
}

# ---------------------------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------------------------

# The settings both the container and Aspire.Dashboard.dll get. Names checked against
# aspire.dev/dashboard/configuration and the dashboard's DashboardOptions (v13.6.2):
# Dashboard:Api:Enabled is the older spelling of !Dashboard:Api:Disabled; both are set so either
# version enables the API. ASPIRE_DASHBOARD_UNSECURED_ALLOW_ANONYMOUS is never set: it would also
# turn OTLP authentication off.
function Get-DashboardSettings {
    [ordered]@{
        'Dashboard__Otlp__AuthMode'      = 'ApiKey'
        'Dashboard__Otlp__PrimaryApiKey' = $script:Dashboard.OtlpKey
        'Dashboard__Api__Enabled'        = 'true'
        'Dashboard__Api__Disabled'       = 'false'
        'Dashboard__Api__AuthMode'       = 'ApiKey'
        'Dashboard__Api__PrimaryApiKey'  = $script:Dashboard.ApiKey
    }
}

function Invoke-Docker([string[]] $Arguments) {
    $output = & docker @Arguments 2>&1 | ForEach-Object { "$_" }
    [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = ($output -join "`n") }
}

function Start-Dashboard {
    $dash = $script:Dashboard
    $dashDir = Join-Path $OutDir 'dashboard'
    New-Item -ItemType Directory -Path $dashDir -Force | Out-Null

    if ($SkipDocker) {
        $dash.Mode = 'external'
        $dash.Url = $DashboardUrl.TrimEnd('/')
        $dash.ApiKey = $DashboardApiKey
        $dash.OtlpKey = $OtlpApiKey
        $hostName = ([uri]$dash.Url).Host
        $dash.Grpc = "http://${hostName}:$OtlpGrpcPort"
        $dash.Http = "http://${hostName}:$OtlpHttpPort"
        Write-Step "Using the dashboard at $($dash.Url)"
    } else {
        foreach ($p in $DashboardUiPort, $OtlpGrpcPort, $OtlpHttpPort) {
            if (Test-TcpPort '127.0.0.1' $p) {
                throw "Port $p is already in use (another dashboard or collector?). Stop it, choose other ports, or pass -SkipDocker."
            }
        }
        $dash.ApiKey = New-RandomHex 16
        $dash.OtlpKey = New-RandomHex 16
        $dash.Url = "http://127.0.0.1:$DashboardUiPort"
        $dash.Grpc = "http://127.0.0.1:$OtlpGrpcPort"
        $dash.Http = "http://127.0.0.1:$OtlpHttpPort"
        if ($DashboardDll) { Start-DashboardDll $dashDir } else { Start-DashboardContainer }
    }
    $script:Api.Base = $dash.Url
    $script:Api.Key = $dash.ApiKey
    Wait-DashboardReady
    Save-DashboardLogin $dashDir
}

function Start-DashboardContainer {
    $dash = $script:Dashboard
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'docker is not on PATH. Install Docker, pass -DashboardDll <Aspire.Dashboard.dll>, or pass -SkipDocker with a running dashboard.'
    }
    $info = Invoke-Docker @('info', '--format', '{{.ServerVersion}}')
    if ($info.ExitCode -ne 0) { throw "The Docker daemon is not reachable: $($info.Output)" }
    $dash.Mode = 'docker'
    $dash.Image = $DashboardImage
    if ((Invoke-Docker @('image', 'inspect', $DashboardImage)).ExitCode -ne 0) {
        Write-Step "Pulling $DashboardImage"
        $pull = Invoke-Docker @('pull', $DashboardImage)
        if ($pull.ExitCode -ne 0) { throw "docker pull $DashboardImage failed: $($pull.Output)" }
    }
    $dash.ImageId = (Invoke-Docker @('image', 'inspect', '--format', '{{.Id}}', $DashboardImage)).Output.Trim()
    $digests = ConvertFrom-JsonText (Invoke-Docker @('image', 'inspect', '--format', '{{json .RepoDigests}}', $DashboardImage)).Output
    $dash.Digests = @($digests | Where-Object { $_ })

    $name = "tensorsharp-otel-e2e-$script:RunTag"
    # Container ports: 18888 UI and telemetry API, 18889 OTLP/gRPC, 18890 OTLP/HTTP. Published on
    # loopback only. No --rm, so the logs survive a crash; Stop-Dashboard removes it.
    $arguments = @('run', '-d', '--name', $name,
        '-p', "127.0.0.1:${DashboardUiPort}:18888",
        '-p', "127.0.0.1:${OtlpGrpcPort}:18889",
        '-p', "127.0.0.1:${OtlpHttpPort}:18890")
    $settings = Get-DashboardSettings
    foreach ($key in $settings.Keys) { $arguments += @('-e', "$key=$($settings[$key])") }
    $arguments += $DashboardImage
    Write-Step "Starting dashboard container $name from $DashboardImage"
    $run = Invoke-Docker $arguments
    if ($run.ExitCode -ne 0) { throw "docker run failed: $($run.Output)" }
    $dash.Container = $name
}

function Start-DashboardDll([string] $DashDir) {
    $dash = $script:Dashboard
    $dll = (Resolve-Path -LiteralPath $DashboardDll).Path
    $dotnet = Get-Command dotnet -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $dotnet) { throw 'dotnet is not on PATH; it is needed to run Aspire.Dashboard.dll.' }
    $dash.Mode = 'dll'
    $dash.Dll = $dll
    $settings = Get-DashboardSettings
    # Endpoint settings: ASPNETCORE_URLS and the ASPIRE_DASHBOARD_OTLP_* names are the ones
    # aspire.dev documents; the Dashboard__* names are the same options in DashboardOptions.
    $settings['ASPNETCORE_URLS'] = $dash.Url
    $settings['Dashboard__Frontend__EndpointUrls'] = $dash.Url
    $settings['ASPIRE_DASHBOARD_OTLP_ENDPOINT_URL'] = $dash.Grpc
    $settings['Dashboard__Otlp__GrpcEndpointUrl'] = $dash.Grpc
    $settings['ASPIRE_DASHBOARD_OTLP_HTTP_ENDPOINT_URL'] = $dash.Http
    $settings['Dashboard__Otlp__HttpEndpointUrl'] = $dash.Http
    $environment = @{}
    foreach ($key in $settings.Keys) { $environment[$key] = $settings[$key] }
    Write-Step "Starting the dashboard from $dll"
    $dash.Process = Start-LoggedProcess -FileName $dotnet.Source -Arguments @($dll) -Environment $environment `
        -WorkingDirectory (Split-Path -Parent $dll) -StdoutPath (Join-Path $DashDir 'dashboard.stdout.log') `
        -StderrPath (Join-Path $DashDir 'dashboard.stderr.log') -DropTelemetryVariables
}

function Get-DashboardLogText {
    $dash = $script:Dashboard
    switch ($dash.Mode) {
        'docker' { return (Invoke-Docker @('logs', $dash.Container)).Output }
        'dll' { return (Read-SharedText $dash.Process.StdoutPath) + "`n" + (Read-SharedText $dash.Process.StderrPath) }
        default { return '' }
    }
}

# Ready when the telemetry API answers. A 401 or 404 will not change by waiting: the API is then
# marked unusable and the telemetry checks become MANUAL.
function Wait-DashboardReady {
    $dash = $script:Dashboard
    $headers = @{}
    if ($dash.ApiKey) { $headers['X-API-Key'] = $dash.ApiKey }
    $deadline = [DateTime]::UtcNow.AddSeconds(180)
    while ($true) {
        if ($dash.Mode -eq 'docker') {
            $state = Invoke-Docker @('inspect', '-f', '{{.State.Running}}', $dash.Container)
            if ($state.Output.Trim() -ne 'true') { throw "The dashboard container stopped. Its log:`n$(Get-DashboardLogText)" }
        } elseif ($dash.Mode -eq 'dll' -and $dash.Process.Process.HasExited) {
            throw "Aspire.Dashboard.dll exited with code $($dash.Process.Process.ExitCode); see $($dash.Process.StderrPath)"
        }
        $probe = Invoke-Http -Url "$($dash.Url)/api/telemetry/resources" -Headers $headers -TimeoutSec 10
        if ($probe.Status -eq 200) {
            $script:Api.Usable = $true
            $script:Api.Reason = ''
            Write-Step "Dashboard ready at $($dash.Url) (telemetry API answers)"
            return
        }
        if ($probe.Status -in 401, 403) {
            $script:Api.Reason = "the dashboard rejected the telemetry API key (HTTP $($probe.Status))"
            break
        }
        if ($probe.Status -eq 404) {
            $script:Api.Reason = 'the dashboard has no /api/telemetry routes (HTTP 404: the API is disabled, or the dashboard predates it)'
            break
        }
        if ([DateTime]::UtcNow -ge $deadline) { throw "The dashboard at $($dash.Url) did not answer within 180 s ($($probe.Error))." }
        Start-Sleep -Seconds 2
    }
    $script:Api.Usable = $false
    $script:Limitations.Add("Telemetry API unusable: $($script:Api.Reason). Telemetry checks are MANUAL.")
    Write-Warning "Telemetry API unusable: $($script:Api.Reason). Telemetry checks will be MANUAL."
}

# The browser login link the dashboard prints at startup ("...login?t=<token>").
function Save-DashboardLogin([string] $DashDir) {
    $dash = $script:Dashboard
    $file = Join-Path $DashDir 'login.txt'
    if ($dash.Mode -eq 'external') {
        Save-Text $file "Dashboard: $($dash.Url) (started outside this script; use its own login token)`n"
        $script:Api.UiHint = "open $($dash.Url)"
        return
    }
    $found = Wait-Until -TimeoutSec 30 -IntervalSec 1 -Probe {
        $match = [regex]::Match((Get-DashboardLogText), 'login\?t=([A-Za-z0-9_\-]+)')
        if ($match.Success) { [pscustomobject]@{ Done = $true; Token = $match.Groups[1].Value } }
    }
    if ($found) {
        $dash.LoginUrl = "$($dash.Url)/login?t=$($found.Token)"
        Save-Text $file ("Dashboard login (valid until the dashboard stops): $($dash.LoginUrl)`n")
        $script:Api.UiHint = "open the login URL in dashboard/login.txt (rerun with -KeepDashboard so the data is still there afterwards)"
    } else {
        Save-Text $file "No login?t= token found in the dashboard log within 30 s. Dashboard: $($dash.Url)`n"
        $script:Api.UiHint = "open $($dash.Url) (the login token is in the dashboard log)"
    }
}

function Save-DashboardSnapshot {
    if (-not $script:Api.Usable) { return }
    $dashDir = Join-Path $OutDir 'dashboard'
    Save-Text (Join-Path $dashDir 'resources.json') (Invoke-DashboardApi '/api/telemetry/resources').Body
    Save-Text (Join-Path $dashDir 'spans-all.json') (Invoke-DashboardApi '/api/telemetry/spans' @{ limit = $AllLimit }).Body
    Save-Text (Join-Path $dashDir 'logs-all.json') (Invoke-DashboardApi '/api/telemetry/logs' @{ limit = $AllLimit }).Body
}

function Stop-Dashboard {
    $dash = $script:Dashboard
    $dashDir = Join-Path $OutDir 'dashboard'
    if ($dash.Mode -in 'docker', 'dll') { Save-Text (Join-Path $dashDir 'dashboard.log') (Get-DashboardLogText) }
    if ($KeepDashboard -and $dash.Mode -in 'docker', 'dll') {
        Save-Json (Join-Path $dashDir 'keys.json') ([ordered]@{
                url = $dash.Url; telemetryApiKey = $dash.ApiKey; otlpApiKey = $dash.OtlpKey
                grpc = $dash.Grpc; http = $dash.Http; container = $dash.Container
            })
        Write-Step "Dashboard left running: $($dash.LoginUrl) (keys in dashboard/keys.json)"
        if ($dash.Container) { Write-Step "Remove it with: docker rm -f $($dash.Container)" }
        return
    }
    switch ($dash.Mode) {
        'docker' {
            Write-Step "Stopping dashboard container $($dash.Container) (its telemetry is deleted)"
            [void](Invoke-Docker @('stop', $dash.Container))
            [void](Invoke-Docker @('rm', '-f', $dash.Container))
        }
        'dll' { Stop-LoggedProcess $dash.Process -GraceSec 10 }
    }
}

# ---------------------------------------------------------------------------------------------
# Telemetry API and OTLP JSON
#
# Routes and query parameters, as in aspire.dev/dashboard/apis and the dashboard's own
# DashboardEndpointsBuilder.cs / TelemetryApiService.cs at v13.6.2:
#   GET /api/telemetry/resources                 [{ name, instanceId, displayName, hasLogs, ... }]
#   GET /api/telemetry/spans?resource=&traceId=&limit=
#   GET /api/telemetry/logs?resource=&traceId=&limit=
#   GET /api/telemetry/traces/{traceId}
# Header X-API-Key. List responses are { data: { resourceSpans | resourceLogs }, totalCount,
# returnedCount }, limit defaults to 200 and keeps the newest items, an unknown resource is 404,
# and every attribute value comes back as a stringValue.
# ---------------------------------------------------------------------------------------------

function Invoke-DashboardApi([string] $Path, [hashtable] $Query = @{}) {
    $parts = foreach ($key in $Query.Keys) {
        foreach ($value in @($Query[$key])) { "$([uri]::EscapeDataString([string]$key))=$([uri]::EscapeDataString([string]$value))" }
    }
    $url = $script:Api.Base.TrimEnd('/') + $Path
    if ($parts) { $url += '?' + ($parts -join '&') }
    $headers = @{}
    if ($script:Api.Key) { $headers['X-API-Key'] = $script:Api.Key }
    $result = Invoke-Http -Url $url -Headers $headers -TimeoutSec 120
    if ($result.Status -ne 200 -and $result.Status -ne 404) {
        $script:Api.LastError = "GET $Path -> HTTP $($result.Status) $($result.Error)"
    }
    if ($result.Json -and $result.Json.PSObject.Properties['totalCount'] -and
        [long]$result.Json.totalCount -gt [long]$result.Json.returnedCount) {
        $script:Api.LastTruncation = "GET $Path returned $($result.Json.returnedCount) of $($result.Json.totalCount)"
    }
    return $result
}

function ConvertTo-HexId($Value) {
    if ($null -eq $Value) { return '' }
    $text = [string]$Value
    if ($text -match '^0*$') { return '' }
    if ($text -match '^[0-9a-fA-F]+$') { return $text.ToLowerInvariant() }
    try { return [Convert]::ToHexString([Convert]::FromBase64String($text)).ToLowerInvariant() } catch { return $text }
}

function ConvertTo-Nanos($Value) {
    if ($null -eq $Value) { return [decimal]0 }
    $parsed = [decimal]0
    if ([decimal]::TryParse([string]$Value, [Globalization.NumberStyles]::Integer, $Invariant, [ref]$parsed)) { return $parsed }
    return [decimal]0
}

function ConvertTo-SpanKind($Value) {
    if ($null -eq $Value) { return 'UNSPECIFIED' }
    switch ([string]$Value) {
        '0' { return 'UNSPECIFIED' }
        '1' { return 'INTERNAL' }
        '2' { return 'SERVER' }
        '3' { return 'CLIENT' }
        '4' { return 'PRODUCER' }
        '5' { return 'CONSUMER' }
        default { return (([string]$Value).ToUpperInvariant() -replace '^SPAN_KIND_', '') }
    }
}

function ConvertFrom-OtlpAnyValue($Value) {
    if ($null -eq $Value) { return $null }
    foreach ($name in 'stringValue', 'intValue', 'doubleValue', 'boolValue', 'bytesValue') {
        $property = $Value.PSObject.Properties[$name]
        if ($property) { return $property.Value }
    }
    $array = $Value.PSObject.Properties['arrayValue']
    if ($array) { return , @(foreach ($item in @($array.Value.values)) { ConvertFrom-OtlpAnyValue $item }) }
    $kv = $Value.PSObject.Properties['kvlistValue']
    if ($kv) { return (ConvertFrom-OtlpAttributeList $kv.Value.values) }
    return (ConvertTo-Json -InputObject $Value -Compress -Depth 8)
}

# OTLP [{ key, value: { stringValue | ... } }] -> case-insensitive hashtable.
function ConvertFrom-OtlpAttributeList($List) {
    $table = @{}
    foreach ($pair in @($List)) {
        if ($null -eq $pair -or $null -eq $pair.key) { continue }
        $table[[string]$pair.key] = ConvertFrom-OtlpAnyValue $pair.value
    }
    return $table
}

function Get-OtlpData($Json) {
    if ($null -eq $Json) { return $null }
    if ($Json.PSObject.Properties['data']) { return $Json.data }
    return $Json
}

function ConvertFrom-OtlpSpans($Json) {
    $data = Get-OtlpData $Json
    if ($null -eq $data) { return }
    foreach ($resourceSpans in @($data.resourceSpans)) {
        if ($null -eq $resourceSpans) { continue }
        $resource = ConvertFrom-OtlpAttributeList $resourceSpans.resource.attributes
        foreach ($scopeSpans in @($resourceSpans.scopeSpans)) {
            if ($null -eq $scopeSpans) { continue }
            $scope = [string]$scopeSpans.scope.name
            foreach ($span in @($scopeSpans.spans)) {
                if ($null -eq $span) { continue }
                $start = ConvertTo-Nanos $span.startTimeUnixNano
                $end = ConvertTo-Nanos $span.endTimeUnixNano
                [pscustomobject]@{
                    TraceId      = ConvertTo-HexId $span.traceId
                    SpanId       = ConvertTo-HexId $span.spanId
                    ParentSpanId = ConvertTo-HexId $span.parentSpanId
                    Name         = [string]$span.name
                    Kind         = ConvertTo-SpanKind $span.kind
                    StartNs      = $start
                    EndNs        = $end
                    DurationMs   = [double](($end - $start) / 1000000)
                    Attr         = ConvertFrom-OtlpAttributeList $span.attributes
                    Resource     = $resource
                    Scope        = $scope
                    StatusCode   = [string]$span.status.code
                    Links        = @(foreach ($link in @($span.links)) {
                            if ($link) { [pscustomobject]@{ TraceId = ConvertTo-HexId $link.traceId; SpanId = ConvertTo-HexId $link.spanId } }
                        })
                    Events       = @(foreach ($event in @($span.events)) { if ($event) { [string]$event.name } })
                }
            }
        }
    }
}

function ConvertFrom-OtlpLogs($Json) {
    $data = Get-OtlpData $Json
    if ($null -eq $data) { return }
    foreach ($resourceLogs in @($data.resourceLogs)) {
        if ($null -eq $resourceLogs) { continue }
        $resource = ConvertFrom-OtlpAttributeList $resourceLogs.resource.attributes
        foreach ($scopeLogs in @($resourceLogs.scopeLogs)) {
            if ($null -eq $scopeLogs) { continue }
            $scope = [string]$scopeLogs.scope.name
            foreach ($record in @($scopeLogs.logRecords)) {
                if ($null -eq $record) { continue }
                [pscustomobject]@{
                    TraceId  = ConvertTo-HexId $record.traceId
                    SpanId   = ConvertTo-HexId $record.spanId
                    Severity = [string]$record.severityText
                    Body     = [string](ConvertFrom-OtlpAnyValue $record.body)
                    Attr     = ConvertFrom-OtlpAttributeList $record.attributes
                    Resource = $resource
                    Scope    = $scope
                }
            }
        }
    }
}

function Get-ServiceSpans([string] $Service, [int] $Limit = $RecentSpanLimit) {
    $result = Invoke-DashboardApi '/api/telemetry/spans' @{ resource = $Service; limit = $Limit }
    if ($result.Status -ne 200) { return }
    ConvertFrom-OtlpSpans $result.Json | Where-Object { $_.Resource['service.name'] -eq $Service }
}

function Get-ServiceLogs([string] $Service, [string] $TraceId = '', [int] $Limit = $AllLimit) {
    $query = @{ resource = $Service; limit = $Limit }
    if ($TraceId) { $query['traceId'] = $TraceId }
    $result = Invoke-DashboardApi '/api/telemetry/logs' $query
    if ($result.Status -ne 200) { return }
    ConvertFrom-OtlpLogs $result.Json | Where-Object { $_.Resource['service.name'] -eq $Service }
}

function Get-TraceSpans([string] $TraceId) {
    $result = Invoke-DashboardApi "/api/telemetry/traces/$TraceId"
    if ($result.Status -ne 200) { return }
    $script:TraceBodies[$TraceId] = $result.Body
    ConvertFrom-OtlpSpans $result.Json
}

function Save-TraceEvidence($Ctx, [string] $TraceId, [string] $Label) {
    if (-not $TraceId -or -not $script:TraceBodies.ContainsKey($TraceId)) { return }
    $safe = ($Label -replace '[^A-Za-z0-9_.-]+', '-').Trim('-')
    Save-Text (Join-Path $Ctx.Dir "telemetry/traces/$safe-$TraceId.json") $script:TraceBodies[$TraceId]
}

function Get-AttrNumber($Attr, [string] $Key) {
    if ($null -eq $Attr -or -not $Attr.ContainsKey($Key) -or $null -eq $Attr[$Key]) { return $null }
    $parsed = 0.0
    if ([double]::TryParse([Convert]::ToString($Attr[$Key], $Invariant), [Globalization.NumberStyles]::Float, $Invariant, [ref]$parsed)) { return $parsed }
    return $null
}

# ---------------------------------------------------------------------------------------------
# Span trees
# ---------------------------------------------------------------------------------------------

function Get-SpanIndex($Spans) {
    $children = @{}
    $byId = @{}
    foreach ($span in @($Spans)) {
        if ($null -eq $span) { continue }
        $byId[$span.SpanId] = $span
        $parent = [string]$span.ParentSpanId
        if (-not $children.ContainsKey($parent)) { $children[$parent] = [System.Collections.Generic.List[object]]::new() }
        $children[$parent].Add($span)
    }
    [pscustomobject]@{ Children = $children; ById = $byId }
}

function Get-ChildSpans($Index, $Span) {
    if ($Span -and $Index.Children.ContainsKey($Span.SpanId)) { $Index.Children[$Span.SpanId] }
}

function Get-DescendantSpans($Index, $Span) {
    $queue = [System.Collections.Generic.Queue[object]]::new()
    $queue.Enqueue($Span)
    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        foreach ($child in @(Get-ChildSpans $Index $current)) {
            $child
            $queue.Enqueue($child)
        }
    }
}

function Test-IsDescendant($Index, $Span, $Ancestor) {
    $cursor = $Span
    for ($i = 0; $i -lt 128 -and $cursor; $i++) {
        if (-not $cursor.ParentSpanId) { return $false }
        if ($cursor.ParentSpanId -eq $Ancestor.SpanId) { return $true }
        $cursor = $Index.ById[$cursor.ParentSpanId]
    }
    return $false
}

# Every span under $Root must lie within its parent's start and end.
function Test-Containment($Index, $Root, $Check, [string] $Label, [switch] $AsNotes) {
    $tolerance = [decimal]($ContainmentToleranceMs * 1000000)
    foreach ($span in @(Get-DescendantSpans $Index $Root)) {
        $parent = $Index.ById[$span.ParentSpanId]
        if ($null -eq $parent) { continue }
        if ($span.StartNs -lt $parent.StartNs - $tolerance -or $span.EndNs -gt $parent.EndNs + $tolerance) {
            $text = "{0}span '{1}' ({2:F1}..{3:F1} ms) lies outside its parent '{4}' ({5:F1}..{6:F1} ms, times from the trace root)" -f `
                $(if ($Label) { "${Label}: " } else { '' }), $span.Name,
            [double](($span.StartNs - $Root.StartNs) / 1000000), [double](($span.EndNs - $Root.StartNs) / 1000000),
            $parent.Name, [double](($parent.StartNs - $Root.StartNs) / 1000000), [double](($parent.EndNs - $Root.StartNs) / 1000000)
            if ($AsNotes) { $Check.Notes.Add($text) } else { $Check.Issues.Add($text) }
        }
    }
}

function Test-SpanPath($Span, [string] $Path) {
    ($Span.Attr['url.path'] -eq $Path) -or ($Span.Attr['http.route'] -eq $Path) -or ($Span.Name -eq "GET $Path")
}

# ---------------------------------------------------------------------------------------------
# Request checks
# ---------------------------------------------------------------------------------------------

function Test-ApiUsable($Check, [string] $What) {
    if ($script:Api.Usable) { return $true }
    $Check.Manual = "telemetry API unusable ($($script:Api.Reason)); check $What in the dashboard UI: $($script:Api.UiHint)"
    return $false
}

function Get-ApiDiagnostics {
    $parts = @()
    if ($script:Api.LastError) { $parts += "last API error: $($script:Api.LastError)" }
    if ($script:Api.LastTruncation) { $parts += "listing truncated: $($script:Api.LastTruncation)" }
    if ($parts) { return ' (' + ($parts -join '; ') + ')' }
    return ''
}

# Wait until the HTTP server span of a request (found by its tensorsharp.request_id, or in a known
# trace) and the spans the caller needs are in the dashboard. Returns { Done, Server, Trace, Index }
# with whatever arrived, or $null when not even the server span did.
function Find-RequestTelemetry {
    param(
        $Ctx, [string] $RequestId, [string] $TraceId = '',
        # A gen_ai span with this gen_ai.operation.name and the same request id must be there ...
        [string] $GenAiOperation = '',
        # ... with at least one child whose name is in this list (the request has finished) ...
        [string[]] $WaitChildAny = @(),
        # ... and spans with these names (wildcards allowed) somewhere in the trace.
        [string[]] $RequireNames = @(),
        [int] $TimeoutSec = $TelemetryTimeoutSec
    )
    $found = Wait-Until -TimeoutSec $TimeoutSec -Probe {
        $pool = if ($TraceId) { @(Get-TraceSpans $TraceId) } else { @(Get-ServiceSpans $Ctx.Service) }
        $server = $pool | Where-Object { $_.Kind -eq 'SERVER' -and $_.Attr['tensorsharp.request_id'] -eq $RequestId } | Select-Object -First 1
        if (-not $server) { return $null }
        $trace = if ($TraceId) { $pool } else { @(Get-TraceSpans $server.TraceId) }
        if ($trace.Count -eq 0) { $trace = @($pool | Where-Object { $_.TraceId -eq $server.TraceId }) }
        $ready = $true
        if ($GenAiOperation) {
            $gens = @($trace | Where-Object { $_.Attr['gen_ai.operation.name'] -eq $GenAiOperation -and $_.Attr['tensorsharp.request_id'] -eq $RequestId })
            if ($gens.Count -eq 0) {
                $ready = $false
            } elseif ($WaitChildAny.Count -gt 0) {
                $genIds = @($gens | ForEach-Object SpanId)
                $kids = @($trace | Where-Object { $genIds -contains $_.ParentSpanId -and $WaitChildAny -contains $_.Name })
                if ($kids.Count -eq 0) { $ready = $false }
            }
        }
        $names = @($trace | ForEach-Object Name)
        foreach ($pattern in $RequireNames) {
            if (-not ($names | Where-Object { $_ -like $pattern })) { $ready = $false }
        }
        [pscustomobject]@{ Done = $ready; Server = $server; Trace = $trace }
    }
    if ($found) { $found | Add-Member -NotePropertyName Index -NotePropertyValue (Get-SpanIndex $found.Trace) -Force }
    return $found
}

# Log records in the request's trace must carry tensorsharp.request_id = the response header.
function Test-RequestLogs($Ctx, [string] $RequestId, [string] $TraceId, $Check, [string] $Prefix = '') {
    $state = Wait-Until -TimeoutSec $TelemetryTimeoutSec -Probe {
        $logs = @(Get-ServiceLogs -Service $Ctx.Service -TraceId $TraceId)
        $mine = @($logs | Where-Object { $_.Attr['tensorsharp.request_id'] -eq $RequestId })
        [pscustomobject]@{ Done = ($mine.Count -gt 0); Logs = $logs; Mine = $mine }
    }
    if (-not $state -or $state.Mine.Count -eq 0) {
        $count = if ($state) { $state.Logs.Count } else { 0 }
        $Check.Issues.Add("${Prefix}none of the $count exported log record(s) in trace $TraceId carries tensorsharp.request_id=$RequestId$(Get-ApiDiagnostics)")
        return
    }
    $other = @($state.Logs | Where-Object { $_.Attr.ContainsKey('tensorsharp.request_id') -and $_.Attr['tensorsharp.request_id'] -ne $RequestId })
    if ($other.Count -gt 0) {
        $Check.Issues.Add("${Prefix}$($other.Count) log record(s) in the trace carry another tensorsharp.request_id ($($other[0].Attr['tensorsharp.request_id']))")
    }
    $Check.Notes.Add("${Prefix}$($state.Mine.Count) of $($state.Logs.Count) log records in the trace carry the request id")
}

# The telemetry of one generation request (chat, text_completion or embeddings), or of a plain
# request with -NoGenAiSpan. Returns what Find-RequestTelemetry found, or $null.
function Test-GenAiRequest {
    param(
        $Ctx, $Response, $Check,
        [string] $Operation = 'chat',
        [string] $Protocol = '',
        # @{ Input; Output; Cached } from the response's usage block, or $null.
        $Usage = $null,
        [string] $Outcome = 'completed',
        # A traceparent the request sent: the server span must continue it.
        [string] $TraceId = '', [string] $ParentSpanId = '',
        [string] $Label = '',
        [switch] $NoGenAiSpan
    )
    $prefix = if ($Label) { "${Label}: " } else { '' }
    $requestId = $Response.RequestId
    if (-not $requestId) { $Check.Issues.Add("${prefix}the response has no X-TensorSharp-Request-Id header"); return $null }
    if (-not (Test-ApiUsable $Check "the trace of request $requestId (search tensorsharp.request_id)")) { return $null }

    $waitKids = @()
    if (-not $NoGenAiSpan -and $Operation -ne 'embeddings') {
        $waitKids = if ($Outcome -eq 'completed') { @('inference.decode', 'diffusion.denoise') } else { @('inference.queue', 'diffusion.denoise') }
    }
    $found = Find-RequestTelemetry -Ctx $Ctx -RequestId $requestId -TraceId $TraceId `
        -GenAiOperation $(if ($NoGenAiSpan) { '' } else { $Operation }) -WaitChildAny $waitKids
    if (-not $found) {
        $Check.Issues.Add("${prefix}no HTTP server span with tensorsharp.request_id=$requestId reached the dashboard within $TelemetryTimeoutSec s$(Get-ApiDiagnostics)")
        return $null
    }
    $server = $found.Server
    $trace = $found.Trace
    $index = $found.Index
    Save-TraceEvidence $Ctx $server.TraceId "$($Ctx.CurrentScenario)-$Label"

    if ($TraceId -and $server.TraceId -ne $TraceId) { $Check.Issues.Add("${prefix}the server span is in trace $($server.TraceId), not the client's $TraceId") }
    if ($ParentSpanId -and $server.ParentSpanId -ne $ParentSpanId) {
        $Check.Issues.Add("${prefix}the server span's parent is '$($server.ParentSpanId)', not the client span $ParentSpanId from traceparent")
    }
    if (-not $ParentSpanId -and $server.ParentSpanId) { $Check.Issues.Add("${prefix}the server span has a parent ($($server.ParentSpanId)) although the request sent no traceparent") }
    if ($Protocol -and $server.Attr['tensorsharp.protocol'] -ne $Protocol) {
        $Check.Issues.Add("${prefix}server span tensorsharp.protocol is '$($server.Attr['tensorsharp.protocol'])', expected '$Protocol'")
    }
    $status = Get-AttrNumber $server.Attr 'http.response.status_code'
    if ($null -ne $status -and $Response.Status -gt 0 -and [int]$status -ne $Response.Status) {
        $Check.Issues.Add("${prefix}server span http.response.status_code is $status, the response was $($Response.Status)")
    }

    $gens = @($trace | Where-Object {
            $_.Attr['gen_ai.operation.name'] -eq $Operation -and $_.Attr['tensorsharp.request_id'] -eq $requestId -and
            $_.Attr['tensorsharp.internal'] -ne 'true'
        })
    if ($NoGenAiSpan) {
        if ($gens.Count -gt 0) { $Check.Issues.Add("${prefix}unexpected '$($gens[0].Name)' span for a request that generated nothing") }
    } elseif ($gens.Count -eq 0) {
        $Check.Issues.Add("${prefix}no '$Operation <model>' span with tensorsharp.request_id=$requestId in trace $($server.TraceId) (spans: $((@($trace | ForEach-Object Name | Select-Object -Unique)) -join ', '))")
    } else {
        # A tool-calling reply has one span per round. Only the last round can end the way the
        # request did; a round that handed over to a tool call completed.
        $gens = @($gens | Sort-Object StartNs)
        for ($i = 0; $i -lt $gens.Count; $i++) {
            $expected = if ($i -lt $gens.Count - 1) { 'completed' } else { $Outcome }
            Test-GenAiSpan $gens[$i] $server $index $Check $Operation $Protocol $expected $prefix
        }
        if ($Usage) { Test-TokenUsage $gens $Usage $Check $Operation $prefix }
    }
    Test-Containment $index $server $Check $Label
    Test-RequestLogs $Ctx $requestId $server.TraceId $Check $prefix
    return $found
}

function Test-GenAiSpan($Gen, $Server, $Index, $Check, [string] $Operation, [string] $Protocol, [string] $Outcome, [string] $Prefix) {
    if (-not (Test-IsDescendant $Index $Gen $Server)) { $Check.Issues.Add("${Prefix}'$($Gen.Name)' is not under the HTTP server span") }
    $model = [string]$Gen.Attr['gen_ai.request.model']
    if (-not $model) {
        $Check.Issues.Add("${Prefix}'$($Gen.Name)' has no gen_ai.request.model")
    } elseif ($Gen.Name -ne "$Operation $model") {
        $Check.Issues.Add("${Prefix}span name '$($Gen.Name)' is not '$Operation $model'")
    }
    if ($Gen.Attr['gen_ai.provider.name'] -ne 'tensorsharp') { $Check.Issues.Add("${Prefix}gen_ai.provider.name is '$($Gen.Attr['gen_ai.provider.name'])', expected tensorsharp") }
    if ($Protocol -and $Gen.Attr['tensorsharp.protocol'] -ne $Protocol) { $Check.Issues.Add("${Prefix}'$($Gen.Name)' tensorsharp.protocol is '$($Gen.Attr['tensorsharp.protocol'])', expected '$Protocol'") }
    if ($Outcome -and $Gen.Attr['tensorsharp.outcome'] -ne $Outcome) { $Check.Issues.Add("${Prefix}tensorsharp.outcome is '$($Gen.Attr['tensorsharp.outcome'])', expected '$Outcome'") }

    $kids = @(Get-ChildSpans $Index $Gen | ForEach-Object Name)
    if ($Operation -in 'chat', 'text_completion') {
        $required = if ($kids -contains 'diffusion.denoise') { @('prompt.render', 'diffusion.denoise') }
        elseif ($Outcome -eq 'completed') { @('prompt.render', 'inference.queue', 'inference.prefill', 'inference.decode') }
        else { @('inference.queue') }
        $missing = @($required | Where-Object { $kids -notcontains $_ })
        if ($missing) { $Check.Issues.Add("${Prefix}'$($Gen.Name)' children are [$($kids -join ', ')]; missing [$($missing -join ', ')]") }
        if ($Outcome -eq 'completed') {
            $ttft = Get-AttrNumber $Gen.Attr 'tensorsharp.ttft_ms'
            if ($null -eq $ttft -or $ttft -le 0) { $Check.Issues.Add("${Prefix}'$($Gen.Name)' has no positive tensorsharp.ttft_ms ('$($Gen.Attr['tensorsharp.ttft_ms'])')") }
            else { $Check.Notes.Add("${Prefix}ttft $([math]::Round($ttft, 1)) ms") }
        }
    }
    $Check.Notes.Add("${Prefix}'$($Gen.Name)' children: $($kids -join ', ')")
}

# Token counts on the gen_ai span(s) must equal the response's usage block. Several spans (tool
# rounds) are summed.
function Test-TokenUsage($Gens, $Usage, $Check, [string] $Operation, [string] $Prefix) {
    $isChat = $Operation -in 'chat', 'text_completion'
    $pairs = @(
        @{ Key = 'gen_ai.usage.input_tokens'; Expected = $Usage.Input; Name = 'input' },
        @{ Key = 'gen_ai.usage.output_tokens'; Expected = $Usage.Output; Name = 'output' },
        @{ Key = 'gen_ai.usage.cache_read.input_tokens'; Expected = $Usage.Cached; Name = 'cache_read'; ZeroWhenMissing = $true }
    )
    $shown = @()
    foreach ($pair in $pairs) {
        if ($null -eq $pair.Expected) { continue }
        $sum = 0.0
        $missing = 0
        foreach ($gen in $Gens) {
            $value = Get-AttrNumber $gen.Attr $pair.Key
            if ($null -eq $value) { $missing++ } else { $sum += $value }
        }
        if ($missing -eq $Gens.Count -and -not $pair.ZeroWhenMissing) {
            if ($isChat) { $Check.Issues.Add("${Prefix}no $($pair.Key) on the span") } else { $Check.Notes.Add("${Prefix}no $($pair.Key) on the span") }
            continue
        }
        if ([int]$sum -ne [int]$pair.Expected) {
            $Check.Issues.Add("${Prefix}span $($pair.Key) = $sum, the response says $($pair.Expected)")
        }
        $shown += "$($pair.Name) $([int]$sum)"
    }
    if ($shown) { $Check.Notes.Add("${Prefix}tokens on span match the response ($($shown -join ', '))") }
}

# A video generation's trace: generate_video <model> under the server span, its phases and stages,
# video.export, children inside parents, and the logs' request id.
function Test-VideoRequest {
    param(
        $Ctx, $Response, $Check, [string] $Label = 'clip',
        [string[]] $Phases = @(),
        # Phase name -> stage span names expected somewhere under it.
        [hashtable] $Stages = @{},
        [string] $Outcome = 'completed'
    )
    $prefix = "${Label}: "
    $requestId = $Response.RequestId
    if (-not $requestId) { $Check.Issues.Add("${prefix}the response has no X-TensorSharp-Request-Id header"); return $null }
    if (-not (Test-ApiUsable $Check "the generate_video trace of request $requestId")) { return $null }

    $require = @('generate_video *')
    if ($Outcome -eq 'completed') { $require += 'video.export' }
    $found = Find-RequestTelemetry -Ctx $Ctx -RequestId $requestId -RequireNames $require -TimeoutSec $VideoTelemetryTimeoutSec
    if (-not $found) {
        $Check.Issues.Add("${prefix}no HTTP server span with tensorsharp.request_id=$requestId reached the dashboard$(Get-ApiDiagnostics)")
        return $null
    }
    Save-TraceEvidence $Ctx $found.Server.TraceId "$($Ctx.CurrentScenario)-$Label"
    $gen = @($found.Trace | Where-Object { $_.Name -like 'generate_video *' })[0]
    if (-not $gen) {
        $Check.Issues.Add("${prefix}no 'generate_video <model>' span in trace $($found.Server.TraceId) (spans: $((@($found.Trace | ForEach-Object Name | Select-Object -Unique)) -join ', '))")
        return $null
    }
    if (-not (Test-IsDescendant $found.Index $gen $found.Server)) { $Check.Issues.Add("${prefix}'$($gen.Name)' is not under the HTTP server span") }
    $descendants = @(Get-DescendantSpans $found.Index $gen)
    $Check.Notes.Add("${prefix}under '$($gen.Name)': $((@($descendants | ForEach-Object Name | Select-Object -Unique)) -join ', ')")

    $outcomeValue = [string]$gen.Attr['tensorsharp.outcome']
    if ($Outcome -eq 'cancelled') {
        $errorType = [string]$gen.Attr['error.type']
        if ($outcomeValue -ne 'cancelled' -and $errorType -notmatch 'Cancel') {
            $Check.Issues.Add("${prefix}'$($gen.Name)' outcome is '$outcomeValue' (error.type '$errorType'), expected cancelled")
        }
    } elseif ($outcomeValue -and $outcomeValue -ne $Outcome) {
        $Check.Issues.Add("${prefix}'$($gen.Name)' tensorsharp.outcome is '$outcomeValue', expected '$Outcome'")
    }

    foreach ($phase in $Phases) {
        $phaseSpans = @($descendants | Where-Object { $_.Name -eq $phase })
        if ($phaseSpans.Count -eq 0) { $Check.Issues.Add("${prefix}no '$phase' phase span under '$($gen.Name)'"); continue }
        if ($Stages.ContainsKey($phase)) {
            $stageNames = @($phaseSpans | ForEach-Object { Get-DescendantSpans $found.Index $_ } | ForEach-Object Name | Select-Object -Unique)
            $missing = @($Stages[$phase] | Where-Object { $stageNames -notcontains $_ })
            if ($missing) { $Check.Issues.Add("${prefix}'$phase' has no stage span(s) $($missing -join ', ') (it has: $($stageNames -join ', '))") }
        }
    }
    if ($Outcome -eq 'completed' -and -not ($descendants | Where-Object { $_.Name -eq 'video.export' })) {
        $Check.Issues.Add("${prefix}no video.export span under '$($gen.Name)'")
    }
    # A cancelled stream's handler returns before the worker unwinds, so its spans may outlive
    # the server span; that is reported, not failed.
    Test-Containment $found.Index $found.Server $Check $Label -AsNotes:($Outcome -ne 'completed')
    Test-RequestLogs $Ctx $requestId $found.Server.TraceId $Check $prefix
    return [pscustomobject]@{ Found = $found; Gen = $gen; Descendants = $descendants }
}

# ---------------------------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------------------------

# Prompts carry a unique marker, so the run can prove no exported log or span contains them.
function New-Prompt($Ctx, [string] $Suffix = '', [switch] $Video, [switch] $Long) {
    $Ctx.MarkerSent = $true
    if ($Video) { return "Telemetry check $script:Marker$Suffix. A red ball rolls slowly across a wooden table in soft daylight." }
    # A reply long enough to still be streaming when a client drops it (the cancel checks).
    if ($Long) { return "Telemetry check $script:Marker. Count from 1 to 500, one number per line, with no other text." }
    return "Telemetry check $script:Marker$Suffix. Reply with one short sentence about the sea."
}

function Get-OpenAIUsage($Usage) {
    if ($null -eq $Usage) { return $null }
    [pscustomobject]@{ Input = [int]$Usage.prompt_tokens; Output = [int]$Usage.completion_tokens; Cached = [int]$Usage.prompt_tokens_details.cached_tokens }
}

function Invoke-OpenAIChat {
    param($Ctx, [switch] $Stream, [int] $Tokens = 0, [string] $Prompt = '', [hashtable] $Headers = @{}, [int] $AbortAfterChunks = 0,
        # tool_choice "none": no skill or tool rounds, so the reply streams from its first token.
        [switch] $NoTools)
    if (-not $Tokens) { $Tokens = $MaxTokens }
    if (-not $Prompt) { $Prompt = New-Prompt $Ctx }
    $body = [ordered]@{
        model       = $Ctx.Model
        messages    = @(@{ role = 'user'; content = $Prompt })
        max_tokens  = $Tokens
        temperature = 0
        stream      = [bool]$Stream
    }
    if ($NoTools) { $body['tool_choice'] = 'none' }
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/v1/chat/completions" -Body $body -Headers $Headers `
        -TimeoutSec $RequestTimeoutSec -Stream:$Stream -AbortAfterMatches $AbortAfterChunks
    $usage = $null
    $sawDone = $false
    if ($Stream) {
        # The final chunk carries the usage block.
        foreach ($event in @(ConvertFrom-Sse $response.Lines)) { if ($event.usage) { $usage = $event.usage } }
        $sawDone = [bool]($response.Lines | Where-Object { $_ -match '^data:\s*\[DONE\]' })
    } elseif ($response.Json) {
        $usage = $response.Json.usage
    }
    [pscustomobject]@{ Response = $response; Usage = (Get-OpenAIUsage $usage); SawDone = $sawDone }
}

function Test-RequireModel($Ctx, $Check) {
    if ($Ctx.Model) { return $true }
    $Check.Issues.Add('the server lists no model under /v1/models, so there is nothing to send requests to')
    return $false
}

function Get-ModelId($Ctx) {
    $result = Invoke-Http -Url "$($Ctx.Base)/v1/models" -TimeoutSec 60
    Save-Exchange $Ctx 'v1-models' $result
    $id = @($result.Json.data)[0].id
    if ($id) { return [string]$id }
    return $null
}

# Median latency of a fixed request: a short chat when a model is loaded, else GET /v1/models.
# The first request warms up and is not counted.
function Measure-Latency($Ctx) {
    $kind = if ($Ctx.Model) { 'chat' } else { 'models' }
    $samples = [System.Collections.Generic.List[double]]::new()
    for ($i = 0; $i -lt 6; $i++) {
        $response = if ($kind -eq 'chat') { (Invoke-OpenAIChat -Ctx $Ctx -Tokens 8).Response } else { Invoke-Http -Url "$($Ctx.Base)/v1/models" -TimeoutSec 60 }
        if ($response.Status -ne 200) { throw "latency probe request failed: HTTP $($response.Status) $(Get-ErrorText $response)" }
        if ($i -gt 0) { $samples.Add($response.ElapsedMs) }
    }
    [pscustomobject]@{ Kind = $kind; Run = $Ctx.Name; SamplesMs = @($samples | ForEach-Object { [math]::Round($_, 1) }); MedianMs = (Get-Median $samples.ToArray()) }
}

# ---------------------------------------------------------------------------------------------
# Scenarios. Each takes the run context and a check and records what it finds.
# ---------------------------------------------------------------------------------------------

function Test-LlmOpenAI($Ctx, $Check, [switch] $Stream) {
    if (-not (Test-RequireModel $Ctx $Check)) { return }
    $chat = Invoke-OpenAIChat -Ctx $Ctx -Stream:$Stream
    Save-Exchange $Ctx $Ctx.CurrentScenario $chat.Response
    if ($chat.Response.Status -ne 200) { $Check.Issues.Add("HTTP $($chat.Response.Status): $(Get-ErrorText $chat.Response)"); return }
    if ($Stream -and -not $chat.SawDone) { $Check.Issues.Add('the stream ended without data: [DONE]') }
    if (-not $chat.Usage) { $Check.Issues.Add('the response has no usage block') }
    $Check.Notes.Add("request id $($chat.Response.RequestId)")
    Test-GenAiRequest -Ctx $Ctx -Response $chat.Response -Check $Check -Operation chat -Protocol openai -Usage $chat.Usage | Out-Null
}

function Test-LlmOllamaGenerate($Ctx, $Check) {
    if (-not (Test-RequireModel $Ctx $Check)) { return }
    $body = [ordered]@{ model = $Ctx.Model; prompt = (New-Prompt $Ctx); stream = $false; options = @{ num_predict = $MaxTokens; temperature = 0 } }
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/api/generate" -Body $body -TimeoutSec $RequestTimeoutSec
    Save-Exchange $Ctx $Ctx.CurrentScenario $response
    if ($response.Status -ne 200) { $Check.Issues.Add("HTTP $($response.Status): $(Get-ErrorText $response)"); return }
    $usage = [pscustomobject]@{ Input = [int]$response.Json.prompt_eval_count; Output = [int]$response.Json.eval_count; Cached = $null }
    Test-GenAiRequest -Ctx $Ctx -Response $response -Check $Check -Operation text_completion -Protocol ollama -Usage $usage | Out-Null
}

function Test-LlmOllamaChat($Ctx, $Check) {
    if (-not (Test-RequireModel $Ctx $Check)) { return }
    $body = [ordered]@{
        model    = $Ctx.Model
        messages = @(@{ role = 'user'; content = (New-Prompt $Ctx) })
        stream   = $false
        options  = @{ num_predict = $MaxTokens; temperature = 0 }
    }
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/api/chat/ollama" -Body $body -TimeoutSec $RequestTimeoutSec
    Save-Exchange $Ctx $Ctx.CurrentScenario $response
    if ($response.Status -ne 200) { $Check.Issues.Add("HTTP $($response.Status): $(Get-ErrorText $response)"); return }
    $usage = [pscustomobject]@{ Input = [int]$response.Json.prompt_eval_count; Output = [int]$response.Json.eval_count; Cached = $null }
    Test-GenAiRequest -Ctx $Ctx -Response $response -Check $Check -Operation chat -Protocol ollama -Usage $usage | Out-Null
}

function Test-LlmWebUi($Ctx, $Check) {
    if (-not (Test-RequireModel $Ctx $Check)) { return }
    $body = [ordered]@{ messages = @(@{ role = 'user'; content = (New-Prompt $Ctx) }); maxTokens = $MaxTokens; newChat = $true }
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/api/chat" -Body $body -TimeoutSec $RequestTimeoutSec -Stream
    Save-Exchange $Ctx $Ctx.CurrentScenario $response
    if ($response.Status -ne 200) { $Check.Issues.Add("HTTP $($response.Status): $(Get-ErrorText $response)"); return }
    $done = @(ConvertFrom-Sse $response.Lines | Where-Object { $_.done -eq $true }) | Select-Object -Last 1
    if (-not $done) { $Check.Issues.Add('the event stream has no done frame'); return }
    if ($done.error) { $Check.Issues.Add("the done frame reports an error: $($done.error)") }
    $usage = [pscustomobject]@{ Input = [int]$done.promptTokens; Output = [int]$done.tokenCount; Cached = [int]$done.kvReusedTokens }
    Test-GenAiRequest -Ctx $Ctx -Response $response -Check $Check -Operation chat -Protocol webui -Usage $usage | Out-Null
}

# A W3C traceparent from the client: the server span must continue that trace under that span.
function Test-LlmClientTraceparent($Ctx, $Check) {
    if (-not (Test-RequireModel $Ctx $Check)) { return }
    $traceId = New-RandomHex 16
    $parentId = New-RandomHex 8
    $chat = Invoke-OpenAIChat -Ctx $Ctx -Headers @{ traceparent = "00-$traceId-$parentId-01" }
    Save-Exchange $Ctx $Ctx.CurrentScenario $chat.Response
    if ($chat.Response.Status -ne 200) { $Check.Issues.Add("HTTP $($chat.Response.Status): $(Get-ErrorText $chat.Response)"); return }
    $Check.Notes.Add("sent traceparent 00-$traceId-$parentId-01")
    Test-GenAiRequest -Ctx $Ctx -Response $chat.Response -Check $Check -Operation chat -Protocol openai -Usage $chat.Usage `
        -TraceId $traceId -ParentSpanId $parentId | Out-Null
}

# Drop a stream mid-way: the chat span must end as cancelled and the engine must let go.
function Test-LlmCancel($Ctx, $Check) {
    if (-not (Test-RequireModel $Ctx $Check)) { return }
    $prompt = New-Prompt $Ctx -Long
    $chat = Invoke-OpenAIChat -Ctx $Ctx -Stream -Tokens 1024 -Prompt $prompt -AbortAfterChunks 3 -NoTools
    Save-Exchange $Ctx $Ctx.CurrentScenario $chat.Response
    if ($chat.Response.Status -ne 200) { $Check.Issues.Add("HTTP $($chat.Response.Status): $(Get-ErrorText $chat.Response)"); return }
    if (-not $chat.Response.Aborted) { $Check.Issues.Add('the stream finished before 3 chunks arrived, so nothing was cancelled'); return }
    if (@($chat.Response.Lines | Where-Object { $_ -match '"finish_reason":"[a-z_]+"' }).Count -gt 0) { $Check.Issues.Add('the model finished its reply before the client disconnected, so there was nothing to cancel; the check needs a reply that is still streaming'); return }
    $Check.Notes.Add("dropped the connection after $($chat.Response.Lines.Count) stream lines")
    Test-GenAiRequest -Ctx $Ctx -Response $chat.Response -Check $Check -Operation chat -Protocol openai -Outcome cancelled | Out-Null
    $idle = Wait-Until -TimeoutSec 60 -Probe {
        $queue = Invoke-Http -Url "$($Ctx.Base)/api/queue/status" -TimeoutSec 10
        if ($queue.Json) { [pscustomobject]@{ Done = ([int]$queue.Json.processing -eq 0); Processing = [int]$queue.Json.processing } }
    }
    if (-not $idle -or -not $idle.Done) { $Check.Issues.Add("the engine still reports processing=$($idle.Processing) 60 s after the client went away") }
}

function Test-LlmBadRequest($Ctx, $Check) {
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/v1/chat/completions" -Body ([ordered]@{ model = [string]$Ctx.Model }) -TimeoutSec 60
    Save-Exchange $Ctx $Ctx.CurrentScenario $response
    if ($response.Status -ne 400) { $Check.Issues.Add("expected HTTP 400 for a request without messages, got $($response.Status): $(Get-ErrorText $response)") }
    Test-GenAiRequest -Ctx $Ctx -Response $response -Check $Check -Protocol openai -NoGenAiSpan | Out-Null
}

# Eight requests at once: each gets its own trace with a complete chat tree.
function Test-LlmBurst($Ctx, $Check) {
    if (-not (Test-RequireModel $Ctx $Check)) { return }
    $specs = foreach ($n in 1..8) {
        @{
            Method  = 'POST'
            Url     = "$($Ctx.Base)/v1/chat/completions"
            Headers = @{}
            Body    = [ordered]@{
                model = $Ctx.Model; messages = @(@{ role = 'user'; content = (New-Prompt $Ctx " (burst $n)") })
                max_tokens = $MaxTokens; temperature = 0; stream = $false
            }
        }
    }
    $results = @(Invoke-HttpConcurrent $specs $RequestTimeoutSec)
    for ($i = 0; $i -lt $results.Count; $i++) { Save-Exchange $Ctx "llm-burst-$($i + 1)" $results[$i] }
    $ok = @($results | Where-Object { $_.Status -eq 200 })
    if ($ok.Count -ne $results.Count) {
        $bad = @($results | Where-Object { $_.Status -ne 200 })[0]
        $Check.Issues.Add("$($results.Count - $ok.Count) of $($results.Count) requests failed (first: HTTP $($bad.Status) $(Get-ErrorText $bad))")
    }
    $ids = @($ok | ForEach-Object RequestId | Where-Object { $_ } | Select-Object -Unique)
    if ($ids.Count -ne $ok.Count) { $Check.Issues.Add("only $($ids.Count) distinct request ids for $($ok.Count) responses") }
    $traces = [System.Collections.Generic.HashSet[string]]::new()
    for ($i = 0; $i -lt $results.Count; $i++) {
        if ($results[$i].Status -ne 200) { continue }
        $found = Test-GenAiRequest -Ctx $Ctx -Response $results[$i] -Check $Check -Operation chat -Protocol openai `
            -Usage (Get-OpenAIUsage $results[$i].Json.usage) -Label "request $($i + 1)"
        if ($found) { [void]$traces.Add($found.Server.TraceId) }
    }
    if ($script:Api.Usable -and $traces.Count -ne $ok.Count) { $Check.Issues.Add("$($ok.Count) requests produced $($traces.Count) distinct traces") }
}

# The Web UI polls /api/queue/status; those requests must not become server spans.
function Test-QueueStatusPoll($Ctx, $Check) {
    $ids = [System.Collections.Generic.List[string]]::new()
    for ($i = 1; $i -le 20; $i++) {
        $response = Invoke-Http -Url "$($Ctx.Base)/api/queue/status" -TimeoutSec 30
        if ($response.Status -ne 200) { $Check.Issues.Add("poll ${i}: HTTP $($response.Status) $(Get-ErrorText $response)"); return }
        if ($response.RequestId) { $ids.Add($response.RequestId) }
    }
    $Check.Notes.Add('20 polls answered 200')
    if (-not (Test-ApiUsable $Check "that no 'GET /api/queue/status' span exists for $($Ctx.Service)")) { return }
    # A traced request made after the polls must arrive, so "no span" means filtered, not lost.
    $control = Invoke-Http -Url "$($Ctx.Base)/api/models" -TimeoutSec 60
    $found = Find-RequestTelemetry -Ctx $Ctx -RequestId $control.RequestId
    if (-not $found) {
        $Check.Issues.Add("the control request GET /api/models never reached the dashboard, so the missing queue-status spans prove nothing$(Get-ApiDiagnostics)")
        return
    }
    $spans = @(Get-ServiceSpans $Ctx.Service -Limit $AllLimit)
    $byPath = @($spans | Where-Object { $_.Kind -eq 'SERVER' -and (Test-SpanPath $_ '/api/queue/status') })
    $byId = @($spans | Where-Object { $ids -contains $_.Attr['tensorsharp.request_id'] })
    if ($byPath.Count -gt 0) { $Check.Issues.Add("$($byPath.Count) server span(s) for /api/queue/status were exported") }
    if ($byId.Count -gt 0) { $Check.Issues.Add("$($byId.Count) span(s) carry the request id of a queue-status poll") }
    $Check.Notes.Add("searched $($spans.Count) spans; the control request's span arrived")
}

function Test-Embeddings($Ctx, $Check) {
    if (-not (Test-RequireModel $Ctx $Check)) { return }
    $openai = Invoke-Http -Method POST -Url "$($Ctx.Base)/v1/embeddings" -TimeoutSec $RequestTimeoutSec -Body ([ordered]@{
            model = $Ctx.Model; input = @((New-Prompt $Ctx ' first'), 'A second, plain sentence.')
        })
    Save-Exchange $Ctx 'embeddings-openai' $openai
    if ($openai.Status -ne 200) {
        $Check.Issues.Add("POST /v1/embeddings: HTTP $($openai.Status) $(Get-ErrorText $openai)")
    } else {
        if (@($openai.Json.data).Count -ne 2) { $Check.Issues.Add("POST /v1/embeddings returned $(@($openai.Json.data).Count) vectors for 2 inputs") }
        $usage = [pscustomobject]@{ Input = [int]$openai.Json.usage.prompt_tokens; Output = $null; Cached = $null }
        Test-GenAiRequest -Ctx $Ctx -Response $openai -Check $Check -Operation embeddings -Protocol openai -Usage $usage -Label 'openai' | Out-Null
    }
    $ollama = Invoke-Http -Method POST -Url "$($Ctx.Base)/api/embed" -TimeoutSec $RequestTimeoutSec -Body ([ordered]@{
            model = $Ctx.Model; input = (New-Prompt $Ctx ' third')
        })
    Save-Exchange $Ctx 'embeddings-ollama' $ollama
    if ($ollama.Status -ne 200) {
        $Check.Issues.Add("POST /api/embed: HTTP $($ollama.Status) $(Get-ErrorText $ollama)")
    } else {
        $usage = [pscustomobject]@{ Input = [int]$ollama.Json.prompt_eval_count; Output = $null; Cached = $null }
        Test-GenAiRequest -Ctx $Ctx -Response $ollama -Check $Check -Operation embeddings -Protocol ollama -Usage $usage -Label 'ollama' | Out-Null
    }
}

# Text-to-video through the OpenAI-shaped route (POST /v1/videos/generations; it blocks until the
# MP4 is written and returns { data: [{ url }], frames, width, height, ... }).
function Invoke-VideoGeneration($Ctx, [string] $Size, [int] $Frames, [string] $Label) {
    $body = [ordered]@{ prompt = (New-Prompt $Ctx -Video); size = $Size; frames = $Frames; seed = 42; response_format = 'url' }
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/v1/videos/generations" -Body $body -TimeoutSec $VideoTimeoutSec
    Save-Exchange $Ctx $Label $response
    return $response
}

function Test-VideoLtx($Ctx, $Check) {
    $response = Invoke-VideoGeneration $Ctx $LtxSize $LtxFrames 'video-ltx'
    if ($response.Status -ne 200) { $Check.Issues.Add("HTTP $($response.Status): $(Get-ErrorText $response)"); return }
    if (-not @($response.Json.data)[0].url) { $Check.Issues.Add('the response has no data[0].url') }
    $Check.Notes.Add("$($response.Json.frames) frames $($response.Json.width)x$($response.Json.height) in $($response.Json.elapsed_seconds) s")
    Test-VideoRequest -Ctx $Ctx -Response $response -Check $Check -Label 'clip' `
        -Phases @('text-encode', 'denoise', 'vae-decode') -Stages @{ 'denoise' = $LtxExpectStages } | Out-Null
}

# Two clips in a row: the first loads the DiT (dit-load cached=false), the second finds it loaded.
function Test-VideoMiniMaxH3($Ctx, $Check) {
    $clips = @($null, $null)
    for ($n = 1; $n -le 2; $n++) {
        $response = Invoke-VideoGeneration $Ctx $MiniMaxH3Size $MiniMaxH3Frames "video-minimax-h3-clip$n"
        if ($response.Status -ne 200) { $Check.Issues.Add("clip ${n}: HTTP $($response.Status) $(Get-ErrorText $response)"); return }
        $Check.Notes.Add("clip ${n}: $($response.Json.frames) frames $($response.Json.width)x$($response.Json.height) in $($response.Json.elapsed_seconds) s")
        $clips[$n - 1] = Test-VideoRequest -Ctx $Ctx -Response $response -Check $Check -Label "clip $n" -Phases @('text-encode', 'denoise', 'vae-decode')
    }
    if (-not $script:Api.Usable -or -not $clips[0] -or -not $clips[1]) { return }
    $dit1 = @($clips[0].Descendants | Where-Object { $_.Name -eq 'dit-load' })[0]
    $dit2 = @($clips[1].Descendants | Where-Object { $_.Name -eq 'dit-load' })[0]
    $cached1 = if ($dit1) { [string]$dit1.Attr['tensorsharp.video.cached'] } else { '' }
    $cached2 = if ($dit2) { [string]$dit2.Attr['tensorsharp.video.cached'] } else { '' }
    if (-not $dit1) { $Check.Issues.Add('clip 1 has no dit-load stage span') }
    elseif ($cached1 -ne 'false') { $Check.Issues.Add("clip 1 dit-load has tensorsharp.video.cached='$cached1', expected false") }
    $released = [bool]($clips[0].Descendants | Where-Object { $_.Name -eq 'denoiser-release' })
    if (-not $dit2) {
        $Check.Issues.Add('clip 2 has no dit-load stage span')
    } elseif ($cached2 -ne 'true') {
        if ($released) {
            $Check.Manual = "clip 2 reloaded the DiT (cached=$cached2) because clip 1 released it (denoiser-release span), so the cached path was not exercised on this host"
        } else {
            $Check.Issues.Add("clip 2 dit-load has tensorsharp.video.cached='$cached2', expected true")
        }
    } else {
        $Check.Notes.Add('dit-load cached: clip 1 false, clip 2 true')
    }
}

# Drop the progress stream of /api/video-generate/stream (the route that honours cancellation).
function Test-VideoCancel($Ctx, $Check) {
    $endedBefore = Get-RequestEndCount $Ctx 'POST' '/api/video-generate/stream'
    $size = ConvertTo-Size $Ctx.Run.VideoSize
    $body = [ordered]@{ prompt = (New-Prompt $Ctx -Video); width = $size.Width; height = $size.Height; frames = $Ctx.Run.VideoFrames; seed = 7 }
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/api/video-generate/stream" -Body $body -TimeoutSec $VideoTimeoutSec -Stream `
        -AbortAfterMatches 1 -AbortPattern '"phase"\s*:\s*"denoise"' -AbortAfterSec $VideoCancelAfterSec
    Save-Exchange $Ctx $Ctx.CurrentScenario $response
    if ($response.Status -ne 200) { $Check.Issues.Add("HTTP $($response.Status): $(Get-ErrorText $response)"); return }
    if (-not $response.Aborted) {
        $done = @(ConvertFrom-Sse $response.Lines | Where-Object { $_.done -eq $true })[0]
        $Check.Issues.Add("the generation ended before it could be cancelled ($(if ($done.error) { "error: $($done.error)" } else { 'completed' }))")
        return
    }
    $Check.Notes.Add("dropped the stream after $($response.Lines.Count) lines ($([math]::Round($response.ElapsedMs / 1000, 1)) s)")
    Test-VideoRequest -Ctx $Ctx -Response $response -Check $Check -Label 'cancelled clip' -Outcome cancelled | Out-Null
    # The clip runs on until it next checks its token (LTX: at its next progress report or sampling
    # step, and the denoiser upload before the first step can take tens of seconds). The next
    # scenario would compete with it for the GPU, memory and disk (Studio's ten-second ffprobe probe
    # timed out that way), so wait for its request to end first.
    if (-not (Wait-RequestEnded $Ctx 'POST' '/api/video-generate/stream' $endedBefore 600)) {
        $Check.Issues.Add('the cancelled clip''s request had not ended 600 s after the client went away')
    }
    $health = Invoke-Http -Url "$($Ctx.Base)/health" -TimeoutSec 30
    if ($health.Status -ne 200) { $Check.Issues.Add("the server stopped answering /health after the cancel (HTTP $($health.Status))") }
}

function Invoke-Studio($Ctx, [string] $Action, $Payload) {
    Invoke-Http -Method POST -Url "$($Ctx.Base)/api/studio" -Headers @{ 'X-TensorSharp-Studio' = '1' } -TimeoutSec 120 `
        -Body ([ordered]@{ action = $Action; payload = $Payload })
}
# Studio's capabilities. A probe of FFmpeg or ffprobe that ran out of time is not cached by
# Studio, and on a busy machine (a cancelled clip's read-ahead still filling memory and the disk)
# a ten-second limit can pass before the executable has even loaded, so such an answer is asked
# again a few times before the scenario gives up on Studio.
function Get-StudioCapabilities($Ctx, $Check) {
    for ($attempt = 1; ; $attempt++) {
        $capabilities = Invoke-Studio $Ctx 'capabilities' @{}
        $text = if ($capabilities.Json) { $capabilities.Json | ConvertTo-Json -Depth 6 -Compress } else { '' }
        if ($capabilities.Status -ne 200 -or $attempt -ge 4 -or $text -notmatch 'exceeded its time limit') { break }
        $Check.Notes.Add("Studio's media probe ran out of time (attempt $attempt); asking again in 15 s")
        Start-Sleep -Seconds 15
    }
    return $capabilities
}

# Studio jobs: a trace of their own, linked to the request that queued them.
function Test-StudioGenerate($Ctx, $Check) {
    $capabilities = Get-StudioCapabilities $Ctx $Check
    Save-Exchange $Ctx 'studio-capabilities' $capabilities
    if ($capabilities.Status -ne 200) { $Check.Skip = "the Video Studio API is not reachable: HTTP $($capabilities.Status) $(Get-ErrorText $capabilities)"; return }
    $models = @($capabilities.Json.models | Where-Object { $_ })
    $model = @($models | Where-Object { $_.available -and $_.textToVideo })[0]
    if (-not $model) {
        $why = @(@($capabilities.Json.generationUnavailableReason) + @($models | ForEach-Object { "$($_.modelId): $($_.unavailableReason)" }) | Where-Object { $_ })
        $Check.Skip = "Video Studio has no available text-to-video model ($($why -join '; '))"
        return
    }
    # Fit the run's clip size to the model's grid; 0 asks for the model's default.
    $size = ConvertTo-Size $Ctx.Run.VideoSize
    $multiple = [math]::Max(1, [int]$model.spatialMultiple)
    $width = [int]([math]::Floor($size.Width / $multiple) * $multiple)
    $height = [int]([math]::Floor($size.Height / $multiple) * $multiple)
    if ($width -lt 16 -or $height -lt 16 -or $width -gt [int]$model.maxDimension -or $height -gt [int]$model.maxDimension) { $width = 0; $height = 0 }
    $frames = [int]$Ctx.Run.VideoFrames
    $temporal = [int]$model.temporalMultiple
    $offset = [int]$model.temporalOffset
    if ($temporal -le 0 -or $frames -lt $offset -or $frames -gt [int]$model.maxFrames -or (($frames - $offset) % $temporal) -ne 0) { $frames = 0 }

    $project = Invoke-Studio $Ctx 'projects/create' ([ordered]@{ name = "otel-e2e $script:RunTag" })
    Save-Exchange $Ctx 'studio-project' $project
    if ($project.Status -ne 200 -or -not $project.Json.id) { $Check.Issues.Add("projects/create: HTTP $($project.Status) $(Get-ErrorText $project)"); return }
    $projectId = [string]$project.Json.id
    $generate = Invoke-Studio $Ctx 'jobs/generate' ([ordered]@{
            projectId = $projectId; requestId = (New-RandomHex 16); modelId = [string]$model.modelId
            prompt = (New-Prompt $Ctx -Video); width = $width; height = $height; frameCount = $frames
        })
    Save-Exchange $Ctx 'studio-generate' $generate
    if ($generate.Status -ne 200 -or -not $generate.Json.id) { $Check.Issues.Add("jobs/generate: HTTP $($generate.Status) $(Get-ErrorText $generate)"); return }
    $jobId = [string]$generate.Json.id
    $Check.Notes.Add("job $jobId on $($model.modelId) ($width x $height, $frames frames; 0 = model default)")

    $job = Wait-Until -TimeoutSec $VideoTimeoutSec -IntervalSec 5 -Probe {
        $list = Invoke-Studio $Ctx 'jobs/list' ([ordered]@{ projectId = $projectId })
        $item = @($list.Json | Where-Object { $_.id -eq $jobId })[0]
        if ($item) { [pscustomobject]@{ Done = ($item.state -notin 'queued', 'running', 'cancelling'); Job = $item } }
    }
    if (-not $job -or -not $job.Done) { $Check.Issues.Add("the job did not finish within $VideoTimeoutSec s (state $($job.Job.state))"); return }
    if ($job.Job.state -ne 'completed') { $Check.Issues.Add("the job ended '$($job.Job.state)': $($job.Job.error)") }
    if (-not (Test-ApiUsable $Check "the 'studio.job generate' trace of job $jobId")) { return }

    $state = Wait-Until -TimeoutSec $VideoTelemetryTimeoutSec -Probe {
        $jobSpan = @(Get-ServiceSpans $Ctx.Service | Where-Object { $_.Name -like 'studio.job*' -and $_.Attr['tensorsharp.studio.job_id'] -eq $jobId })[0]
        if ($jobSpan) {
            $trace = @(Get-TraceSpans $jobSpan.TraceId)
            [pscustomobject]@{ Done = $true; JobSpan = $jobSpan; Trace = $trace }
        }
    }
    if (-not $state) { $Check.Issues.Add("no studio.job span with tensorsharp.studio.job_id=$jobId reached the dashboard$(Get-ApiDiagnostics)"); return }
    $jobSpan = $state.JobSpan
    Save-TraceEvidence $Ctx $jobSpan.TraceId 'studio-job'
    $index = Get-SpanIndex $state.Trace
    if ($jobSpan.Name -ne 'studio.job generate') { $Check.Issues.Add("the job span is named '$($jobSpan.Name)', expected 'studio.job generate'") }
    if ($jobSpan.ParentSpanId) { $Check.Issues.Add("the job span has a parent ($($jobSpan.ParentSpanId)); it should start its own trace") }
    if ($jobSpan.Attr['tensorsharp.outcome'] -ne $job.Job.state) { $Check.Issues.Add("job span tensorsharp.outcome is '$($jobSpan.Attr['tensorsharp.outcome'])', the job is '$($job.Job.state)'") }
    $children = @(Get-ChildSpans $index $jobSpan | ForEach-Object Name)
    if ($children -notcontains 'studio.queue') { $Check.Issues.Add("the job span has no studio.queue child (children: $($children -join ', '))") }
    $stageSpans = @($children | Where-Object { $_ -like 'studio.stage *' })
    if ($stageSpans.Count -eq 0) { $Check.Issues.Add('the job span has no studio.stage children') }
    $descendants = @(Get-DescendantSpans $index $jobSpan)
    if ($job.Job.state -eq 'completed' -and -not ($descendants | Where-Object { $_.Name -like 'generate_video *' })) {
        $Check.Issues.Add('no generate_video span under the job span')
    }
    $Check.Notes.Add("job span children: $($children -join ', ')")
    Test-Containment $index $jobSpan $Check 'job'

    $queuing = Find-RequestTelemetry -Ctx $Ctx -RequestId $generate.RequestId
    if (-not $queuing) {
        $Check.Issues.Add("the queuing request's server span (request id $($generate.RequestId)) did not reach the dashboard")
    } else {
        if ($queuing.Server.TraceId -eq $jobSpan.TraceId) { $Check.Issues.Add('the job span is in the queuing request''s trace instead of its own') }
        if (-not ($jobSpan.Links | Where-Object { $_.TraceId -eq $queuing.Server.TraceId })) {
            $Check.Issues.Add("the job span has no link to the queuing request's trace $($queuing.Server.TraceId) (links: $((@($jobSpan.Links | ForEach-Object TraceId)) -join ', '))")
        }
    }
}

# Engine step spans (TENSORSHARP_OTEL_TRACE_STEPS=1): roots of their own, linked to the request.
function Test-EngineStepSpans($Ctx, $Check) {
    if (-not (Test-RequireModel $Ctx $Check)) { return }
    $chat = Invoke-OpenAIChat -Ctx $Ctx
    Save-Exchange $Ctx $Ctx.CurrentScenario $chat.Response
    if ($chat.Response.Status -ne 200) { $Check.Issues.Add("HTTP $($chat.Response.Status): $(Get-ErrorText $chat.Response)"); return }
    $found = Test-GenAiRequest -Ctx $Ctx -Response $chat.Response -Check $Check -Operation chat -Protocol openai -Usage $chat.Usage
    if (-not $found) { return }
    $traceId = $found.Server.TraceId
    $steps = Wait-Until -TimeoutSec $TelemetryTimeoutSec -Probe {
        $linked = @(Get-ServiceSpans $Ctx.Service | Where-Object { $_.Name -eq 'inference.step' -and ($_.Links | Where-Object { $_.TraceId -eq $traceId }) })
        if ($linked.Count -gt 0) { [pscustomobject]@{ Done = $true; Steps = $linked } }
    }
    if (-not $steps) { $Check.Issues.Add("no inference.step span links to the request's trace $traceId"); return }
    $nested = @($steps.Steps | Where-Object { $_.ParentSpanId })
    if ($nested.Count -gt 0) { $Check.Issues.Add("$($nested.Count) inference.step span(s) have a parent; they should be roots") }
    $noBatch = @($steps.Steps | Where-Object { $null -eq (Get-AttrNumber $_.Attr 'tensorsharp.engine.batch_size') })
    if ($noBatch.Count -gt 0) { $Check.Issues.Add("$($noBatch.Count) inference.step span(s) have no tensorsharp.engine.batch_size") }
    $Check.Notes.Add("$($steps.Steps.Count) inference.step spans link to the request")
}

# With OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true the prompt is exported.
function Test-ContentCapture($Ctx, $Check) {
    if (-not (Test-RequireModel $Ctx $Check)) { return }
    $chat = Invoke-OpenAIChat -Ctx $Ctx
    Save-Exchange $Ctx $Ctx.CurrentScenario $chat.Response
    if ($chat.Response.Status -ne 200) { $Check.Issues.Add("HTTP $($chat.Response.Status): $(Get-ErrorText $chat.Response)"); return }
    if (-not (Test-ApiUsable $Check "that the prompt text '$script:Marker' appears in the exported logs")) { return }
    $found = Wait-Until -TimeoutSec $TelemetryTimeoutSec -Probe {
        $logs = Invoke-DashboardApi '/api/telemetry/logs' @{ resource = $Ctx.Service; limit = $AllLimit }
        if ($logs.Body -and $logs.Body.Contains($script:Marker)) { [pscustomobject]@{ Done = $true } }
    }
    if (-not $found) { $Check.Issues.Add("the prompt marker $script:Marker never appeared in the exported logs with content capture on") }
    else { $Check.Notes.Add('the prompt appears in exported logs') }
    $spans = Invoke-DashboardApi '/api/telemetry/spans' @{ resource = $Ctx.Service; limit = $AllLimit }
    if ($spans.Body -and $spans.Body.Contains($script:Marker)) { $Check.Notes.Add('the prompt appears on spans (gen_ai.input.messages)') }
    else { $Check.Notes.Add('no span attribute carries the prompt') }
}

# Nothing from this server may reach the dashboard. With no service name set (negative-off) the
# server is recognised by its service.instance.id (<machine>:<pid>).
function Test-NothingExported($Ctx, $Check) {
    foreach ($i in 1..3) {
        $response = Invoke-Http -Url "$($Ctx.Base)/api/models" -TimeoutSec 60
        if ($response.Status -ne 200) { $Check.Issues.Add("GET /api/models answered HTTP $($response.Status) $(Get-ErrorText $response)") }
    }
    if (-not (Test-ApiUsable $Check "that no resource $($Ctx.Service) / instance $($Ctx.InstanceId) appears")) { return }
    Start-Sleep -Seconds 12   # longer than the SDK's default 5 s batch delay, had export been on
    $resources = @((Invoke-DashboardApi '/api/telemetry/resources').Json | Where-Object { $_ })
    $spans = @(ConvertFrom-OtlpSpans (Invoke-DashboardApi '/api/telemetry/spans' @{ limit = $AllLimit }).Json)
    $logs = @(ConvertFrom-OtlpLogs (Invoke-DashboardApi '/api/telemetry/logs' @{ limit = $AllLimit }).Json)
    $mineResources = @($resources | Where-Object { $_.instanceId -eq $Ctx.InstanceId -or ($Ctx.Run.Mode -ne 'off' -and $_.name -eq $Ctx.Service) })
    $mineSpans = @($spans | Where-Object { $_.Resource['service.instance.id'] -eq $Ctx.InstanceId -or ($Ctx.Run.Mode -ne 'off' -and $_.Resource['service.name'] -eq $Ctx.Service) })
    $mineLogs = @($logs | Where-Object { $_.Resource['service.instance.id'] -eq $Ctx.InstanceId -or ($Ctx.Run.Mode -ne 'off' -and $_.Resource['service.name'] -eq $Ctx.Service) })
    if ($mineResources.Count + $mineSpans.Count + $mineLogs.Count -gt 0) {
        $Check.Issues.Add("the dashboard received telemetry from this server: $($mineResources.Count) resource(s), $($mineSpans.Count) span(s), $($mineLogs.Count) log record(s)")
    }
    $Check.Notes.Add("checked $($resources.Count) resources, $($spans.Count) spans, $($logs.Count) log records for $($Ctx.InstanceId)")
}

function Test-WrongKey($Ctx, $Check) {
    $latency = $null
    try { $latency = Measure-Latency $Ctx } catch { $Check.Issues.Add("the server did not keep serving: $($_.Exception.Message)") }
    if ($latency) {
        $script:RunInfo[$Ctx.Name]['latency'] = $latency
        $baseline = $script:LatencyBaseline
        if (-not $baseline) {
            $Check.Manual = 'latency not compared: no OTLP run measured a baseline'
        } elseif ($baseline.Kind -ne $latency.Kind) {
            $Check.Manual = "latency not compared: the baseline measured '$($baseline.Kind)' requests, this run '$($latency.Kind)'"
        } else {
            $limit = $baseline.MedianMs * (1 + $LatencyMarginPercent / 100.0) + $LatencyMarginMs
            $text = "median $([math]::Round($latency.MedianMs, 1)) ms vs baseline $([math]::Round($baseline.MedianMs, 1)) ms from $($baseline.Run) (limit $([math]::Round($limit, 1)) ms)"
            if ($latency.MedianMs -gt $limit) { $Check.Issues.Add("latency went up while export fails: $text") } else { $Check.Notes.Add($text) }
        }
    }
    Test-NothingExported $Ctx $Check
}

# ---------------------------------------------------------------------------------------------
# Checks every server run gets
# ---------------------------------------------------------------------------------------------

function Test-StartupLine($Ctx, $Check) {
    $state = Wait-Until -TimeoutSec 15 -IntervalSec 1 -Probe {
        $line = Get-TelemetryLine $Ctx.Server
        if ($line) { [pscustomobject]@{ Done = $true; Line = $line } }
    }
    if (-not $state) { $Check.Issues.Add("no 'Telemetry: ' startup line in the server output ($($Ctx.Server.StdoutPath))"); return }
    $line = $state.Line
    $script:RunInfo[$Ctx.Name]['telemetryLine'] = $line
    $Check.Notes.Add($line)
    switch ($Ctx.Run.Mode) {
        'off' {
            if ($line -notmatch '^Telemetry: off \(') { $Check.Issues.Add('expected "Telemetry: off (...)" with no telemetry variables set') }
        }
        'sdk-disabled' {
            if (-not $line.StartsWith('Telemetry: off (OTEL_SDK_DISABLED=true)')) { $Check.Issues.Add('expected "Telemetry: off (OTEL_SDK_DISABLED=true)"') }
        }
        default {
            foreach ($signal in 'traces', 'metrics', 'logs') {
                if ($line -notmatch "$signal\([^)]*otlp") { $Check.Issues.Add("$signal are not exported over OTLP") }
            }
            $expected = "OTLP $($Ctx.Endpoint) ($($Ctx.Run.Protocol))"
            if (-not $line.Contains($expected)) { $Check.Issues.Add("the line does not name '$expected'") }
            if (-not $line.Contains("service $($Ctx.Service)")) { $Check.Issues.Add("the line does not name service $($Ctx.Service)") }
            $capture = if ($Ctx.Run.Capture) { 'content capture ON' } else { 'content capture off' }
            if (-not $line.Contains($capture)) { $Check.Issues.Add("expected '$capture'") }
            if ($Ctx.Run.TraceSteps -and -not $line.Contains('engine step spans ON')) { $Check.Issues.Add("expected 'engine step spans ON'") }
        }
    }
}

function Test-ResourceAttributes($Ctx, $Check, $Smoke) {
    if (-not (Test-ApiUsable $Check "the resource attributes of $($Ctx.Service)")) { return }
    $found = Find-RequestTelemetry -Ctx $Ctx -RequestId $Smoke.RequestId
    if (-not $found) { $Check.Issues.Add("the span of GET /api/models (request id $($Smoke.RequestId)) never reached the dashboard$(Get-ApiDiagnostics)"); return }
    $resource = $found.Server.Resource
    $expected = [ordered]@{
        'service.name'        = $Ctx.Service
        'service.instance.id' = $Ctx.InstanceId
        'host.name'           = $script:Machine
        'process.pid'         = [string]$Ctx.Server.Pid
    }
    foreach ($key in $expected.Keys) {
        if ([string]$resource[$key] -ne $expected[$key]) { $Check.Issues.Add("resource $key is '$($resource[$key])', expected '$($expected[$key])'") }
    }
    $version = [string]$resource['service.version']
    if (-not $version) { $Check.Issues.Add('the resource has no service.version') } else { $script:RunInfo[$Ctx.Name]['serviceVersion'] = $version }
    $Check.Notes.Add("service.version $version; telemetry.sdk $($resource['telemetry.sdk.name']) $($resource['telemetry.sdk.version'])")
    $listed = @((Invoke-DashboardApi '/api/telemetry/resources').Json | Where-Object { $_ -and $_.name -eq $Ctx.Service })
    if ($listed.Count -eq 0) { $Check.Issues.Add("/api/telemetry/resources does not list $($Ctx.Service)") }
}

# The prompt marker must not be in any exported log record or span of a run without content capture.
function Test-ContentRedaction($Ctx, $Check, [string] $SpansRaw, [string] $LogsRaw, $Spans, $Logs) {
    if (-not $Ctx.MarkerSent) { $Check.Skip = 'no request in this run carried prompt text'; return }
    if (-not (Test-ApiUsable $Check "that no exported log or span contains '$script:Marker'")) { return }
    if ($LogsRaw -and $LogsRaw.Contains($script:Marker)) {
        $hits = foreach ($log in $Logs) {
            if ($log.Body.Contains($script:Marker)) { "$($log.Scope) body" }
            foreach ($key in $log.Attr.Keys) { if (([string]$log.Attr[$key]).Contains($script:Marker)) { "$($log.Scope) attribute $key" } }
        }
        $Check.Issues.Add("the prompt marker is in exported logs: $((@($hits | Select-Object -Unique -First 6)) -join '; ')")
    }
    if ($SpansRaw -and $SpansRaw.Contains($script:Marker)) {
        $hits = foreach ($span in $Spans) {
            if ($span.Name.Contains($script:Marker)) { "span name '$($span.Name)'" }
            foreach ($key in $span.Attr.Keys) { if (([string]$span.Attr[$key]).Contains($script:Marker)) { "'$($span.Name)' attribute $key" } }
        }
        $Check.Issues.Add("the prompt marker is on exported spans: $((@($hits | Select-Object -Unique -First 6)) -join '; ')")
    }
    if (@($Logs).Count -eq 0) { $Check.Issues.Add('no log records were exported, so redaction was not exercised') }
    $Check.Notes.Add("searched $(@($Logs).Count) log records and $(@($Spans).Count) spans for $script:Marker")
}

function Test-SecretNotLogged($Ctx, $Check) {
    $secrets = @($Ctx.Secrets | Where-Object { $_ })
    if ($secrets.Count -eq 0) { $Check.Skip = 'no OTLP or API key was in use'; return }
    $files = @($Ctx.Server.StdoutPath, $Ctx.Server.StderrPath)
    if ($Ctx.Server.LogDir -and (Test-Path -LiteralPath $Ctx.Server.LogDir)) {
        $files += @(Get-ChildItem -LiteralPath $Ctx.Server.LogDir -File -Recurse -ErrorAction SilentlyContinue | ForEach-Object FullName)
    }
    foreach ($file in $files) {
        $text = Read-SharedText $file
        foreach ($secret in $secrets) {
            if ($text.Contains($secret)) { $Check.Issues.Add("a key the harness passed to the server or dashboard appears in $(Split-Path -Leaf $file)") }
        }
    }
    $Check.Notes.Add("searched $($files.Count) output files")
}

function Test-RunTelemetry($Ctx, $Smoke) {
    $check = New-Check
    Test-ResourceAttributes $Ctx $check $Smoke
    Complete-Check $Ctx.Name 'resource-attributes' $check

    $check = New-Check
    $Ctx.CurrentScenario = 'smoke'
    Test-GenAiRequest -Ctx $Ctx -Response $Smoke -Check $check -Protocol webui -NoGenAiSpan -Label 'GET /api/models' | Out-Null
    Complete-Check $Ctx.Name 'request-span-and-logs' $check

    # Whole-run snapshot, once the last batch has been exported.
    Start-Sleep -Seconds 3
    $spansRaw = ''
    $logsRaw = ''
    $spans = @()
    $logs = @()
    if ($script:Api.Usable) {
        $spansResult = Invoke-DashboardApi '/api/telemetry/spans' @{ resource = $Ctx.Service; limit = $AllLimit }
        $logsResult = Invoke-DashboardApi '/api/telemetry/logs' @{ resource = $Ctx.Service; limit = $AllLimit }
        $spansRaw = [string]$spansResult.Body
        $logsRaw = [string]$logsResult.Body
        Save-Text (Join-Path $Ctx.Dir 'telemetry/spans.json') $spansRaw
        Save-Text (Join-Path $Ctx.Dir 'telemetry/logs.json') $logsRaw
        $spans = @(ConvertFrom-OtlpSpans $spansResult.Json)
        $logs = @(ConvertFrom-OtlpLogs $logsResult.Json)
    }

    $check = New-Check
    if (Test-ApiUsable $check "that no 'GET /health' span exists for $($Ctx.Service)") {
        $health = @($spans | Where-Object { $_.Kind -eq 'SERVER' -and (Test-SpanPath $_ '/health') })
        if ($health.Count -gt 0) { $Check.Issues.Add("$($health.Count) server span(s) for /health were exported") }
        if (-not ($spans | Where-Object { $_.Kind -eq 'SERVER' })) { $check.Issues.Add('no server span at all was exported, so the absence of /health spans proves nothing') }
        $check.Notes.Add("the harness polled /health $($Ctx.Server.HealthPolls) times; $($spans.Count) spans searched")
    }
    Complete-Check $Ctx.Name 'health-not-traced' $check

    if (-not $Ctx.Run.Capture) {
        $check = New-Check
        Test-ContentRedaction $Ctx $check $spansRaw $logsRaw $spans $logs
        Complete-Check $Ctx.Name 'content-redaction' $check
    }
}

# ---------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------

function Invoke-Scenario($Ctx, [string] $Name) {
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $check = New-Check
    $Ctx.CurrentScenario = $Name
    Write-Step "  $($Ctx.Name): $Name"
    try {
        switch ($Name) {
            'llm-openai-stream' { Test-LlmOpenAI $Ctx $check -Stream | Out-Null }
            'llm-openai' { Test-LlmOpenAI $Ctx $check | Out-Null }
            'llm-ollama-generate' { Test-LlmOllamaGenerate $Ctx $check | Out-Null }
            'llm-ollama-chat' { Test-LlmOllamaChat $Ctx $check | Out-Null }
            'llm-webui' { Test-LlmWebUi $Ctx $check | Out-Null }
            'llm-client-traceparent' { Test-LlmClientTraceparent $Ctx $check | Out-Null }
            'llm-cancel' { Test-LlmCancel $Ctx $check | Out-Null }
            'llm-bad-request' { Test-LlmBadRequest $Ctx $check | Out-Null }
            'llm-burst' { Test-LlmBurst $Ctx $check | Out-Null }
            'queue-status-poll' { Test-QueueStatusPoll $Ctx $check | Out-Null }
            'embeddings' { Test-Embeddings $Ctx $check | Out-Null }
            'video-ltx' { Test-VideoLtx $Ctx $check | Out-Null }
            'video-minimax-h3' { Test-VideoMiniMaxH3 $Ctx $check | Out-Null }
            'video-cancel' { Test-VideoCancel $Ctx $check | Out-Null }
            'studio-generate' { Test-StudioGenerate $Ctx $check | Out-Null }
            'engine-step-spans' { Test-EngineStepSpans $Ctx $check | Out-Null }
            'content-capture' { Test-ContentCapture $Ctx $check | Out-Null }
            'negative-off' { Test-NothingExported $Ctx $check | Out-Null }
            'negative-sdk-disabled' { Test-NothingExported $Ctx $check | Out-Null }
            'negative-wrong-key' { Test-WrongKey $Ctx $check | Out-Null }
            default { $check.Issues.Add("no implementation for scenario $Name") }
        }
    } catch {
        $check.Issues.Add("harness error: $($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))")
    }
    Complete-Check $Ctx.Name $Name $check $watch.Elapsed.TotalSeconds
}

function New-Run([string] $Name, [string] $Mode, [string] $Protocol, $ModelArgs, [string[]] $RunScenarios) {
    [pscustomobject]@{
        Name            = $Name
        Mode            = $Mode       # otlp | off | sdk-disabled | wrong-key
        Protocol        = $Protocol
        ModelArgs       = $ModelArgs
        Scenarios       = @($RunScenarios)
        MeasureBaseline = $false
        Capture         = $false
        UseCliFlag      = $false
        TraceSteps      = $false
        VideoSize       = ''
        VideoFrames     = 0
    }
}

# Which server runs to make. A selected scenario whose model or setup is missing is recorded as
# SKIPPED here and left out of every run.
function Get-RunPlan($ChatArgs, $EmbeddingArgs, $LtxArgs, $H3Args) {
    $missing = @{}
    foreach ($s in @($LlmScenarios) + 'content-capture') { if (-not $ChatArgs) { $missing[$s] = 'needs -ChatConfig' } }
    if (-not $EmbeddingArgs) { $missing['embeddings'] = 'needs -EmbeddingConfig' }
    if (-not $LtxArgs) { $missing['video-ltx'] = 'needs -LtxConfig' }
    if (-not $H3Args) { $missing['video-minimax-h3'] = 'needs -MiniMaxH3Config' }
    if (-not $LtxArgs -and -not $H3Args) {
        $missing['video-cancel'] = 'needs -LtxConfig or -MiniMaxH3Config'
        $missing['studio-generate'] = 'needs -LtxConfig or -MiniMaxH3Config'
    }
    if (-not $TraceSteps) { $missing['engine-step-spans'] = 'needs -TraceSteps' }
    elseif (-not $ChatArgs) { $missing['engine-step-spans'] = 'needs -ChatConfig' }
    elseif ($Protocols -notcontains 'grpc') { $missing['engine-step-spans'] = 'needs the grpc run (-Protocols grpc)' }
    if ($Protocols.Count -eq 0) { foreach ($s in @($LlmScenarios) + 'queue-status-poll') { $missing[$s] = 'no -Protocols given' } }
    if ($SkipDocker -and -not $OtlpApiKey) {
        $missing['negative-wrong-key'] = 'the external dashboard is not known to require an OTLP API key; pass -OtlpApiKey'
    }
    foreach ($s in $Selected) { if ($missing.ContainsKey($s)) { Add-Result '-' $s 'SKIPPED' $missing[$s] } }
    $runnable = @($Selected | Where-Object { -not $missing.ContainsKey($_) })

    $plan = [System.Collections.Generic.List[object]]::new()
    $first = $true
    foreach ($protocol in $Protocols) {
        $name = if ($protocol -eq 'grpc') { 'otlp-grpc' } else { 'otlp-http' }
        $runScenarios = @(@($LlmScenarios) + 'queue-status-poll' + 'engine-step-spans' |
                Where-Object { $runnable -contains $_ -and ($_ -ne 'engine-step-spans' -or $protocol -eq 'grpc') })
        $baseline = $first -and ($runnable -contains 'negative-wrong-key')
        if ($runScenarios.Count -eq 0 -and -not $baseline -and $runnable -notcontains 'metrics') { continue }
        $needsChat = [bool]($runScenarios | Where-Object { $_ -like 'llm-*' -or $_ -eq 'engine-step-spans' }) -or ($baseline -and $ChatArgs)
        $run = New-Run $name 'otlp' $protocol $(if ($needsChat) { $ChatArgs } else { $null }) $runScenarios
        $run.MeasureBaseline = $baseline
        $run.TraceSteps = [bool]$TraceSteps -and $protocol -eq 'grpc' -and ($runScenarios -contains 'engine-step-spans')
        $plan.Add($run)
        $first = $false
    }
    if ($runnable -contains 'content-capture') {
        # Also exercises --otel-endpoint in place of OTEL_EXPORTER_OTLP_ENDPOINT.
        $run = New-Run 'content-capture' 'otlp' 'grpc' $ChatArgs @('content-capture')
        $run.Capture = $true
        $run.UseCliFlag = $true
        $plan.Add($run)
    }
    if ($runnable -contains 'embeddings') { $plan.Add((New-Run 'embeddings' 'otlp' 'grpc' $EmbeddingArgs @('embeddings'))) }
    $shared = @('video-cancel', 'studio-generate' | Where-Object { $runnable -contains $_ })
    if ($LtxArgs -and ($runnable -contains 'video-ltx' -or $shared.Count)) {
        $run = New-Run 'video-ltx' 'otlp' 'grpc' $LtxArgs (@('video-ltx' | Where-Object { $runnable -contains $_ }) + $shared)
        $run.VideoSize = $LtxSize
        $run.VideoFrames = $LtxFrames
        $plan.Add($run)
        $shared = @()
    }
    if ($H3Args -and ($runnable -contains 'video-minimax-h3' -or $shared.Count)) {
        $run = New-Run 'video-minimax-h3' 'otlp' 'grpc' $H3Args (@('video-minimax-h3' | Where-Object { $runnable -contains $_ }) + $shared)
        $run.VideoSize = $MiniMaxH3Size
        $run.VideoFrames = $MiniMaxH3Frames
        $plan.Add($run)
    }
    if ($runnable -contains 'negative-off') { $plan.Add((New-Run 'negative-off' 'off' '' $null @('negative-off'))) }
    if ($runnable -contains 'negative-sdk-disabled') { $plan.Add((New-Run 'negative-sdk-disabled' 'sdk-disabled' 'grpc' $null @('negative-sdk-disabled'))) }
    if ($runnable -contains 'negative-wrong-key') { $plan.Add((New-Run 'negative-wrong-key' 'wrong-key' 'grpc' $ChatArgs @('negative-wrong-key'))) }
    return $plan
}

function Get-RunEnvironment($Run, [string] $Service, [string] $Endpoint, [string] $Key, [string] $RunDir) {
    $environment = [ordered]@{
        TENSORSHARP_LOG_DIR = (Join-Path $RunDir 'logs')
        TS_STUDIO_DIRECTORY = (Join-Path $RunDir 'studio-projects')
    }
    if ($Run.Mode -eq 'off') { return $environment }
    if (-not $Run.UseCliFlag) { $environment['OTEL_EXPORTER_OTLP_ENDPOINT'] = $Endpoint }
    $environment['OTEL_EXPORTER_OTLP_PROTOCOL'] = $Run.Protocol
    if ($Key) { $environment['OTEL_EXPORTER_OTLP_HEADERS'] = "x-otlp-api-key=$Key" }
    $environment['OTEL_SERVICE_NAME'] = $Service
    # Short export intervals so the checks do not wait for the SDK defaults (5 s, 5 s, 60 s).
    $environment['OTEL_BSP_SCHEDULE_DELAY'] = '1000'
    $environment['OTEL_BLRP_SCHEDULE_DELAY'] = '1000'
    $environment['OTEL_METRIC_EXPORT_INTERVAL'] = '5000'
    if ($Run.Mode -eq 'sdk-disabled') { $environment['OTEL_SDK_DISABLED'] = 'true' }
    if ($Run.Capture) { $environment['OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'] = 'true' }
    if ($Run.TraceSteps) { $environment['TENSORSHARP_OTEL_TRACE_STEPS'] = '1' }
    return $environment
}

function Invoke-Run($Run) {
    $runDir = Join-Path $OutDir "runs/$($Run.Name)"
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    $service = if ($Run.Mode -eq 'off') { 'tensorsharp-server' } else { "ts-e2e-$script:RunTag-$($Run.Name)" }
    $endpoint = if ($Run.Protocol -eq 'http/protobuf') { $script:Dashboard.Http } else { $script:Dashboard.Grpc }
    $wrongKey = if ($Run.Mode -eq 'wrong-key') { 'wrong' + (New-RandomHex 16) } else { '' }
    $otlpKey = if ($Run.Mode -eq 'wrong-key') { $wrongKey } else { $script:Dashboard.OtlpKey }
    $environment = Get-RunEnvironment $Run $service $endpoint $otlpKey $runDir

    $serverArgs = @(@($Run.ModelArgs) + @($script:ExtraArgs) | Where-Object { $null -ne $_ -and "$_" -ne '' })
    $serverArgs += @('--host', '127.0.0.1', '--port', "$Port")
    if ($Run.UseCliFlag) { $serverArgs += @('--otel-endpoint', $endpoint) }

    $shownEnvironment = [ordered]@{}
    foreach ($key in $environment.Keys) {
        $shownEnvironment[$key] = if ($key -eq 'OTEL_EXPORTER_OTLP_HEADERS') { 'x-otlp-api-key=<redacted>' } else { $environment[$key] }
    }
    Save-Json (Join-Path $runDir 'server-launch.json') ([ordered]@{
            file = $script:ServerHostInfo.FileName; arguments = @($script:ServerHostInfo.Prefix) + $serverArgs
            environment = $shownEnvironment; note = 'Inherited OTEL_*, TENSORSHARP_OTEL*, APPLICATIONINSIGHTS_CONNECTION_STRING, PORT, HOST and ASPNETCORE_URLS were removed.'
        })
    $script:RunInfo[$Run.Name] = [ordered]@{
        mode = $Run.Mode; protocol = $Run.Protocol; serviceName = $service; arguments = $serverArgs; scenarios = $Run.Scenarios
    }

    Write-Step "Run $($Run.Name): starting TensorSharp.Server.Host ($($Run.Mode)$(if ($Run.Protocol) { ", $($Run.Protocol)" }))"
    $server = $null
    try {
        # Something else answering on the port would be mistaken for this server.
        Wait-PortFree $Port
        if (Test-TcpPort '127.0.0.1' $Port) { throw "port $Port is already in use; stop what listens there or pass -Port" }
        $server = Start-TsServer -Dir $runDir -ServerArgs $serverArgs -Environment $environment
        $script:CurrentServer = $server
        $script:RunInfo[$Run.Name]['pid'] = $server.Pid
        Wait-ServerReady $server
    } catch {
        Add-Result $Run.Name 'server-start' 'FAIL' "the server did not start: $($_.Exception.Message)"
        foreach ($s in $Run.Scenarios) { Add-Result $Run.Name $s 'FAIL' 'not run: the server did not start' }
        if ($server) { Stop-TsServer $server }
        $script:CurrentServer = $null
        return
    }

    $ctx = [pscustomobject]@{
        Name            = $Run.Name
        Run             = $Run
        Base            = $server.Base
        Service         = $service
        Endpoint        = $endpoint
        Server          = $server
        Dir             = $runDir
        InstanceId      = "$($script:Machine):$($server.Pid)"
        Model           = $null
        MarkerSent      = $false
        ExchangeCount   = 0
        CurrentScenario = ''
        Secrets         = @($script:Dashboard.OtlpKey, $script:Dashboard.ApiKey, $wrongKey)
    }
    try {
        $check = New-Check
        Test-StartupLine $ctx $check
        Complete-Check $Run.Name 'startup-line' $check

        $smoke = Invoke-Http -Url "$($ctx.Base)/api/models" -TimeoutSec 60
        Save-Exchange $ctx 'api-models' $smoke
        if ($smoke.Json) {
            $script:RunInfo[$Run.Name]['loadedModel'] = $smoke.Json.loaded
            $script:RunInfo[$Run.Name]['loadedBackend'] = $smoke.Json.loadedBackend
            $script:RunInfo[$Run.Name]['architecture'] = $smoke.Json.architecture
        }
        $ctx.Model = Get-ModelId $ctx
        $script:RunInfo[$Run.Name]['modelId'] = $ctx.Model

        if ($Run.MeasureBaseline) {
            try {
                $script:LatencyBaseline = Measure-Latency $ctx
                $script:RunInfo[$Run.Name]['latencyBaseline'] = $script:LatencyBaseline
            } catch {
                Write-Warning "Latency baseline: $($_.Exception.Message)"
            }
        }

        foreach ($scenario in $Run.Scenarios) { Invoke-Scenario $ctx $scenario }

        if ($Run.Mode -eq 'otlp') { Test-RunTelemetry $ctx $smoke }
    } finally {
        Stop-TsServer $server
        $script:CurrentServer = $null
        $script:RunInfo[$Run.Name]['exitCode'] = $server.ExitCode
    }
    if ($Run.Mode -ne 'off') {
        $check = New-Check
        Test-SecretNotLogged $ctx $check
        Complete-Check $Run.Name 'otlp-key-not-logged' $check
    }
}

function Stop-TsServer($Server) {
    if ($null -eq $Server) { return }
    Stop-LoggedProcess $Server
    Wait-PortFree $Port
}

function Add-MetricsManual {
    $services = @($script:RunInfo.Values | Where-Object { $_.mode -eq 'otlp' -and $_.pid } | ForEach-Object { $_.serviceName })
    if ($services.Count -eq 0) { Add-Result '-' 'metrics' 'SKIPPED' 'no run exported telemetry'; return }
    $instruments = 'gen_ai.server.request.duration, gen_ai.server.time_to_first_token, gen_ai.server.time_per_output_token, ' +
    'tensorsharp.tokens (tensorsharp.token.type input/output), tensorsharp.tokens.cache_read, tensorsharp.engine.requests.running/.waiting, ' +
    'tensorsharp.engine.steps, tensorsharp.engine.forward.time, tensorsharp.engine.step.duration, tensorsharp.engine.batch.size, ' +
    'tensorsharp.kv_cache.blocks.used/.total, tensorsharp.model.load.duration; video runs: tensorsharp.video.generation.duration, ' +
    'tensorsharp.video.phase.duration, tensorsharp.video.denoise.step.duration, tensorsharp.video.dit.loads, tensorsharp.media.generations.active, ' +
    'tensorsharp.gpu.memory.used/.total/.peak; Studio: tensorsharp.studio.job.duration, tensorsharp.studio.jobs.queued/.running; ' +
    'and the System.Runtime, System.Net.Http and Microsoft.AspNetCore.* meters'
    Add-Result '-' 'metrics' 'MANUAL' ("the dashboard telemetry API has no metrics route (v13.6.2 serves resources, spans, logs and traces). " +
        "Rerun with -KeepDashboard, $($script:Api.UiHint), open Metrics, pick $($services -join ', ') and confirm the TensorSharp and TensorSharp.Studio meters show: $instruments")
}

# ---------------------------------------------------------------------------------------------
# Environment and result files
# ---------------------------------------------------------------------------------------------

function Get-GitInfo([string] $Path) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { return [ordered]@{ note = 'git not found' } }
    $commit = & git -C $Path rev-parse HEAD 2>$null
    if ($LASTEXITCODE -ne 0) { return [ordered]@{ note = "not a git checkout: $Path" } }
    $branch = & git -C $Path rev-parse --abbrev-ref HEAD 2>$null
    $describe = & git -C $Path describe --tags --always 2>$null
    $changes = @(& git --no-optional-locks -C $Path status --porcelain 2>$null)
    [ordered]@{ commit = "$commit".Trim(); branch = "$branch".Trim(); describe = "$describe".Trim(); uncommittedChanges = $changes.Count }
}

function Get-GgmlInfo {
    $path = Join-Path $RepoRoot 'ExternalProjects/ggml'
    if (-not (Test-Path -LiteralPath $path)) {
        return [ordered]@{ path = $path; revision = $null; note = 'ExternalProjects/ggml is not present in this checkout; the native library may come from another checkout or a package' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $path '.git'))) {
        return [ordered]@{ path = $path; revision = $null; note = 'source copy without git metadata' }
    }
    $info = Get-GitInfo $path
    $info['path'] = $path
    return $info
}

function Get-GpuInfo {
    $smi = Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($smi) {
        $lines = @(& $smi.Source '--query-gpu=index,name,driver_version,memory.total' '--format=csv,noheader' 2>$null)
        if ($LASTEXITCODE -eq 0) { return [ordered]@{ source = 'nvidia-smi'; devices = @($lines | ForEach-Object { "$_".Trim() } | Where-Object { $_ }) } }
    }
    if ($IsWindows) {
        try { return [ordered]@{ source = 'Win32_VideoController'; devices = @(Get-CimInstance Win32_VideoController | ForEach-Object { "$($_.Name) (driver $($_.DriverVersion))" }) } } catch { }
    }
    if ($IsMacOS) {
        try {
            $lines = @(& system_profiler SPDisplaysDataType 2>$null | Select-String 'Chipset Model|Metal' | ForEach-Object { $_.Line.Trim() })
            return [ordered]@{ source = 'system_profiler'; devices = $lines }
        } catch { }
    }
    return [ordered]@{ source = 'none'; devices = @(); note = 'nvidia-smi not found' }
}

function Get-DotnetInfo {
    $info = [ordered]@{ powershell = $PSVersionTable.PSVersion.ToString(); scriptRuntime = [Runtime.InteropServices.RuntimeInformation]::FrameworkDescription }
    if (Get-Command dotnet -ErrorAction SilentlyContinue) {
        $info['sdk'] = "$(& dotnet --version 2>$null)".Trim()
        $info['runtimes'] = @(& dotnet --list-runtimes 2>$null | ForEach-Object { "$_" })
    }
    if ($script:ServerHostInfo) {
        $config = Join-Path $script:ServerHostInfo.Directory 'TensorSharp.Server.Host.runtimeconfig.json'
        if (Test-Path -LiteralPath $config) {
            $runtimeConfig = ConvertFrom-JsonText (Get-Content -LiteralPath $config -Raw)
            $frameworks = @($runtimeConfig.runtimeOptions.framework) + @($runtimeConfig.runtimeOptions.frameworks) | Where-Object { $_ }
            $info['serverFrameworks'] = @($frameworks | ForEach-Object { "$($_.name) $($_.version)" })
        }
    }
    return $info
}

function Get-EnvironmentInfo {
    $serverHost = $null
    if ($script:ServerHostInfo) {
        $target = $script:ServerHostInfo.Path
        $item = Get-Item -LiteralPath $target -ErrorAction SilentlyContinue
        $serverHost = [ordered]@{
            path           = $target
            fileVersion    = $item.VersionInfo.FileVersion
            productVersion = $item.VersionInfo.ProductVersion
            lastWriteUtc   = if ($item) { $item.LastWriteTimeUtc.ToString('o') } else { $null }
        }
    }
    $dash = $script:Dashboard
    [ordered]@{
        generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
        script         = 'eng/telemetry/aspire-e2e.ps1'
        parameters     = [ordered]@{
            chatConfig = $ChatConfig; embeddingConfig = $EmbeddingConfig; ltxConfig = $LtxConfig; miniMaxH3Config = $MiniMaxH3Config
            scenarios = $Selected; protocols = $Protocols; port = $Port; extraServerArgs = $ExtraServerArgs; maxTokens = $MaxTokens
            ltx = "$LtxSize x $LtxFrames frames"; miniMaxH3 = "$MiniMaxH3Size x $MiniMaxH3Frames frames"; traceSteps = [bool]$TraceSteps
        }
        promptMarker   = $script:Marker
        tensorSharp    = Get-GitInfo $RepoRoot
        ggml           = Get-GgmlInfo
        serverHost     = $serverHost
        dashboard      = [ordered]@{
            mode = $dash.Mode; image = $dash.Image; imageId = $dash.ImageId; repoDigests = $dash.Digests; dll = $dash.Dll
            url = $dash.Url; otlpGrpc = $dash.Grpc; otlpHttp = $dash.Http
            telemetryApi = if ($script:Api.Usable) { 'usable' } else { "unusable: $($script:Api.Reason)" }
        }
        os             = [ordered]@{
            description  = [Runtime.InteropServices.RuntimeInformation]::OSDescription
            architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
            machine      = $script:Machine
        }
        dotnet         = Get-DotnetInfo
        gpu            = Get-GpuInfo
        runs           = $script:RunInfo
        limitations    = @($script:Limitations)
    }
}

function ConvertTo-MarkdownCell([string] $Text) {
    ($Text -replace '\|', '\|' -replace "`r?`n", '<br>')
}

function Write-ResultFiles($Environment) {
    $counts = [ordered]@{ PASS = 0; FAIL = 0; SKIPPED = 0; MANUAL = 0 }
    foreach ($row in $script:Results) { $counts[$row.status]++ }
    $summary = [ordered]@{
        total = $script:Results.Count; pass = $counts.PASS; fail = $counts.FAIL; skipped = $counts.SKIPPED; manual = $counts.MANUAL
        failed = ($counts.FAIL -gt 0)
        note = 'SKIPPED and MANUAL rows are not passes.'
    }
    Save-Json (Join-Path $OutDir 'results.json') ([ordered]@{ summary = $summary; promptMarker = $script:Marker; results = $script:Results })

    $md = [System.Text.StringBuilder]::new()
    [void]$md.AppendLine('# OpenTelemetry end-to-end results')
    [void]$md.AppendLine()
    $tensorSharp = $Environment.tensorSharp
    $digest = @($Environment.dashboard.repoDigests)[0]
    [void]$md.AppendLine("Run $($Environment.generatedAtUtc) on $($Environment.os.description) ($($Environment.os.architecture)).")
    $tensorSharpText = if ($tensorSharp.commit) { "$($tensorSharp.commit) on $($tensorSharp.branch) with $($tensorSharp.uncommittedChanges) uncommitted change(s)" } else { $tensorSharp.note }
    $ggmlText = if ($Environment.ggml.commit) { $Environment.ggml.commit } else { $Environment.ggml.note }
    [void]$md.AppendLine("TensorSharp: $tensorSharpText. ggml: $ggmlText.")
    $dashboardText = switch ($Environment.dashboard.mode) {
        'docker' { "container $($Environment.dashboard.image)$(if ($digest) { " ($digest)" })" }
        'dll' { "Aspire.Dashboard.dll at $($Environment.dashboard.dll)" }
        'external' { "already running at $($Environment.dashboard.url)" }
        default { 'not started' }
    }
    [void]$md.AppendLine("Dashboard: $dashboardText; telemetry API $($Environment.dashboard.telemetryApi).")
    [void]$md.AppendLine("GPU: $((@($Environment.gpu.devices)) -join '; ').")
    [void]$md.AppendLine()
    [void]$md.AppendLine("**$($counts.PASS) passed, $($counts.FAIL) failed, $($counts.SKIPPED) skipped, $($counts.MANUAL) manual.** SKIPPED and MANUAL rows are not passes.")
    [void]$md.AppendLine()
    [void]$md.AppendLine('| Run | Scenario | Status | Seconds | Details |')
    [void]$md.AppendLine('| --- | --- | --- | ---: | --- |')
    foreach ($row in $script:Results) {
        [void]$md.AppendLine("| $(ConvertTo-MarkdownCell $row.run) | $(ConvertTo-MarkdownCell $row.scenario) | $($row.status) | $($row.seconds) | $(ConvertTo-MarkdownCell $row.details) |")
    }
    [void]$md.AppendLine()
    [void]$md.AppendLine('Evidence: `environment.json` (versions, runs, models), `runs/<run>/` (server output, file logs, every request in `requests/`, exported spans and logs in `telemetry/`), `dashboard/` (login link, dashboard log, final telemetry snapshot).')
    if ($Environment.limitations.Count -gt 0) {
        [void]$md.AppendLine()
        [void]$md.AppendLine('Limitations:')
        foreach ($item in $Environment.limitations) { [void]$md.AppendLine("- $item") }
    }
    Save-Text (Join-Path $OutDir 'results.md') $md.ToString()
}

# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

if (-not $OutDir) { $OutDir = Join-Path $RepoRoot ("docs/validation/otel-e2e-" + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
$OutDir = [IO.Path]::GetFullPath($OutDir)
$evidenceRoots = @('docs/validation', 'artifacts') | ForEach-Object { [IO.Path]::GetFullPath((Join-Path $RepoRoot $_)) + [IO.Path]::DirectorySeparatorChar }
if (-not ($evidenceRoots | Where-Object { $OutDir.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) })) {
    throw "-OutDir must be inside docs/validation/ or artifacts/ under $RepoRoot (both are git-ignored); got $OutDir"
}
if ((Test-Path -LiteralPath $OutDir) -and (Get-ChildItem -LiteralPath $OutDir -Force | Select-Object -First 1)) {
    throw "-OutDir $OutDir already has files; use a fresh folder for each run."
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$script:Marker = 'TSOTEL' + (New-RandomHex 6).ToUpperInvariant()
$script:RunTag = New-RandomHex 3
$script:Machine = [Environment]::MachineName
$script:Dashboard = [pscustomobject]@{
    Mode = ''; Url = ''; Grpc = ''; Http = ''; ApiKey = ''; OtlpKey = ''; Container = ''; Process = $null
    LoginUrl = ''; Image = ''; ImageId = ''; Digests = @(); Dll = ''
}
$script:Api = [pscustomobject]@{ Base = ''; Key = ''; Usable = $false; Reason = 'the dashboard was not started'; UiHint = ''; LastError = ''; LastTruncation = '' }
$script:RunInfo = [ordered]@{}
$script:CurrentServer = $null
$script:LatencyBaseline = $null
$script:TraceBodies = @{}
$script:ServerHostInfo = $null
$script:ExtraArgs = @()
$script:Limitations = [System.Collections.Generic.List[string]]::new()
$script:Limitations.Add('Metrics are not readable through the dashboard telemetry API; the metrics row is MANUAL.')

Write-Step "OpenTelemetry end-to-end check; evidence in $OutDir; prompt marker $script:Marker"
try {
    $script:ServerHostInfo = Resolve-ServerHost $ServerHost
    Write-Step "Server host: $($script:ServerHostInfo.Path)"
    $chatArgs = Resolve-ModelArgs $ChatConfig 'ChatConfig'
    $embeddingArgs = Resolve-ModelArgs $EmbeddingConfig 'EmbeddingConfig' -Embeddings
    $ltxArgs = Resolve-ModelArgs $LtxConfig 'LtxConfig'
    $h3Args = Resolve-ModelArgs $MiniMaxH3Config 'MiniMaxH3Config'
    $script:ExtraArgs = if ($ExtraServerArgs.Trim()) { @(Split-CommandLine $ExtraServerArgs.Trim()) } else { @() }
    foreach ($size in $LtxSize, $MiniMaxH3Size) { [void](ConvertTo-Size $size) }

    $plan = Get-RunPlan $chatArgs $embeddingArgs $ltxArgs $h3Args
    if ($plan.Count -eq 0) {
        Write-Warning 'Nothing to run: every selected scenario was skipped.'
    } else {
        Write-Step "Runs: $(($plan | ForEach-Object Name) -join ', ')"
        Start-Dashboard
        foreach ($run in $plan) { Invoke-Run $run }
        if ($Selected -contains 'metrics') { Add-MetricsManual }
    }
} catch {
    Add-Result 'harness' 'setup' 'FAIL' "$($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))"
} finally {
    if ($script:CurrentServer) { try { Stop-TsServer $script:CurrentServer } catch { Write-Warning $_.Exception.Message } }
    # The dashboard keeps telemetry only while it runs: snapshot it before stopping it.
    try { Save-DashboardSnapshot } catch { Write-Warning "Dashboard snapshot: $($_.Exception.Message)" }
    try { Stop-Dashboard } catch { Write-Warning "Stopping the dashboard: $($_.Exception.Message)" }
    $environmentInfo = $null
    try {
        $environmentInfo = Get-EnvironmentInfo
        Save-Json (Join-Path $OutDir 'environment.json') $environmentInfo
    } catch {
        Write-Warning "environment.json: $($_.Exception.Message)"
    }
    if ($environmentInfo) { Write-ResultFiles $environmentInfo }
    else { Save-Json (Join-Path $OutDir 'results.json') ([ordered]@{ results = $script:Results }) }
    $script:Http.Dispose()
}

$failedCount = @($script:Results | Where-Object { $_.status -eq 'FAIL' }).Count
$counts = $script:Results | Group-Object status | ForEach-Object { "$($_.Count) $($_.Name)" }
Write-Step "Done: $($counts -join ', '). Results: $(Join-Path $OutDir 'results.md')"
exit ([int]($failedCount -gt 0))
