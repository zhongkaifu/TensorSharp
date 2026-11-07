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
End-to-end check of TensorSharp's in-process Application Insights export against a real resource.

.DESCRIPTION
Reads the connection string and AppId of an Application Insights resource with the Azure CLI,
starts TensorSharp.Server.Host several times with APPLICATIONINSIGHTS_CONNECTION_STRING set (and a
unique OTEL_SERVICE_NAME per run), sends real requests, stops each server gracefully so its
exporters flush, then queries the resource through the Application Insights query API
(az rest ... https://api.applicationinsights.io/v1/apps/<AppId>/query) until the expected rows
arrive or -IngestionTimeoutMinutes passes, and checks them.

The connection string is never printed, logged or written: it reaches the server processes through
their environment only. The resource id is not written either; results name the resource by its
name. Every query is filtered on the run's own cloud_RoleName and start time, so the resource can be
shared. The script sends real telemetry to the resource, which is billed like any other ingestion.

Every check ends as PASS, FAIL, SKIPPED (its model or setup was not given) or MANUAL (it needs a
person to look). SKIPPED and MANUAL never count as passed. results.json, results.md,
environment.json, every request, and every query with its last result go to -OutDir, which must be
under the git-ignored docs/validation/ or artifacts/ folders. The script exits with 1 when anything
FAILED.

eng/telemetry/README.md lists the runs and checks and explains how to read the results.

.EXAMPLE
pwsh eng/telemetry/appinsights-e2e.ps1 -AppInsightsResourceId $id -ChatModel D:\models\qwen3-4b-Q8_0.gguf

.EXAMPLE
pwsh eng/telemetry/appinsights-e2e.ps1 -AppInsightsResourceId $id -ChatModel D:\models\qwen3-4b-Q8_0.gguf -EntraId -Scenarios llm-openai,llm-burst

.EXAMPLE
pwsh eng/telemetry/appinsights-e2e.ps1 -AppInsightsResourceId $id -LtxConfig D:\models\ltx25.json -MiniMaxH3Config D:\models\minimax-h3.json -Scenarios video-ltx,video-minimax-h3,studio-generate,metrics
#>
#Requires -Version 7.2
[CmdletBinding()]
param(
    # ARM id of the Application Insights resource, a microsoft.insights/components resource:
    # /subscriptions/<id>/resourceGroups/<group>/providers/microsoft.insights/components/<name>.
    # Its connection string and AppId are read with az at run time. The id is never written to
    # the evidence.
    [Parameter(Mandatory)] [string] $AppInsightsResourceId,

    # A built TensorSharp.Server.Host: its .dll (run with dotnet), its executable, or a folder to
    # search. Default: the newest TensorSharp.Server.Host.dll under TensorSharp.Server.Host/bin.
    [string] $ServerHost = '',

    # Chat model for the llm-* and content-capture scenarios: a GGUF file (passed as --model), a
    # --config JSON file, or server arguments in one string ("--model x.gguf --backend ggml_cuda").
    [string] $ChatModel = '',

    # Embedding model for the embeddings scenario, in the same forms. A GGUF file also gets
    # --embeddings; a --config file or an argument string must turn embeddings on itself.
    [string] $EmbeddingModel = '',

    # LTX 2.5 video model for video-ltx (and video-cancel / studio-generate), in the same forms.
    [string] $LtxConfig = '',

    # MiniMax-H3 video model for video-minimax-h3 (and video-cancel / studio-generate without LTX).
    [string] $MiniMaxH3Config = '',

    # Run only these scenarios (a list, or one comma-separated string as `pwsh -File` passes it).
    # Default: all. A scenario whose model or switch is not given is SKIPPED.
    [ArgumentCompletions('llm-openai-stream', 'llm-openai', 'llm-ollama-generate', 'llm-ollama-chat', 'llm-webui',
        'llm-client-traceparent', 'llm-cancel', 'llm-bad-request', 'llm-burst', 'queue-status-poll',
        'embeddings', 'video-ltx', 'video-minimax-h3', 'video-cancel', 'studio-generate',
        'content-capture', 'metrics', 'entra-id')]
    [string[]] $Scenarios = @(),

    # Port TensorSharp.Server.Host listens on, bound to 127.0.0.1. The server runs one at a time.
    [ValidateRange(1, 65535)] [int] $Port = 5055,

    # Evidence folder. Default: docs/validation/appinsights-e2e-<yyyyMMdd-HHmmss> under the repository root.
    [string] $OutDir = '',

    # How long after a server stops to keep querying for its rows. Ingestion usually takes 1-5 minutes.
    [ValidateRange(1, 240)] [int] $IngestionTimeoutMinutes = 12,

    # Also run one server with TENSORSHARP_AZURE_MONITOR_AAD=1 (Entra ID ingestion) and check that it ingests.
    [switch] $EntraId,

    # Also export OTLP to a local Aspire dashboard (or any OTLP receiver) when -OtlpEndpoint is reachable.
    [switch] $BothSinks,

    # With -BothSinks: the OTLP endpoint, its protocol, and its x-otlp-api-key (if it needs one).
    [string] $OtlpEndpoint = 'http://localhost:4317',
    [ArgumentCompletions('grpc', 'http/protobuf')] [string] $OtlpProtocol = 'grpc',
    [string] $OtlpApiKey = '',

    # With -BothSinks: the dashboard UI URL and its telemetry API key (X-API-Key). With the key, the
    # script checks that the dashboard holds the same requests; without it that row is MANUAL.
    [string] $DashboardUrl = 'http://localhost:18888',
    [string] $DashboardApiKey = '',

    # Write an OTEL_DIAGNOSTICS.json into each run folder and run the server there, so the
    # OpenTelemetry SDK's self-diagnostics (the Azure Monitor exporter's own warnings) land in the
    # evidence. Server arguments given as a string must then use absolute paths.
    [switch] $ExporterDiagnostics,

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

    # Seconds to wait after a run's last request before stopping the server, so the exporters'
    # batches leave on their own as well as in the shutdown flush.
    [ValidateRange(0, 600)] [int] $SettleSec = 8,

    # How long a server may take to shut down after the stop signal before it is killed.
    [ValidateRange(5, 1800)] [int] $StopGraceSec = 90
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$Invariant = [Globalization.CultureInfo]::InvariantCulture
$QueryApiResource = 'https://api.applicationinsights.io'

$AllScenarioNames = @('llm-openai-stream', 'llm-openai', 'llm-ollama-generate', 'llm-ollama-chat', 'llm-webui',
    'llm-client-traceparent', 'llm-cancel', 'llm-bad-request', 'llm-burst', 'queue-status-poll',
    'embeddings', 'video-ltx', 'video-minimax-h3', 'video-cancel', 'studio-generate',
    'content-capture', 'metrics', 'entra-id')
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
# -EntraId asks for its run whatever -Scenarios says.
if ($EntraId -and $Selected -notcontains 'entra-id') { $Selected = @($Selected) + 'entra-id' }
$LtxExpectStages = Split-ListParameter $LtxExpectStages

# Inherited variables that would change what a server run tests. Every run starts without them
# and gets only what the run sets. AZURE_* (the identity DefaultAzureCredential may use) is kept.
$DroppedVariablePrefixes = @('OTEL_', 'TENSORSHARP_OTEL', 'APPLICATIONINSIGHTS_')
$DroppedVariables = @('TENSORSHARP_AZURE_MONITOR_AAD', 'PORT', 'HOST', 'ASPNETCORE_URLS', 'TENSORSHARP_LOG_DIR', 'TS_STUDIO_DIRECTORY')

# ---------------------------------------------------------------------------------------------
# Secrets: every text the script writes or prints goes through Protect-Text first.
# ---------------------------------------------------------------------------------------------

$script:SecretValues = [System.Collections.Generic.List[object]]::new()

function Register-Secret([string] $Value, [string] $Label) {
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value.Trim().Length -lt 8) { return }
    $script:SecretValues.Add([pscustomobject]@{ Value = $Value.Trim(); Label = $Label })
    # Longest first, so the connection string is replaced before the key inside it.
    $sorted = @($script:SecretValues | Sort-Object { $_.Value.Length } -Descending)
    $script:SecretValues.Clear()
    foreach ($item in $sorted) { $script:SecretValues.Add($item) }
}

function Protect-Text([string] $Text) {
    if ([string]::IsNullOrEmpty($Text) -or $script:SecretValues.Count -eq 0) { return $Text }
    foreach ($secret in $script:SecretValues) {
        $Text = $Text.Replace($secret.Value, $secret.Label, [StringComparison]::OrdinalIgnoreCase)
    }
    return $Text
}

# ---------------------------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------------------------

function Write-Step([string] $Message) {
    Write-Host ("[{0:HH:mm:ss}] {1}" -f (Get-Date), (Protect-Text $Message))
}

function New-RandomHex([int] $Bytes) {
    [Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes($Bytes)).ToLowerInvariant()
}

function Save-Text([string] $Path, [string] $Text) {
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    [IO.File]::WriteAllText($Path, (Protect-Text ([string]$Text)), [Text.UTF8Encoding]::new($false))
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

# A -ChatModel style value: a model file, a --config JSON file, or server arguments.
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

function ConvertTo-NullableDouble($Value) {
    if ($null -eq $Value) { return $null }
    $text = [Convert]::ToString($Value, $Invariant)
    if ([string]::IsNullOrWhiteSpace($text)) { return $null }
    $parsed = 0.0
    if ([double]::TryParse($text, [Globalization.NumberStyles]::Float, $Invariant, [ref]$parsed)) { return $parsed }
    return $null
}

function Get-UrlPath([string] $Url) {
    if (-not $Url) { return '' }
    try { return ([uri]$Url).AbsolutePath } catch { return '' }
}

# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------

$script:Results = [System.Collections.Generic.List[object]]::new()

function Add-Result([string] $Run, [string] $Scenario, [string] $Status, [string] $Details = '', [double] $Seconds = 0) {
    if ($Status -notin 'PASS', 'FAIL', 'SKIPPED', 'MANUAL') { throw "Unknown status $Status" }
    $Details = Protect-Text $Details
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
# HTTP (to the server, and to the dashboard with -BothSinks)
# ---------------------------------------------------------------------------------------------

$script:HttpHandler = [System.Net.Http.SocketsHttpHandler]::new()
$script:HttpHandler.UseProxy = $false          # everything here is on this machine
$script:HttpHandler.AllowAutoRedirect = $false
# Disposing an unfinished response must close the connection, so the server sees the client go.
# By default the handler first reads up to 1 MB / 2 s of the rest to reuse the connection,
# which for a short reply means the server streams it all and never sees a disconnect.
$script:HttpHandler.MaxResponseDrainSize = 0
$script:Http =[System.Net.Http.HttpClient]::new($script:HttpHandler)
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
    $file = "requests/{0:D3}-{1}.json" -f $Ctx.ExchangeCount, $Name
    Save-Json (Join-Path $Ctx.Dir $file) ([ordered]@{
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
    $Ctx.Sent.Add([ordered]@{
            file = $file; method = $Result.Method; path = (Get-UrlPath $Result.Url); status = $Result.Status
            requestId = $Result.RequestId; aborted = $Result.Aborted
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
# The server process
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

# Start the server with its output streamed straight into files (unbuffered, so they can be read
# while it runs), inherited telemetry variables dropped. CreateNoWindow gives it a console of its
# own on Windows, without a window: Stop-TsServer sends CTRL_BREAK to that console, which reaches
# the server and nothing else, so it shuts down and its exporters flush.
function Start-LoggedProcess {
    param(
        [string] $FileName, [string[]] $Arguments, [System.Collections.IDictionary] $Environment, [string] $WorkingDirectory,
        [string] $StdoutPath, [string] $StderrPath
    )
    $psi = [System.Diagnostics.ProcessStartInfo]::new($FileName)
    foreach ($argument in $Arguments) { $psi.ArgumentList.Add([string]$argument) }
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.WorkingDirectory = $WorkingDirectory
    foreach ($key in @($psi.Environment.Keys)) { if (Test-DroppedVariable $key) { [void]$psi.Environment.Remove($key) } }
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
        StopInfo   = $null
    }
}

# A tiny helper, run in its own pwsh with its own hidden console: it detaches from that console,
# attaches to the server's, ignores the event itself, and raises CTRL_BREAK there. The server's
# ConsoleLifetime maps CTRL_BREAK (SIGQUIT) to a graceful stop. CTRL_BREAK, not CTRL_C, because
# a process started in a new process group (as some launchers do) inherits "ignore CTRL_C".
$script:ConsoleSignalSource = @'
using System;
using System.Runtime.InteropServices;
using System.Threading;

public static class TensorSharpE2eConsoleSignal
{
    private delegate bool HandlerRoutine(uint ctrlType);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool FreeConsole();

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AttachConsole(uint processId);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetConsoleCtrlHandler(HandlerRoutine handler, bool add);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GenerateConsoleCtrlEvent(uint ctrlEvent, uint processGroupId);

    private static bool Ignore(uint ctrlType) { return true; }

    // A field keeps the delegate alive while it is registered.
    private static readonly HandlerRoutine s_ignore = Ignore;

    // 0: raised; 1: could not attach to the process's console; 2: could not raise the event.
    public static int Send(uint processId, uint ctrlEvent)
    {
        FreeConsole();
        if (!AttachConsole(processId)) return 1;
        SetConsoleCtrlHandler(s_ignore, true);
        bool raised = GenerateConsoleCtrlEvent(ctrlEvent, 0);
        Thread.Sleep(500);
        FreeConsole();
        return raised ? 0 : 2;
    }
}
'@

function Send-ConsoleBreak([int] $ProcessId) {
    $exe = Join-Path $PSHOME 'pwsh.exe'
    if (-not (Test-Path -LiteralPath $exe)) {
        $found = Get-Command pwsh -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $found) { return 'no pwsh executable found to run the signal helper' }
        $exe = $found.Source
    }
    $command = "Add-Type -TypeDefinition @'`n" + $script:ConsoleSignalSource + "`n'@`nexit [TensorSharpE2eConsoleSignal]::Send($ProcessId, 1)"
    $psi = [System.Diagnostics.ProcessStartInfo]::new($exe)
    foreach ($argument in @('-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand',
            [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command)))) {
        $psi.ArgumentList.Add($argument)
    }
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $helper = [System.Diagnostics.Process]::Start($psi)
    try {
        $stderr = $helper.StandardError.ReadToEndAsync()
        [void]$helper.StandardOutput.ReadToEndAsync()
        if (-not $helper.WaitForExit(90000)) {
            try { $helper.Kill() } catch { }
            return 'the signal helper did not finish within 90 s'
        }
        switch ($helper.ExitCode) {
            0 { return 'sent' }
            1 { return 'AttachConsole failed: the server has no console of its own' }
            2 { return 'GenerateConsoleCtrlEvent failed' }
            default {
                $text = ''
                try { $text = ([string]$stderr.Result).Trim() } catch { }
                if ($text.Length -gt 300) { $text = $text.Substring(0, 300) + '...' }
                return "the signal helper exited with $($helper.ExitCode) $text"
            }
        }
    } finally {
        $helper.Dispose()
    }
}

function Start-TsServer([string] $Dir, [string[]] $ServerArgs, [System.Collections.IDictionary] $Environment, [string] $WorkingDirectory) {
    $handle = Start-LoggedProcess -FileName $script:ServerHostInfo.FileName `
        -Arguments (@($script:ServerHostInfo.Prefix) + @($ServerArgs)) `
        -Environment $Environment -WorkingDirectory $WorkingDirectory `
        -StdoutPath (Join-Path $Dir 'server.stdout.log') -StderrPath (Join-Path $Dir 'server.stderr.log')
    $handle | Add-Member -NotePropertyName Base -NotePropertyValue "http://127.0.0.1:$Port"
    $handle | Add-Member -NotePropertyName LogDir -NotePropertyValue $Environment['TENSORSHARP_LOG_DIR']
    $handle | Add-Member -NotePropertyName DiagnosticsDir -NotePropertyValue ''
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

# Stop the server so it shuts down by itself (its tracer, meter and logger providers flush on
# dispose): CTRL_BREAK on Windows, SIGTERM elsewhere. Killed only when that fails or times out.
function Stop-TsServer($Server, [switch] $Quick) {
    if ($null -eq $Server) { return $null }
    if ($null -eq $Server.Process) { return $Server.StopInfo }
    $process = $Server.Process
    $info = [ordered]@{ mode = ''; signal = ''; signalResult = ''; settleSec = 0; seconds = 0.0; exitCode = $null }
    try {
        if ($process.HasExited) {
            $info.mode = 'exited before it was stopped'
        } else {
            if (-not $Quick -and $SettleSec -gt 0) {
                Start-Sleep -Seconds $SettleSec
                $info.settleSec = $SettleSec
            }
            $watch = [Diagnostics.Stopwatch]::StartNew()
            $sent = $false
            if ($IsWindows) {
                $info.signal = 'CTRL_BREAK_EVENT'
                $info.signalResult = Send-ConsoleBreak $Server.Pid
                $sent = ($info.signalResult -eq 'sent')
            } elseif (Test-Path '/bin/kill') {
                $info.signal = 'SIGTERM'
                & '/bin/kill' '-TERM' "$($Server.Pid)" 2>$null | Out-Null
                $sent = ($LASTEXITCODE -eq 0)
                $info.signalResult = if ($sent) { 'sent' } else { "kill exited with $LASTEXITCODE" }
            } else {
                $info.signalResult = 'no way to signal the process on this OS'
            }
            $exited = $false
            if ($sent) { $exited = $process.WaitForExit($StopGraceSec * 1000) }
            if ($exited) {
                $info.mode = 'graceful'
            } else {
                $info.mode = if ($sent) { "killed: still running $StopGraceSec s after the signal" } else { "killed: the signal could not be sent ($($info.signalResult))" }
                $process.Kill($true)
                [void]$process.WaitForExit(15000)
            }
            $info.seconds = [math]::Round($watch.Elapsed.TotalSeconds, 1)
        }
        if ($process.HasExited) { $info.exitCode = $process.ExitCode }
    } catch {
        $info.mode = "error while stopping: $($_.Exception.Message)"
        Write-Warning (Protect-Text "Stopping process $($Server.Pid): $($_.Exception.Message)")
        try { if (-not $process.HasExited) { $process.Kill($true) } } catch { }
    }
    try { [void][System.Threading.Tasks.Task]::WaitAll([System.Threading.Tasks.Task[]]@($Server.CopyOut, $Server.CopyErr), 10000) } catch { }
    $Server.OutFile.Dispose()
    $Server.ErrFile.Dispose()
    $Server.ExitCode = $info.exitCode
    $process.Dispose()
    $Server.Process = $null
    $Server.StopInfo = $info
    Wait-PortFree $Port
    return $info
}

function Get-ServerOutputFiles($Server) {
    $files = @($Server.StdoutPath, $Server.StderrPath)
    foreach ($dir in @($Server.LogDir, $Server.DiagnosticsDir)) {
        if ($dir -and (Test-Path -LiteralPath $dir)) {
            $files += @(Get-ChildItem -LiteralPath $dir -File -Recurse -ErrorAction SilentlyContinue | ForEach-Object FullName)
        }
    }
    return , @($files | Where-Object { $_ })
}

# The host's one-line telemetry summary, e.g. "Telemetry: traces(azuremonitor) ... -> Azure Monitor; ...".
function Get-TelemetryLine($Server) {
    foreach ($path in (Get-ServerOutputFiles $Server)) {
        $match = [regex]::Match((Read-SharedText $path), 'Telemetry: (?:off \(|traces\()[^\r\n]*')
        if ($match.Success) { return $match.Value.TrimEnd() }
    }
    return $null
}

# Lines in the server's output that look like exporter or credential trouble.
function Get-ExporterMessages($Ctx) {
    $topic = '(?i)(azure|monitor|exporter|credential|ingest|opentelemetry|applicationinsights)'
    $severity = '(?i)(warn|fail|error|exception|unauthori[sz]ed|forbidden|denied|\b40[13]\b)'
    $lines = [System.Collections.Generic.List[string]]::new()
    foreach ($file in (Get-ServerOutputFiles $Ctx.Server)) {
        $text = (Read-SharedText $file) -replace "`0", ''
        foreach ($line in ($text -split "`r?`n")) {
            if ($lines.Count -ge 200) { break }
            if ($line -match $topic -and $line -match $severity) { $lines.Add((Protect-Text "$(Split-Path -Leaf $file): $($line.Trim())")) }
        }
    }
    if ($lines.Count -gt 0) { Save-Text (Join-Path $Ctx.Dir 'exporter-messages.txt') ($lines -join "`n") }
    return , $lines.ToArray()
}

# ---------------------------------------------------------------------------------------------
# Azure CLI and the Application Insights query API
# ---------------------------------------------------------------------------------------------

# Run az from PowerShell (never through a POSIX shell, which would rewrite the leading
# /subscriptions of a resource id as a path). stdout is returned to the caller only; stderr is
# returned with every known secret replaced.
function Invoke-Az([string[]] $Arguments) {
    $errPath = Join-Path $script:TempDir ("az-" + (New-RandomHex 6) + '.err')
    $previousEncoding = $null
    $previousPythonEncoding = $env:PYTHONIOENCODING
    try { $previousEncoding = [Console]::OutputEncoding; [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false) } catch { $previousEncoding = $null }
    $env:PYTHONIOENCODING = 'utf-8'
    try {
        $stdout = & $script:AzPath @Arguments 2> $errPath
        $code = $LASTEXITCODE
    } finally {
        $env:PYTHONIOENCODING = $previousPythonEncoding
        if ($previousEncoding) { try { [Console]::OutputEncoding = $previousEncoding } catch { } }
    }
    $stderr = ''
    if (Test-Path -LiteralPath $errPath) {
        $stderr = [IO.File]::ReadAllText($errPath)
        Remove-Item -LiteralPath $errPath -Force -ErrorAction SilentlyContinue
    }
    [pscustomobject]@{ ExitCode = $code; Stdout = (@($stdout) -join "`n"); Stderr = (Protect-Text $stderr).Trim() }
}

function Connect-AppInsights {
    $idMatch = [regex]::Match($AppInsightsResourceId.Trim(),
        '^/subscriptions/([^/]+)/resourceGroups/[^/]+/providers/microsoft\.insights/components/([^/]+)/?$',
        [Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $idMatch.Success) {
        throw '-AppInsightsResourceId is not the ARM id of a microsoft.insights/components resource (/subscriptions/<id>/resourceGroups/<group>/providers/microsoft.insights/components/<name>).'
    }
    Register-Secret $AppInsightsResourceId.Trim() '<resource-id>'
    Register-Secret $idMatch.Groups[1].Value '<subscription-id>'
    $script:Ai.ResourceName = $idMatch.Groups[2].Value

    $az = Get-Command az -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $az) { throw 'The Azure CLI (az) is not on PATH. Install it and run az login.' }
    $script:AzPath = $az.Source

    $version = Invoke-Az @('version', '--output', 'json', '--only-show-errors')
    if ($version.ExitCode -eq 0) {
        $parsed = ConvertFrom-JsonText $version.Stdout
        if ($parsed) { $script:Ai.AzVersion = [string]$parsed.'azure-cli' }
    }
    $account = Invoke-Az @('account', 'show', '--query', 'user.type', '--output', 'tsv', '--only-show-errors')
    if ($account.ExitCode -ne 0) { throw "az is not signed in (az account show: $($account.Stderr)). Run az login first." }
    $script:Ai.AccountType = $account.Stdout.Trim()

    Write-Step "Reading the connection string of Application Insights resource $($script:Ai.ResourceName) with az"
    $show = Invoke-Az @('resource', 'show', '--ids', $AppInsightsResourceId.Trim(),
        '--query', '[properties.ConnectionString,properties.AppId,name,type]', '--output', 'json', '--only-show-errors')
    if ($show.ExitCode -ne 0) { throw "az resource show failed for the resource: $($show.Stderr)" }
    # The output holds the connection string: it is never echoed, not even in an error.
    $values = $null
    try { $values = @(ConvertFrom-Json -InputObject $show.Stdout) } catch { $values = $null }
    if ($null -eq $values -or $values.Count -lt 4) { throw 'az resource show did not return the expected [ConnectionString, AppId, name, type] list.' }
    $connectionString = [string]$values[0]
    $appId = [string]$values[1]
    if ([string]$values[2]) { $script:Ai.ResourceName = [string]$values[2] }
    if ([string]$values[3] -and [string]$values[3] -ne 'microsoft.insights/components') {
        throw "The resource is a '$([string]$values[3])', not a microsoft.insights/components resource."
    }
    if ([string]::IsNullOrWhiteSpace($connectionString)) { throw 'The resource has no properties.ConnectionString (or the signed-in identity cannot read it).' }
    if ([string]::IsNullOrWhiteSpace($appId)) { throw 'The resource has no properties.AppId.' }
    Register-Secret $connectionString '<connection-string>'
    $keyMatch = [regex]::Match($connectionString, '(?:^|;)\s*InstrumentationKey\s*=\s*([^;]+)', [Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if ($keyMatch.Success) {
        $script:Ai.InstrumentationKey = $keyMatch.Groups[1].Value.Trim()
        Register-Secret $script:Ai.InstrumentationKey '<instrumentation-key>'
    }
    Register-Secret $appId '<app-id>'
    $script:Ai.ConnectionString = $connectionString.Trim()
    $script:Ai.QueryUrl = "https://api.applicationinsights.io/v1/apps/$($appId.Trim())/query"

    # Fail now, before any server runs, when the query API cannot be read.
    $probe = Invoke-AiQuery 'print ok = 1'
    Save-Json (Join-Path $OutDir 'appinsights-preflight.json') ([ordered]@{ query = 'print ok = 1'; ok = $probe.Ok; error = $probe.Error; rows = $probe.Rows })
    if (-not $probe.Ok) { throw "The Application Insights query API is not readable with the signed-in identity: $($probe.Error)" }
    $script:Ai.Connected = $true
    Write-Step "Application Insights resource $($script:Ai.ResourceName): connection string read, query API answers"
}

# One query through the Application Insights query API. Rows come back as objects keyed by column.
function Invoke-AiQuery([string] $Query) {
    $bodyPath = Join-Path $script:TempDir ("query-" + (New-RandomHex 6) + '.json')
    [IO.File]::WriteAllText($bodyPath, (ConvertTo-Json -InputObject ([ordered]@{ query = $Query; timespan = 'P2D' }) -Compress), [Text.UTF8Encoding]::new($false))
    try {
        $az = Invoke-Az @('rest', '--method', 'post', '--url', $script:Ai.QueryUrl, '--resource', $QueryApiResource,
            '--body', "@$bodyPath", '--output', 'json', '--only-show-errors')
    } finally {
        Remove-Item -LiteralPath $bodyPath -Force -ErrorAction SilentlyContinue
    }
    if ($az.ExitCode -ne 0) {
        $text = if ($az.Stderr) { $az.Stderr } else { "az rest exited with $($az.ExitCode)" }
        if ($text.Length -gt 1500) { $text = $text.Substring(0, 1500) + '...' }
        # A query or permission error will not change by waiting; anything else (throttling,
        # a gateway error, a token refresh) is retried.
        $permanent = $text -match '(?i)SyntaxError|SemanticError|BadArgumentError|az login|AuthorizationFailed|InsufficientAccess|Forbidden|PathNotFound|InvalidAuthenticationToken'
        return [pscustomobject]@{ Ok = $false; Permanent = $permanent; Error = $text; Columns = @(); Rows = @() }
    }
    $json = $null
    try { $json = ConvertFrom-Json -InputObject $az.Stdout } catch { $json = $null }
    if ($null -eq $json -or -not $json.PSObject.Properties['tables']) {
        return [pscustomobject]@{ Ok = $false; Permanent = $false; Error = 'the query API answered with something other than a result table'; Columns = @(); Rows = @() }
    }
    $table = @($json.tables)[0]
    $columns = @(if ($table) { @($table.columns) | ForEach-Object { [string]$_.name } })
    $rows = [System.Collections.Generic.List[object]]::new()
    if ($table) {
        foreach ($row in @($table.rows)) {
            $values = @($row)
            $item = [ordered]@{}
            for ($i = 0; $i -lt $columns.Count; $i++) { $item[$columns[$i]] = if ($i -lt $values.Count) { $values[$i] } else { $null } }
            $rows.Add([pscustomobject]$item)
        }
    }
    $partial = if ($json.PSObject.Properties['error'] -and $json.error) { Protect-Text (ConvertTo-Json -InputObject $json.error -Compress -Depth 8) } else { '' }
    [pscustomobject]@{ Ok = $true; Permanent = $false; Error = $partial; Columns = $columns; Rows = $rows.ToArray() }
}

# Query until $Missing (given the rows) returns nothing, a permanent error comes back, or the
# deadline passes; with -Once, query once. The query and its last result are saved under the
# run's appinsights/ folder. Locals start with __ because $Missing sees the caller's variables.
function Invoke-AiPoll {
    param($Ctx, [string] $Name, [string] $Query, [scriptblock] $Missing, [datetime] $DeadlineUtc, [switch] $Once)
    $Ctx.QueryCount++
    $__base = "{0:D2}-{1}" -f $Ctx.QueryCount, $Name
    $__dir = Join-Path $Ctx.Dir 'appinsights'
    Save-Text (Join-Path $__dir "$__base.kql") $Query
    $__attempts = 0
    $__delay = 15.0
    $__first = [DateTime]::UtcNow
    $__gap = ''
    $__result = $null
    while ($true) {
        $__attempts++
        $__result = Invoke-AiQuery $Query
        $__gap = ''
        if ($__result.Ok) {
            if ($Missing) { $__gap = [string](& $Missing $__result.Rows) }
            if (-not $__gap) { break }
        } elseif ($__result.Permanent) {
            break
        }
        if ($Once -or [DateTime]::UtcNow -ge $DeadlineUtc) { break }
        $__wait = [math]::Min($__delay, ($DeadlineUtc - [DateTime]::UtcNow).TotalSeconds)
        $__why = if ($__result.Ok) { $__gap } else { "query failed: $($__result.Error)" }
        if ($__why.Length -gt 200) { $__why = $__why.Substring(0, 200) + '...' }
        Write-Step ("    {0} {1}: {2}; next query in {3:F0} s (until {4:HH:mm:ss} UTC)" -f $Ctx.Name, $Name, $__why, [math]::Max(0, $__wait), $DeadlineUtc)
        if ($__wait -gt 0) { Start-Sleep -Milliseconds ([int]($__wait * 1000)) }
        $__delay = [math]::Min($__delay * 1.5, 60)
    }
    $__ready = $__result.Ok -and -not $__gap
    $__rows = @($__result.Rows)
    Save-Json (Join-Path $__dir "$__base.json") ([ordered]@{
            query           = $Query
            ready           = $__ready
            attempts        = $__attempts
            firstAttemptUtc = $__first.ToString('o')
            lastAttemptUtc  = [DateTime]::UtcNow.ToString('o')
            deadlineUtc     = $DeadlineUtc.ToString('o')
            missing         = $__gap
            error           = $__result.Error
            columns         = @($__result.Columns)
            rowCount        = $__rows.Count
            rows            = $__rows
        })
    [pscustomobject]@{
        Name = $Name; File = "appinsights/$__base.json"; Ready = $__ready; Ok = $__result.Ok; Error = $__result.Error
        Missing = $__gap; Rows = $__rows; Attempts = $__attempts
    }
}

function ConvertTo-KqlString([string] $Value) {
    '"' + $Value.Replace('\', '\\').Replace('"', '\"') + '"'
}

function Format-KqlTime([datetime] $Utc) {
    $Utc.ToUniversalTime().ToString("yyyy-MM-dd'T'HH:mm:ss.fffffff'Z'", $Invariant)
}

# A dynamic column comes back as JSON text, or already parsed.
function ConvertFrom-AiDynamic($Value) {
    if ($null -eq $Value) { return }
    if ($Value -is [string]) {
        if (-not $Value) { return }
        $parsed = ConvertFrom-JsonText $Value
        if ($null -ne $parsed) { return $parsed }
        return $Value
    }
    return $Value
}

# The queries of one run. Each keeps only the run's own rows: its cloud_RoleName (a service name
# no other process uses) and timestamps after the server started. Numeric span attributes arrive
# as strings in customDimensions (the in-process exporter), hence todouble().
function Get-RunQueries($Ctx) {
    $where = "| where timestamp > datetime($(Format-KqlTime $Ctx.StartUtc)) and cloud_RoleName == $(ConvertTo-KqlString $Ctx.Service)"
    $prefix = $script:MarkerPrefix
    $queries = [ordered]@{}
    $queries['requests'] = @"
requests
$where
| project timestamp, id, name, url, resultCode, success, duration, itemCount, operation_Id, operation_ParentId, cloud_RoleInstance,
    rid = tostring(customDimensions["tensorsharp.request_id"]),
    protocol = tostring(customDimensions["tensorsharp.protocol"])
| order by timestamp asc
"@
    $queries['dependencies'] = @"
dependencies
$where
| extend cd = customDimensions, cm = customMeasurements
| project timestamp, id, name, type, success, duration, itemCount, operation_Id, operation_ParentId,
    rid = tostring(cd["tensorsharp.request_id"]),
    genOperation = tostring(cd["gen_ai.operation.name"]),
    genModel = tostring(cd["gen_ai.request.model"]),
    genProvider = tostring(cd["gen_ai.provider.name"]),
    protocol = tostring(cd["tensorsharp.protocol"]),
    outcome = tostring(cd["tensorsharp.outcome"]),
    isInternal = tolower(tostring(cd["tensorsharp.internal"])),
    errorType = tostring(cd["error.type"]),
    inputTokens = coalesce(todouble(cd["gen_ai.usage.input_tokens"]), todouble(cm["gen_ai.usage.input_tokens"])),
    outputTokens = coalesce(todouble(cd["gen_ai.usage.output_tokens"]), todouble(cm["gen_ai.usage.output_tokens"])),
    cacheReadTokens = coalesce(todouble(cd["gen_ai.usage.cache_read.input_tokens"]), todouble(cm["gen_ai.usage.cache_read.input_tokens"])),
    ttftMs = coalesce(todouble(cd["tensorsharp.ttft_ms"]), todouble(cm["tensorsharp.ttft_ms"])),
    videoCached = tolower(tostring(cd["tensorsharp.video.cached"])),
    videoPhase = tostring(cd["tensorsharp.video.phase"]),
    videoStage = tostring(cd["tensorsharp.video.stage"]),
    jobId = tostring(cd["tensorsharp.studio.job_id"]),
    links = tostring(cd["_MS.links"])
| order by timestamp asc
"@
    $queries['traces'] = @"
traces
$where
| extend rid = tostring(customDimensions["tensorsharp.request_id"])
| summarize rows = count() by rid, operation_Id
"@
    $queries['metrics'] = @"
customMetrics
$where
| extend tokenType = tostring(customDimensions["tensorsharp.token.type"])
| summarize rows = count(), valueCount = sum(valueCount), valueSum = sum(valueSum), tokenTypes = make_set(tokenType, 16) by name
| order by name asc
"@
    $queries['markers'] = @"
union withsource = sourceTable requests, dependencies, traces, exceptions
$where
| extend text = strcat(name, " ", url, " ", data, " ", message, " ", outerMessage, " ", innermostMessage, " ", tostring(details), " ", tostring(customDimensions))
| where text contains_cs "$($prefix)"
| project sourceTable, timestamp, itemType, name, operation_Id, marker = extract("($($prefix)P[0-9]+)", 1, text)
| take 200
"@
    $queries['role-instance'] = @"
union requests, dependencies, traces, exceptions, customMetrics
$where
| summarize rows = count() by itemType, cloud_RoleInstance
"@
    return $queries
}

# ---------------------------------------------------------------------------------------------
# Expectations: what each request the run sent must leave in Application Insights
# ---------------------------------------------------------------------------------------------

function New-PendingEntry([string] $Name) {
    [pscustomobject]@{
        Name         = $Name
        Check        = (New-Check)
        Seconds      = 0.0
        Expectations = [System.Collections.Generic.List[object]]::new()
        Extra        = @{}
    }
}

# Kind: plain (a requests row and, with ExpectLogs, traces rows), gen (plus a chat /
# text_completion / embeddings dependency, or with NoGenAi none), video (generate_video and its
# phases), studio (a Studio job's own operation).
function Add-Expectation($Entry, $Response, [hashtable] $Fields = @{}) {
    $exp = [pscustomobject]@{
        Scenario = $Entry.Name; Label = ''; Kind = 'plain'
        RequestId = if ($Response) { [string]$Response.RequestId } else { '' }
        Method = if ($Response) { $Response.Method } else { '' }
        Path = if ($Response) { Get-UrlPath $Response.Url } else { '' }
        Status = if ($Response) { [int]$Response.Status } else { 0 }
        Aborted = if ($Response) { [bool]$Response.Aborted } else { $false }
        Operation = ''; Protocol = ''; Usage = $null; Outcome = 'completed'; NoGenAi = $false
        TraceId = ''; ParentSpanId = ''; Phases = @(); Stages = @{}; JobId = ''; JobState = ''
        ExpectLogs = $false; Marker = ''
    }
    foreach ($key in $Fields.Keys) { $exp.$key = $Fields[$key] }
    $Entry.Expectations.Add($exp)
    return $exp
}

# Prompts carry a marker (a prefix unique to this script run, plus a counter), so the run can
# prove that no exported row contains prompt text, and the content-capture run that one does.
function New-Prompt($Ctx, [string] $Suffix = '', [switch] $Video, [switch] $Long) {
    $script:PromptCount++
    $marker = '{0}P{1:D3}' -f $script:MarkerPrefix, $script:PromptCount
    $Ctx.MarkerSent = $true
    $Ctx.Markers.Add($marker)
    $Ctx.LastMarker = $marker
    if ($Video) { return "Telemetry check $marker$Suffix. A red ball rolls slowly across a wooden table in soft daylight." }
    # A reply long enough to still be streaming when a client drops it (the cancel checks).
    if ($Long) { return "Telemetry check $marker. Count from 1 to 500, one number per line, with no other text." }
    return "Telemetry check $marker$Suffix. Reply with one short sentence about the sea."
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

# ---------------------------------------------------------------------------------------------
# Scenarios, driving phase: send the requests, check the HTTP side, record what must arrive.
# ---------------------------------------------------------------------------------------------

function Invoke-LlmOpenAI($Ctx, $Entry, [switch] $Stream) {
    $check = $Entry.Check
    if (-not (Test-RequireModel $Ctx $check)) { return }
    $chat = Invoke-OpenAIChat -Ctx $Ctx -Stream:$Stream
    Save-Exchange $Ctx $Entry.Name $chat.Response
    if ($chat.Response.Status -ne 200) { $check.Issues.Add("HTTP $($chat.Response.Status): $(Get-ErrorText $chat.Response)"); return }
    if ($Stream -and -not $chat.SawDone) { $check.Issues.Add('the stream ended without data: [DONE]') }
    if (-not $chat.Usage) { $check.Issues.Add('the response has no usage block') }
    $check.Notes.Add("request id $($chat.Response.RequestId)")
    Add-Expectation $Entry $chat.Response @{
        Kind = 'gen'; Operation = 'chat'; Protocol = 'openai'; Usage = $chat.Usage; ExpectLogs = $true; Marker = $Ctx.LastMarker
    } | Out-Null
}

function Invoke-LlmOllamaGenerate($Ctx, $Entry) {
    $check = $Entry.Check
    if (-not (Test-RequireModel $Ctx $check)) { return }
    $body = [ordered]@{ model = $Ctx.Model; prompt = (New-Prompt $Ctx); stream = $false; options = @{ num_predict = $MaxTokens; temperature = 0 } }
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/api/generate" -Body $body -TimeoutSec $RequestTimeoutSec
    Save-Exchange $Ctx $Entry.Name $response
    if ($response.Status -ne 200) { $check.Issues.Add("HTTP $($response.Status): $(Get-ErrorText $response)"); return }
    $usage = [pscustomobject]@{ Input = [int]$response.Json.prompt_eval_count; Output = [int]$response.Json.eval_count; Cached = $null }
    Add-Expectation $Entry $response @{
        Kind = 'gen'; Operation = 'text_completion'; Protocol = 'ollama'; Usage = $usage; ExpectLogs = $true; Marker = $Ctx.LastMarker
    } | Out-Null
}

function Invoke-LlmOllamaChat($Ctx, $Entry) {
    $check = $Entry.Check
    if (-not (Test-RequireModel $Ctx $check)) { return }
    $body = [ordered]@{
        model    = $Ctx.Model
        messages = @(@{ role = 'user'; content = (New-Prompt $Ctx) })
        stream   = $false
        options  = @{ num_predict = $MaxTokens; temperature = 0 }
    }
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/api/chat/ollama" -Body $body -TimeoutSec $RequestTimeoutSec
    Save-Exchange $Ctx $Entry.Name $response
    if ($response.Status -ne 200) { $check.Issues.Add("HTTP $($response.Status): $(Get-ErrorText $response)"); return }
    $usage = [pscustomobject]@{ Input = [int]$response.Json.prompt_eval_count; Output = [int]$response.Json.eval_count; Cached = $null }
    Add-Expectation $Entry $response @{
        Kind = 'gen'; Operation = 'chat'; Protocol = 'ollama'; Usage = $usage; ExpectLogs = $true; Marker = $Ctx.LastMarker
    } | Out-Null
}

function Invoke-LlmWebUi($Ctx, $Entry) {
    $check = $Entry.Check
    if (-not (Test-RequireModel $Ctx $check)) { return }
    $body = [ordered]@{ messages = @(@{ role = 'user'; content = (New-Prompt $Ctx) }); maxTokens = $MaxTokens; newChat = $true }
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/api/chat" -Body $body -TimeoutSec $RequestTimeoutSec -Stream
    Save-Exchange $Ctx $Entry.Name $response
    if ($response.Status -ne 200) { $check.Issues.Add("HTTP $($response.Status): $(Get-ErrorText $response)"); return }
    $done = @(ConvertFrom-Sse $response.Lines | Where-Object { $_.done -eq $true }) | Select-Object -Last 1
    if (-not $done) { $check.Issues.Add('the event stream has no done frame'); return }
    if ($done.error) { $check.Issues.Add("the done frame reports an error: $($done.error)") }
    $usage = [pscustomobject]@{ Input = [int]$done.promptTokens; Output = [int]$done.tokenCount; Cached = [int]$done.kvReusedTokens }
    Add-Expectation $Entry $response @{
        Kind = 'gen'; Operation = 'chat'; Protocol = 'webui'; Usage = $usage; ExpectLogs = $true; Marker = $Ctx.LastMarker
    } | Out-Null
}

# A W3C traceparent from the client: the requests row must be in that operation, under that span.
function Invoke-LlmClientTraceparent($Ctx, $Entry) {
    $check = $Entry.Check
    if (-not (Test-RequireModel $Ctx $check)) { return }
    $traceId = New-RandomHex 16
    $parentId = New-RandomHex 8
    $chat = Invoke-OpenAIChat -Ctx $Ctx -Headers @{ traceparent = "00-$traceId-$parentId-01" }
    Save-Exchange $Ctx $Entry.Name $chat.Response
    if ($chat.Response.Status -ne 200) { $check.Issues.Add("HTTP $($chat.Response.Status): $(Get-ErrorText $chat.Response)"); return }
    $check.Notes.Add("sent traceparent 00-$traceId-$parentId-01")
    Add-Expectation $Entry $chat.Response @{
        Kind = 'gen'; Operation = 'chat'; Protocol = 'openai'; Usage = $chat.Usage; ExpectLogs = $true; Marker = $Ctx.LastMarker
        TraceId = $traceId; ParentSpanId = $parentId
    } | Out-Null
}

# Drop a stream mid-way: the chat dependency must end as cancelled and the engine must let go.
function Invoke-LlmCancel($Ctx, $Entry) {
    $check = $Entry.Check
    if (-not (Test-RequireModel $Ctx $check)) { return }
    $prompt = New-Prompt $Ctx -Long
    $chat = Invoke-OpenAIChat -Ctx $Ctx -Stream -Tokens 1024 -Prompt $prompt -AbortAfterChunks 3 -NoTools
    Save-Exchange $Ctx $Entry.Name $chat.Response
    if ($chat.Response.Status -ne 200) { $check.Issues.Add("HTTP $($chat.Response.Status): $(Get-ErrorText $chat.Response)"); return }
    if (-not $chat.Response.Aborted) { $check.Issues.Add('the stream finished before 3 chunks arrived, so nothing was cancelled'); return }
    if (@($chat.Response.Lines | Where-Object { $_ -match '"finish_reason":"[a-z_]+"' }).Count -gt 0) { $check.Issues.Add('the model finished its reply before the client disconnected, so there was nothing to cancel; the check needs a reply that is still streaming'); return }
    $check.Notes.Add("dropped the connection after $($chat.Response.Lines.Count) stream lines")
    Add-Expectation $Entry $chat.Response @{
        Kind = 'gen'; Operation = 'chat'; Protocol = 'openai'; Outcome = 'cancelled'; ExpectLogs = $true; Marker = $Ctx.LastMarker
    } | Out-Null
    $idle = Wait-Until -TimeoutSec 60 -Probe {
        $Ctx.QueueStatusPolls++
        $queue = Invoke-Http -Url "$($Ctx.Base)/api/queue/status" -TimeoutSec 10
        if ($queue.Json) { [pscustomobject]@{ Done = ([int]$queue.Json.processing -eq 0); Processing = [int]$queue.Json.processing } }
    }
    if (-not $idle -or -not $idle.Done) { $check.Issues.Add("the engine still reports processing=$($idle.Processing) 60 s after the client went away") }
}

function Invoke-LlmBadRequest($Ctx, $Entry) {
    $check = $Entry.Check
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/v1/chat/completions" -Body ([ordered]@{ model = [string]$Ctx.Model }) -TimeoutSec 60
    Save-Exchange $Ctx $Entry.Name $response
    if ($response.Status -ne 400) { $check.Issues.Add("expected HTTP 400 for a request without messages, got $($response.Status): $(Get-ErrorText $response)") }
    Add-Expectation $Entry $response @{ Kind = 'gen'; NoGenAi = $true; Protocol = 'openai'; ExpectLogs = $true } | Out-Null
}

# Eight requests at once, above the Azure sampler's default 5 traces/s: every one must arrive,
# each in its own operation, with a complete chat tree.
function Invoke-LlmBurst($Ctx, $Entry) {
    $check = $Entry.Check
    if (-not (Test-RequireModel $Ctx $check)) { return }
    $markers = @()
    $specs = foreach ($n in 1..8) {
        $prompt = New-Prompt $Ctx " (burst $n)"
        $markers += $Ctx.LastMarker
        @{
            Method  = 'POST'
            Url     = "$($Ctx.Base)/v1/chat/completions"
            Headers = @{}
            Body    = [ordered]@{
                model = $Ctx.Model; messages = @(@{ role = 'user'; content = $prompt })
                max_tokens = $MaxTokens; temperature = 0; stream = $false
            }
        }
    }
    $results = @(Invoke-HttpConcurrent $specs $RequestTimeoutSec)
    for ($i = 0; $i -lt $results.Count; $i++) { Save-Exchange $Ctx "llm-burst-$($i + 1)" $results[$i] }
    $ok = @($results | Where-Object { $_.Status -eq 200 })
    if ($ok.Count -ne $results.Count) {
        $bad = @($results | Where-Object { $_.Status -ne 200 })[0]
        $check.Issues.Add("$($results.Count - $ok.Count) of $($results.Count) requests failed (first: HTTP $($bad.Status) $(Get-ErrorText $bad))")
    }
    $ids = @($ok | ForEach-Object RequestId | Where-Object { $_ } | Select-Object -Unique)
    if ($ids.Count -ne $ok.Count) { $check.Issues.Add("only $($ids.Count) distinct request ids for $($ok.Count) responses") }
    for ($i = 0; $i -lt $results.Count; $i++) {
        if ($results[$i].Status -ne 200) { continue }
        Add-Expectation $Entry $results[$i] @{
            Kind = 'gen'; Operation = 'chat'; Protocol = 'openai'; Usage = (Get-OpenAIUsage $results[$i].Json.usage)
            Label = "request $($i + 1)"; ExpectLogs = $true; Marker = $markers[$i]
        } | Out-Null
    }
}

# The Web UI polls /api/queue/status; those requests must not become requests rows. A traced
# control request afterwards must arrive, so "no row" means filtered, not lost.
function Invoke-QueueStatusPoll($Ctx, $Entry) {
    $check = $Entry.Check
    $ids = [System.Collections.Generic.List[string]]::new()
    for ($i = 1; $i -le 20; $i++) {
        $Ctx.QueueStatusPolls++
        $response = Invoke-Http -Url "$($Ctx.Base)/api/queue/status" -TimeoutSec 30
        if ($response.Status -ne 200) { $check.Issues.Add("poll ${i}: HTTP $($response.Status) $(Get-ErrorText $response)"); return }
        if ($response.RequestId) { $ids.Add($response.RequestId) }
    }
    $check.Notes.Add('20 polls answered 200')
    $Entry.Extra['PollIds'] = $ids.ToArray()
    $control = Invoke-Http -Url "$($Ctx.Base)/api/models" -TimeoutSec 60
    Save-Exchange $Ctx 'queue-status-control' $control
    if ($control.Status -ne 200) { $check.Issues.Add("control GET /api/models: HTTP $($control.Status) $(Get-ErrorText $control)"); return }
    Add-Expectation $Entry $control @{ Label = 'control GET /api/models' } | Out-Null
}

function Invoke-Embeddings($Ctx, $Entry) {
    $check = $Entry.Check
    if (-not (Test-RequireModel $Ctx $check)) { return }
    $first = New-Prompt $Ctx ' first'
    $firstMarker = $Ctx.LastMarker
    $openai = Invoke-Http -Method POST -Url "$($Ctx.Base)/v1/embeddings" -TimeoutSec $RequestTimeoutSec -Body ([ordered]@{
            model = $Ctx.Model; input = @($first, 'A second, plain sentence.')
        })
    Save-Exchange $Ctx 'embeddings-openai' $openai
    if ($openai.Status -ne 200) {
        $check.Issues.Add("POST /v1/embeddings: HTTP $($openai.Status) $(Get-ErrorText $openai)")
    } else {
        if (@($openai.Json.data).Count -ne 2) { $check.Issues.Add("POST /v1/embeddings returned $(@($openai.Json.data).Count) vectors for 2 inputs") }
        $usage = [pscustomobject]@{ Input = [int]$openai.Json.usage.prompt_tokens; Output = $null; Cached = $null }
        Add-Expectation $Entry $openai @{
            Kind = 'gen'; Operation = 'embeddings'; Protocol = 'openai'; Usage = $usage; Label = 'openai'; ExpectLogs = $true; Marker = $firstMarker
        } | Out-Null
    }
    $third = New-Prompt $Ctx ' third'
    $ollama = Invoke-Http -Method POST -Url "$($Ctx.Base)/api/embed" -TimeoutSec $RequestTimeoutSec -Body ([ordered]@{ model = $Ctx.Model; input = $third })
    Save-Exchange $Ctx 'embeddings-ollama' $ollama
    if ($ollama.Status -ne 200) {
        $check.Issues.Add("POST /api/embed: HTTP $($ollama.Status) $(Get-ErrorText $ollama)")
    } else {
        $usage = [pscustomobject]@{ Input = [int]$ollama.Json.prompt_eval_count; Output = $null; Cached = $null }
        Add-Expectation $Entry $ollama @{
            Kind = 'gen'; Operation = 'embeddings'; Protocol = 'ollama'; Usage = $usage; Label = 'ollama'; ExpectLogs = $true; Marker = $Ctx.LastMarker
        } | Out-Null
    }
}

# Text-to-video through the OpenAI-shaped route (POST /v1/videos/generations; it blocks until the
# MP4 is written).
function Invoke-VideoGeneration($Ctx, [string] $Size, [int] $Frames, [string] $Label) {
    $body = [ordered]@{ prompt = (New-Prompt $Ctx -Video); size = $Size; frames = $Frames; seed = 42; response_format = 'url' }
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/v1/videos/generations" -Body $body -TimeoutSec $VideoTimeoutSec
    Save-Exchange $Ctx $Label $response
    return $response
}

function Invoke-VideoLtx($Ctx, $Entry) {
    $check = $Entry.Check
    $response = Invoke-VideoGeneration $Ctx $LtxSize $LtxFrames 'video-ltx'
    if ($response.Status -ne 200) { $check.Issues.Add("HTTP $($response.Status): $(Get-ErrorText $response)"); return }
    if (-not @($response.Json.data)[0].url) { $check.Issues.Add('the response has no data[0].url') }
    $check.Notes.Add("$($response.Json.frames) frames $($response.Json.width)x$($response.Json.height) in $($response.Json.elapsed_seconds) s")
    Add-Expectation $Entry $response @{
        Kind = 'video'; Label = 'clip'; Phases = @('text-encode', 'denoise', 'vae-decode'); Stages = @{ 'denoise' = $LtxExpectStages }
        ExpectLogs = $true; Marker = $Ctx.LastMarker
    } | Out-Null
}

# Two clips in a row: the first loads the DiT (dit-load cached=false), the second finds it loaded.
function Invoke-VideoMiniMaxH3($Ctx, $Entry) {
    $check = $Entry.Check
    for ($n = 1; $n -le 2; $n++) {
        $response = Invoke-VideoGeneration $Ctx $MiniMaxH3Size $MiniMaxH3Frames "video-minimax-h3-clip$n"
        if ($response.Status -ne 200) { $check.Issues.Add("clip ${n}: HTTP $($response.Status) $(Get-ErrorText $response)"); return }
        $check.Notes.Add("clip ${n}: $($response.Json.frames) frames $($response.Json.width)x$($response.Json.height) in $($response.Json.elapsed_seconds) s")
        Add-Expectation $Entry $response @{
            Kind = 'video'; Label = "clip $n"; Phases = @('text-encode', 'denoise', 'vae-decode'); ExpectLogs = $true; Marker = $Ctx.LastMarker
        } | Out-Null
    }
}

# Drop the progress stream of /api/video-generate/stream (the route that honours cancellation).
function Invoke-VideoCancel($Ctx, $Entry) {
    $endedBefore = Get-RequestEndCount $Ctx 'POST' '/api/video-generate/stream'
    $check = $Entry.Check
    $size = ConvertTo-Size $Ctx.Run.VideoSize
    $body = [ordered]@{ prompt = (New-Prompt $Ctx -Video); width = $size.Width; height = $size.Height; frames = $Ctx.Run.VideoFrames; seed = 7 }
    $response = Invoke-Http -Method POST -Url "$($Ctx.Base)/api/video-generate/stream" -Body $body -TimeoutSec $VideoTimeoutSec -Stream `
        -AbortAfterMatches 1 -AbortPattern '"phase"\s*:\s*"denoise"' -AbortAfterSec $VideoCancelAfterSec
    Save-Exchange $Ctx $Entry.Name $response
    if ($response.Status -ne 200) { $check.Issues.Add("HTTP $($response.Status): $(Get-ErrorText $response)"); return }
    if (-not $response.Aborted) {
        $done = @(ConvertFrom-Sse $response.Lines | Where-Object { $_.done -eq $true })[0]
        $check.Issues.Add("the generation ended before it could be cancelled ($(if ($done.error) { "error: $($done.error)" } else { 'completed' }))")
        return
    }
    $check.Notes.Add("dropped the stream after $($response.Lines.Count) lines ($([math]::Round($response.ElapsedMs / 1000, 1)) s)")
    Add-Expectation $Entry $response @{ Kind = 'video'; Label = 'cancelled clip'; Outcome = 'cancelled'; ExpectLogs = $true; Marker = $Ctx.LastMarker } | Out-Null
    # The clip runs on until it next checks its token (LTX: at its next progress report or sampling
    # step, and the denoiser upload before the first step can take tens of seconds). The next
    # scenario would compete with it for the GPU, memory and disk (Studio's ten-second ffprobe probe
    # timed out that way), so wait for its request to end first.
    if (-not (Wait-RequestEnded $Ctx 'POST' '/api/video-generate/stream' $endedBefore 600)) {
        $check.Issues.Add('the cancelled clip''s request had not ended 600 s after the client went away')
    }
    $health = Invoke-Http -Url "$($Ctx.Base)/health" -TimeoutSec 30
    $Ctx.Server.HealthPolls++
    if ($health.Status -ne 200) { $check.Issues.Add("the server stopped answering /health after the cancel (HTTP $($health.Status))") }
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

# A Studio job: an operation of its own, linked to the request that queued it.
function Invoke-StudioGenerate($Ctx, $Entry) {
    $check = $Entry.Check
    $capabilities = Get-StudioCapabilities $Ctx $check
    Save-Exchange $Ctx 'studio-capabilities' $capabilities
    if ($capabilities.Status -ne 200) { $check.Skip = "the Video Studio API is not reachable: HTTP $($capabilities.Status) $(Get-ErrorText $capabilities)"; return }
    $models = @($capabilities.Json.models | Where-Object { $_ })
    $model = @($models | Where-Object { $_.available -and $_.textToVideo })[0]
    if (-not $model) {
        $why = @(@($capabilities.Json.generationUnavailableReason) + @($models | ForEach-Object { "$($_.modelId): $($_.unavailableReason)" }) | Where-Object { $_ })
        $check.Skip = "Video Studio has no available text-to-video model ($($why -join '; '))"
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

    $project = Invoke-Studio $Ctx 'projects/create' ([ordered]@{ name = "appinsights-e2e $script:RunTag" })
    Save-Exchange $Ctx 'studio-project' $project
    if ($project.Status -ne 200 -or -not $project.Json.id) { $check.Issues.Add("projects/create: HTTP $($project.Status) $(Get-ErrorText $project)"); return }
    $projectId = [string]$project.Json.id
    $generate = Invoke-Studio $Ctx 'jobs/generate' ([ordered]@{
            projectId = $projectId; requestId = (New-RandomHex 16); modelId = [string]$model.modelId
            prompt = (New-Prompt $Ctx -Video); width = $width; height = $height; frameCount = $frames
        })
    Save-Exchange $Ctx 'studio-generate' $generate
    if ($generate.Status -ne 200 -or -not $generate.Json.id) { $check.Issues.Add("jobs/generate: HTTP $($generate.Status) $(Get-ErrorText $generate)"); return }
    $jobId = [string]$generate.Json.id
    $check.Notes.Add("job $jobId on $($model.modelId) ($width x $height, $frames frames; 0 = model default)")
    $Entry.Extra['Queue'] = Add-Expectation $Entry $generate @{ Label = 'queuing request' }

    $job = Wait-Until -TimeoutSec $VideoTimeoutSec -IntervalSec 5 -Probe {
        $list = Invoke-Studio $Ctx 'jobs/list' ([ordered]@{ projectId = $projectId })
        $item = @($list.Json | Where-Object { $_.id -eq $jobId })[0]
        if ($item) { [pscustomobject]@{ Done = ($item.state -notin 'queued', 'running', 'cancelling'); Job = $item } }
    }
    if (-not $job -or -not $job.Done) { $check.Issues.Add("the job did not finish within $VideoTimeoutSec s (state $($job.Job.state))"); return }
    if ($job.Job.state -ne 'completed') { $check.Issues.Add("the job ended '$($job.Job.state)': $($job.Job.error)") }
    $Entry.Extra['Job'] = Add-Expectation $Entry $null @{
        Kind = 'studio'; Label = 'job'; JobId = $jobId; JobState = [string]$job.Job.state; Marker = $Ctx.LastMarker
    }
}

# With OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true the prompt is exported.
function Invoke-ContentCapture($Ctx, $Entry) {
    $check = $Entry.Check
    if (-not (Test-RequireModel $Ctx $check)) { return }
    $chat = Invoke-OpenAIChat -Ctx $Ctx
    Save-Exchange $Ctx $Entry.Name $chat.Response
    if ($chat.Response.Status -ne 200) { $check.Issues.Add("HTTP $($chat.Response.Status): $(Get-ErrorText $chat.Response)"); return }
    Add-Expectation $Entry $chat.Response @{
        Kind = 'gen'; Operation = 'chat'; Protocol = 'openai'; Usage = $chat.Usage; ExpectLogs = $true; Marker = $Ctx.LastMarker
    } | Out-Null
}

# With TENSORSHARP_AZURE_MONITOR_AAD=1 the exporters authenticate with DefaultAzureCredential.
function Invoke-EntraId($Ctx, $Entry) {
    $check = $Entry.Check
    for ($i = 1; $i -le 3; $i++) {
        $response = Invoke-Http -Url "$($Ctx.Base)/api/models" -TimeoutSec 60
        Save-Exchange $Ctx "entra-id-$i" $response
        if ($response.Status -ne 200) { $check.Issues.Add("GET /api/models answered HTTP $($response.Status) $(Get-ErrorText $response)"); continue }
        Add-Expectation $Entry $response @{ Label = "GET /api/models #$i" } | Out-Null
    }
    if (-not $Ctx.Model) { return }
    $chat = Invoke-OpenAIChat -Ctx $Ctx
    Save-Exchange $Ctx 'entra-id-chat' $chat.Response
    if ($chat.Response.Status -ne 200) { $check.Issues.Add("POST /v1/chat/completions answered HTTP $($chat.Response.Status) $(Get-ErrorText $chat.Response)"); return }
    Add-Expectation $Entry $chat.Response @{
        Kind = 'gen'; Operation = 'chat'; Protocol = 'openai'; Usage = $chat.Usage; ExpectLogs = $true; Marker = $Ctx.LastMarker; Label = 'chat'
    } | Out-Null
}

function Invoke-Scenario($Ctx, [string] $Name) {
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $entry = New-PendingEntry $Name
    $Ctx.CurrentScenario = $Name
    Write-Step "  $($Ctx.Name): $Name"
    try {
        switch ($Name) {
            'llm-openai-stream' { Invoke-LlmOpenAI $Ctx $entry -Stream }
            'llm-openai' { Invoke-LlmOpenAI $Ctx $entry }
            'llm-ollama-generate' { Invoke-LlmOllamaGenerate $Ctx $entry }
            'llm-ollama-chat' { Invoke-LlmOllamaChat $Ctx $entry }
            'llm-webui' { Invoke-LlmWebUi $Ctx $entry }
            'llm-client-traceparent' { Invoke-LlmClientTraceparent $Ctx $entry }
            'llm-cancel' { Invoke-LlmCancel $Ctx $entry }
            'llm-bad-request' { Invoke-LlmBadRequest $Ctx $entry }
            'llm-burst' { Invoke-LlmBurst $Ctx $entry }
            'queue-status-poll' { Invoke-QueueStatusPoll $Ctx $entry }
            'embeddings' { Invoke-Embeddings $Ctx $entry }
            'video-ltx' { Invoke-VideoLtx $Ctx $entry }
            'video-minimax-h3' { Invoke-VideoMiniMaxH3 $Ctx $entry }
            'video-cancel' { Invoke-VideoCancel $Ctx $entry }
            'studio-generate' { Invoke-StudioGenerate $Ctx $entry }
            'content-capture' { Invoke-ContentCapture $Ctx $entry }
            'entra-id' { Invoke-EntraId $Ctx $entry }
            default { $entry.Check.Issues.Add("no implementation for scenario $Name") }
        }
    } catch {
        $entry.Check.Issues.Add("harness error: $($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))")
    }
    $entry.Seconds = $watch.Elapsed.TotalSeconds
    $Ctx.Pending.Add($entry)
}

# ---------------------------------------------------------------------------------------------
# Reading the run back from Application Insights
# ---------------------------------------------------------------------------------------------

function Get-AllExpectations($Ctx) {
    @($Ctx.Pending | ForEach-Object { $_.Expectations } | Where-Object { $_ })
}

# The operation (trace) id of a request: its requests row, else a dependency carrying its id.
function Get-ExpectationOperation($Exp, $Requests, $Dependencies) {
    $row = @($Requests | Where-Object { [string]$_.rid -eq $Exp.RequestId })[0]
    if ($row) { return [string]$row.operation_Id }
    $dep = @($Dependencies | Where-Object { [string]$_.rid -eq $Exp.RequestId })[0]
    if ($dep) { return [string]$dep.operation_Id }
    return ''
}

# What the dependencies table still lacks for the run's expectations: '' when complete.
function Get-DependencyGaps($Expectations, $Deps, $Requests) {
    $gaps = [System.Collections.Generic.List[string]]::new()
    foreach ($exp in $Expectations) {
        if ($exp.Kind -eq 'gen' -and -not $exp.NoGenAi -and $exp.Operation -and $exp.RequestId) {
            $gens = @($Deps | Where-Object { [string]$_.rid -eq $exp.RequestId -and [string]$_.genOperation -eq $exp.Operation -and [string]$_.isInternal -ne 'true' })
            if ($gens.Count -eq 0) {
                $gaps.Add("no '$($exp.Operation)' dependency for request $($exp.RequestId)")
            } elseif ($exp.Operation -in 'chat', 'text_completion') {
                $wait = if ($exp.Outcome -eq 'completed') { @('inference.decode', 'diffusion.denoise') } else { @('inference.queue', 'diffusion.denoise') }
                $genIds = @($gens | ForEach-Object { [string]$_.id })
                if (-not ($Deps | Where-Object { $genIds -contains [string]$_.operation_ParentId -and $wait -contains [string]$_.name })) {
                    $gaps.Add("the '$($exp.Operation)' dependency of request $($exp.RequestId) has no $($wait[0]) child yet")
                }
            }
        } elseif ($exp.Kind -eq 'video' -and $exp.RequestId) {
            $operation = Get-ExpectationOperation $exp $Requests $Deps
            if (-not $operation) {
                $gaps.Add("nothing yet for video request $($exp.RequestId)")
            } else {
                $names = @($Deps | Where-Object { [string]$_.operation_Id -eq $operation } | ForEach-Object { [string]$_.name })
                if (-not ($names | Where-Object { $_ -like 'generate_video *' })) { $gaps.Add("no generate_video dependency yet for request $($exp.RequestId)") }
                elseif ($exp.Outcome -eq 'completed' -and $names -notcontains 'video.export') { $gaps.Add("no video.export dependency yet for request $($exp.RequestId)") }
            }
        } elseif ($exp.Kind -eq 'studio' -and $exp.JobId) {
            $job = @($Deps | Where-Object { [string]$_.jobId -eq $exp.JobId -and ([string]$_.name).StartsWith('studio.job') })[0]
            if (-not $job) {
                $gaps.Add("no studio.job dependency yet for job $($exp.JobId)")
            } elseif ($exp.JobState -eq 'completed' -and -not ($Deps | Where-Object { [string]$_.operation_Id -eq [string]$job.operation_Id -and [string]$_.name -like 'generate_video *' })) {
                $gaps.Add("no generate_video dependency yet under job $($exp.JobId)")
            }
        }
    }
    if ($gaps.Count -eq 0) { return '' }
    return "$($gaps.Count) gap(s): " + (@($gaps | Select-Object -First 4) -join '; ')
}

# Metric instruments a run must have produced, by the traffic it sent.
function Get-ExpectedMetrics($Ctx) {
    $names = @()
    $expectations = Get-AllExpectations $Ctx
    $llm = [bool]($expectations | Where-Object { $_.Kind -eq 'gen' -and $_.Operation -in 'chat', 'text_completion' -and $_.Outcome -eq 'completed' })
    $video = [bool]($expectations | Where-Object { ($_.Kind -eq 'video' -and $_.Outcome -eq 'completed') -or ($_.Kind -eq 'studio' -and $_.JobState -eq 'completed') })
    # The engine and KV-cache gauges read only a live engine. A run shorter than the Azure Monitor
    # reader's fixed 60 s interval reports them only through the flush when the host stops, before
    # the models are disposed, so their presence checks that flush too.
    if ($llm) {
        $names += @('gen_ai.server.request.duration', 'gen_ai.server.time_to_first_token', 'tensorsharp.tokens',
            'tensorsharp.engine.requests.running', 'tensorsharp.engine.requests.waiting',
            'tensorsharp.kv_cache.blocks.used', 'tensorsharp.kv_cache.blocks.total',
            'tensorsharp.engine.steps', 'tensorsharp.engine.step.duration')
    }
    if ($video) { $names += @('tensorsharp.video.generation.duration', 'tensorsharp.video.phase.duration') }
    $backend = if ($script:RunInfo.Contains($Ctx.Name)) { [string]$script:RunInfo[$Ctx.Name]['loadedBackend'] } else { '' }
    if (($llm -or $video) -and $backend -match 'cuda|vulkan|metal|mlx') { $names += 'tensorsharp.gpu.memory.used' }
    return , $names
}

function Get-TokenTypes($Row) {
    @(ConvertFrom-AiDynamic $Row.tokenTypes | ForEach-Object { [string]$_ } | Where-Object { $_ })
}

# Query the run's tables until what it sent is there (or the deadline passes), then the absence
# checks once. Returns the rows for the checks.
function Get-RunTelemetry($Ctx) {
    $deadline = $Ctx.StopUtc.AddMinutes($IngestionTimeoutMinutes)
    $floor = [DateTime]::UtcNow.AddMinutes(2)
    if ($deadline -lt $floor) { $deadline = $floor }
    $queries = Get-RunQueries $Ctx
    $expectations = @(Get-AllExpectations $Ctx)
    $data = [pscustomobject]@{
        Requests = @(); Dependencies = @(); Traces = @(); Metrics = @(); Markers = @(); Instances = @()
        Queries = @{}; Deadline = $deadline
    }
    Write-Step "Run $($Ctx.Name): querying Application Insights for cloud_RoleName $($Ctx.Service) (until $($deadline.ToString('HH:mm:ss')) UTC at most)"

    $requestIds = @($expectations | Where-Object { $_.Kind -ne 'studio' -and $_.RequestId } | ForEach-Object RequestId | Select-Object -Unique)
    $poll = Invoke-AiPoll -Ctx $Ctx -Name 'requests' -Query $queries['requests'] -DeadlineUtc $deadline -Missing {
        param($Rows)
        $have = @($Rows | ForEach-Object { [string]$_.rid })
        $gap = @($requestIds | Where-Object { $have -notcontains $_ })
        if ($requestIds.Count -eq 0 -and @($Rows).Count -eq 0) { 'no requests rows yet' }
        elseif ($gap.Count) { "$($gap.Count) of $($requestIds.Count) request ids have no requests row yet" }
    }
    $data.Requests = @($poll.Rows); $data.Queries['requests'] = $poll

    $poll = Invoke-AiPoll -Ctx $Ctx -Name 'dependencies' -Query $queries['dependencies'] -DeadlineUtc $deadline -Missing {
        param($Rows)
        Get-DependencyGaps $expectations $Rows $data.Requests
    }
    $data.Dependencies = @($poll.Rows); $data.Queries['dependencies'] = $poll

    $logIds = @($expectations | Where-Object { $_.ExpectLogs -and $_.RequestId } | ForEach-Object RequestId | Select-Object -Unique)
    $poll = Invoke-AiPoll -Ctx $Ctx -Name 'traces' -Query $queries['traces'] -DeadlineUtc $deadline -Once:($logIds.Count -eq 0) -Missing {
        param($Rows)
        $have = @($Rows | Where-Object { [int]$_.rows -gt 0 } | ForEach-Object { [string]$_.rid })
        $gap = @($logIds | Where-Object { $have -notcontains $_ })
        if ($gap.Count) { "$($gap.Count) of $($logIds.Count) request ids have no traces row yet" }
    }
    $data.Traces = @($poll.Rows); $data.Queries['traces'] = $poll

    $metricNames = Get-ExpectedMetrics $Ctx
    if ($Selected -contains 'metrics' -and $metricNames.Count -gt 0) {
        $poll = Invoke-AiPoll -Ctx $Ctx -Name 'metrics' -Query $queries['metrics'] -DeadlineUtc $deadline -Missing {
            param($Rows)
            $have = @($Rows | ForEach-Object { [string]$_.name })
            $gap = @($metricNames | Where-Object { $have -notcontains $_ })
            $tokens = @($Rows | Where-Object { $_.name -eq 'tensorsharp.tokens' })[0]
            if ($tokens) {
                $types = Get-TokenTypes $tokens
                foreach ($type in 'input', 'output') { if ($types -notcontains $type) { $gap += "tensorsharp.tokens[$type]" } }
            }
            if ($gap.Count) { "missing $($gap -join ', ')" }
        }
        $data.Metrics = @($poll.Rows); $data.Queries['metrics'] = $poll
    }

    # The content-capture run waits for its marker to arrive; every other run asks once, after
    # its other rows arrived, and must find nothing.
    $captureMarkers = if ($Ctx.Run.Capture) { @($Ctx.Markers) } else { @() }
    $poll = Invoke-AiPoll -Ctx $Ctx -Name 'markers' -Query $queries['markers'] -DeadlineUtc $deadline -Once:(-not $Ctx.Run.Capture) -Missing {
        param($Rows)
        if ($captureMarkers.Count -and -not ($Rows | Where-Object { $_.sourceTable -eq 'traces' -and $captureMarkers -contains [string]$_.marker })) {
            'the prompt marker is not in traces yet'
        }
    }
    $data.Markers = @($poll.Rows); $data.Queries['markers'] = $poll

    $poll = Invoke-AiPoll -Ctx $Ctx -Name 'role-instance' -Query $queries['role-instance'] -DeadlineUtc $deadline -Once
    $data.Instances = @($poll.Rows); $data.Queries['role-instance'] = $poll
    return $data
}

# " (appinsights/02-dependencies.json; 9 queries; timed out ...)" for a FAIL detail.
function Get-QueryNote($Data, [string] $Name) {
    $query = $Data.Queries[$Name]
    if (-not $query) { return '' }
    $state = if (-not $query.Ok) { "query error: $($query.Error)" }
    elseif ($query.Ready) { 'complete' }
    elseif ($query.Missing) { "timed out at $($Data.Deadline.ToString('HH:mm:ss')) UTC: $($query.Missing)" }
    else { 'queried once' }
    if ($state.Length -gt 300) { $state = $state.Substring(0, 300) + '...' }
    return " ($($query.File); $($query.Attempts) quer$(if ($query.Attempts -eq 1) { 'y' } else { 'ies' }); $state)"
}

function Get-OperationIndex($Data, [string] $OperationId) {
    $rows = @($Data.Dependencies | Where-Object { [string]$_.operation_Id -eq $OperationId }) +
    @($Data.Requests | Where-Object { [string]$_.operation_Id -eq $OperationId })
    $byId = @{}
    $children = @{}
    foreach ($row in $rows) {
        $id = [string]$row.id
        if ($id) { $byId[$id] = $row }
        $parent = [string]$row.operation_ParentId
        if (-not $children.ContainsKey($parent)) { $children[$parent] = [System.Collections.Generic.List[object]]::new() }
        $children[$parent].Add($row)
    }
    [pscustomobject]@{ ById = $byId; Children = $children; Rows = $rows }
}

function Get-AiChildren($Index, $Row) {
    $id = [string]$Row.id
    if ($id -and $Index.Children.ContainsKey($id)) { $Index.Children[$id] }
}

function Get-AiDescendants($Index, $Row) {
    $queue = [System.Collections.Generic.Queue[object]]::new()
    $queue.Enqueue($Row)
    $seen = 0
    while ($queue.Count -gt 0 -and $seen -lt 100000) {
        $current = $queue.Dequeue()
        foreach ($child in @(Get-AiChildren $Index $current)) {
            $seen++
            $child
            $queue.Enqueue($child)
        }
    }
}

# Is $Row under the row with id $AncestorId, following operation_ParentId?
function Test-AiDescendant($Index, $Row, [string] $AncestorId) {
    $cursor = $Row
    for ($i = 0; $i -lt 128 -and $cursor; $i++) {
        $parent = [string]$cursor.operation_ParentId
        if (-not $parent) { return $false }
        if ($parent -eq $AncestorId) { return $true }
        $cursor = $Index.ById[$parent]
    }
    return $false
}

# ---------------------------------------------------------------------------------------------
# Checks on the rows
# ---------------------------------------------------------------------------------------------

function Test-AiRequestRow($Row, $Exp, $Check, [string] $Prefix) {
    $path = Get-UrlPath ([string]$Row.url)
    if ($Exp.Path -and $path -ne $Exp.Path -and -not ([string]$Row.name).EndsWith(" $($Exp.Path)")) {
        $Check.Issues.Add("${Prefix}the requests row is for '$($Row.name)' ($path), expected $($Exp.Path)")
    }
    if ($Exp.Status -gt 0 -and -not $Exp.Aborted -and [string]$Row.resultCode -ne [string]$Exp.Status) {
        $Check.Issues.Add("${Prefix}requests resultCode is '$($Row.resultCode)', the response was $($Exp.Status)")
    }
    if ($Exp.Protocol -and [string]$Row.protocol -ne $Exp.Protocol) {
        $Check.Issues.Add("${Prefix}requests customDimensions tensorsharp.protocol is '$($Row.protocol)', expected '$($Exp.Protocol)'")
    }
    if ($Exp.TraceId) {
        if ([string]$Row.operation_Id -ne $Exp.TraceId) { $Check.Issues.Add("${Prefix}the requests row's operation_Id is $($Row.operation_Id), not the client's trace $($Exp.TraceId)") }
        if ([string]$Row.operation_ParentId -ne $Exp.ParentSpanId) { $Check.Issues.Add("${Prefix}the requests row's operation_ParentId is '$($Row.operation_ParentId)', not the client span $($Exp.ParentSpanId)") }
    } elseif ($Row.operation_ParentId -and [string]$Row.operation_ParentId -ne [string]$Row.operation_Id) {
        $Check.Issues.Add("${Prefix}the requests row has a parent ($($Row.operation_ParentId)) although the request sent no traceparent")
    }
    $count = ConvertTo-NullableDouble $Row.itemCount
    if ($null -ne $count -and $count -ne 1) { $Check.Issues.Add("${Prefix}the requests row's itemCount is $count, expected 1 (100% sampling)") }
}

function Test-AiGenDependency($Gen, $Request, $Index, $Exp, $Check, [string] $Prefix) {
    $name = [string]$Gen.name
    if ($Request) {
        if ([string]$Gen.operation_Id -ne [string]$Request.operation_Id) { $Check.Issues.Add("${Prefix}'$name' is in operation $($Gen.operation_Id), the requests row in $($Request.operation_Id)") }
        elseif (-not (Test-AiDescendant $Index $Gen ([string]$Request.id))) { $Check.Issues.Add("${Prefix}'$name' is not under the requests row (operation_ParentId chain)") }
    }
    $model = [string]$Gen.genModel
    if (-not $model) { $Check.Issues.Add("${Prefix}'$name' has no gen_ai.request.model") }
    elseif ($name -ne "$($Exp.Operation) $model") { $Check.Issues.Add("${Prefix}dependency name '$name' is not '$($Exp.Operation) $model'") }
    if ([string]$Gen.genProvider -ne 'tensorsharp') { $Check.Issues.Add("${Prefix}gen_ai.provider.name is '$($Gen.genProvider)', expected tensorsharp") }
    if ($Exp.Protocol -and [string]$Gen.protocol -ne $Exp.Protocol) { $Check.Issues.Add("${Prefix}'$name' tensorsharp.protocol is '$($Gen.protocol)', expected '$($Exp.Protocol)'") }
    if ($Exp.Operation -ne 'embeddings' -and $Exp.Outcome -and [string]$Gen.outcome -ne $Exp.Outcome) {
        $Check.Issues.Add("${Prefix}tensorsharp.outcome is '$($Gen.outcome)', expected '$($Exp.Outcome)'")
    }
    if ($Gen.type -and [string]$Gen.type -ne 'InProc') { $Check.Issues.Add("${Prefix}'$name' has dependency type '$($Gen.type)', expected InProc") }
    $count = ConvertTo-NullableDouble $Gen.itemCount
    if ($null -eq $count -or $count -ne 1) { $Check.Issues.Add("${Prefix}'$name' itemCount is '$($Gen.itemCount)', expected 1 (100% sampling)") }

    $kids = @(Get-AiChildren $Index $Gen | ForEach-Object { [string]$_.name })
    if ($Exp.Operation -in 'chat', 'text_completion') {
        $required = if ($kids -contains 'diffusion.denoise') { @('prompt.render', 'diffusion.denoise') }
        elseif ($Exp.Outcome -eq 'completed') { @('prompt.render', 'inference.queue', 'inference.prefill', 'inference.decode') }
        else { @('inference.queue') }
        $missing = @($required | Where-Object { $kids -notcontains $_ })
        if ($missing) { $Check.Issues.Add("${Prefix}'$name' children (operation_ParentId = its id) are [$($kids -join ', ')]; missing [$($missing -join ', ')]") }
        if ($Exp.Outcome -eq 'completed') {
            $ttft = ConvertTo-NullableDouble $Gen.ttftMs
            if ($null -eq $ttft -or $ttft -le 0) { $Check.Issues.Add("${Prefix}'$name' has no positive tensorsharp.ttft_ms ('$($Gen.ttftMs)')") }
            else { $Check.Notes.Add("${Prefix}ttft $([math]::Round($ttft, 1)) ms") }
        }
    }
    $Check.Notes.Add("${Prefix}'$name' children: $($kids -join ', ')")
}

# Token counts in customDimensions (strings, read with todouble) must equal the response's
# usage block. Several dependencies (tool rounds) are summed.
function Test-AiTokenUsage($Gens, $Usage, $Check, [string] $Operation, [string] $Prefix) {
    $isChat = $Operation -in 'chat', 'text_completion'
    $pairs = @(
        @{ Column = 'inputTokens'; Key = 'gen_ai.usage.input_tokens'; Expected = $Usage.Input; Name = 'input' },
        @{ Column = 'outputTokens'; Key = 'gen_ai.usage.output_tokens'; Expected = $Usage.Output; Name = 'output' },
        @{ Column = 'cacheReadTokens'; Key = 'gen_ai.usage.cache_read.input_tokens'; Expected = $Usage.Cached; Name = 'cache_read'; ZeroWhenMissing = $true }
    )
    $shown = @()
    foreach ($pair in $pairs) {
        if ($null -eq $pair.Expected) { continue }
        $sum = 0.0
        $missing = 0
        foreach ($gen in $Gens) {
            $value = ConvertTo-NullableDouble $gen.($pair.Column)
            if ($null -eq $value) { $missing++ } else { $sum += $value }
        }
        if ($missing -eq @($Gens).Count -and -not $pair.ZeroWhenMissing) {
            if ($isChat) { $Check.Issues.Add("${Prefix}no $($pair.Key) in customDimensions") } else { $Check.Notes.Add("${Prefix}no $($pair.Key) in customDimensions") }
            continue
        }
        if ([int]$sum -ne [int]$pair.Expected) { $Check.Issues.Add("${Prefix}customDimensions $($pair.Key) = $sum, the response says $($pair.Expected)") }
        $shown += "$($pair.Name) $([int]$sum)"
    }
    if ($shown) { $Check.Notes.Add("${Prefix}tokens match the response ($($shown -join ', '))") }
}

# traces rows (log records) carrying the request id, in the request's operation.
function Test-AiRequestLogs($Data, [string] $RequestId, [string] $OperationId, $Check, [string] $Prefix) {
    $mine = @($Data.Traces | Where-Object { [string]$_.rid -eq $RequestId })
    $total = 0
    foreach ($group in $mine) { $total += [int]$group.rows }
    if ($total -eq 0) { $Check.Issues.Add("${Prefix}no traces row carries customDimensions tensorsharp.request_id=$RequestId$(Get-QueryNote $Data 'traces')"); return }
    if ($OperationId) {
        $inOperation = @($mine | Where-Object { [string]$_.operation_Id -eq $OperationId })
        if ($inOperation.Count -eq 0) {
            $Check.Issues.Add("${Prefix}the $total traces row(s) with the request id are in operation(s) $((@($mine | ForEach-Object operation_Id | Select-Object -Unique)) -join ', '), not the request's $OperationId")
        }
        $other = @($Data.Traces | Where-Object { [string]$_.operation_Id -eq $OperationId -and [string]$_.rid -and [string]$_.rid -ne $RequestId })
        if ($other.Count -gt 0) { $Check.Issues.Add("${Prefix}traces rows in the request's operation carry another tensorsharp.request_id ($($other[0].rid))") }
    }
    $Check.Notes.Add("${Prefix}$total traces row(s) carry the request id")
}

# One request: its requests row, its generation dependency and children, tokens, and logs.
function Test-AiRequest($Ctx, $Data, $Exp, $Check) {
    $prefix = if ($Exp.Label) { "$($Exp.Label): " } else { '' }
    $id = $Exp.RequestId
    if (-not $id) { $Check.Issues.Add("${prefix}the response has no X-TensorSharp-Request-Id header"); return $null }
    $rows = @($Data.Requests | Where-Object { [string]$_.rid -eq $id })
    $request = $null
    if ($rows.Count -eq 0) {
        $Check.Issues.Add("${prefix}no requests row with customDimensions tensorsharp.request_id=$id$(Get-QueryNote $Data 'requests')")
    } else {
        if ($rows.Count -gt 1) { $Check.Issues.Add("${prefix}$($rows.Count) requests rows carry tensorsharp.request_id=$id (exported more than once?)") }
        $request = $rows[0]
        Test-AiRequestRow $request $Exp $Check $prefix
    }
    $gens = @()
    if ($Exp.Kind -eq 'gen') {
        $withId = @($Data.Dependencies | Where-Object { [string]$_.rid -eq $id })
        $anyGen = @($withId | Where-Object { $_.genOperation })
        if ($Exp.NoGenAi) {
            if ($anyGen.Count -gt 0) { $Check.Issues.Add("${prefix}unexpected '$($anyGen[0].name)' dependency for a request that generated nothing") }
        } else {
            $gens = @($anyGen | Where-Object { [string]$_.genOperation -eq $Exp.Operation -and [string]$_.isInternal -ne 'true' })
            if ($gens.Count -eq 0) {
                $names = @($withId | ForEach-Object { [string]$_.name } | Select-Object -Unique)
                $Check.Issues.Add("${prefix}no '$($Exp.Operation) <model>' dependency with tensorsharp.request_id=$id (dependencies with that id: $($names -join ', '))$(Get-QueryNote $Data 'dependencies')")
            } else {
                $index = Get-OperationIndex $Data ([string]$gens[0].operation_Id)
                # A tool-calling reply has one dependency per round (the query orders them by
                # start). Only the last round can end the way the request did; a round that
                # handed over to a tool call completed.
                for ($i = 0; $i -lt $gens.Count; $i++) {
                    $roundExp = $Exp
                    if ($i -lt $gens.Count - 1 -and $Exp.Outcome -and $Exp.Outcome -ne 'completed') {
                        $roundExp = $Exp.psobject.Copy(); $roundExp.Outcome = 'completed'
                    }
                    Test-AiGenDependency $gens[$i] $request $index $roundExp $Check $prefix
                }
                if ($Exp.Usage) { Test-AiTokenUsage $gens $Exp.Usage $Check $Exp.Operation $prefix }
            }
        }
    }
    if ($Exp.ExpectLogs) {
        $operation = if ($request) { [string]$request.operation_Id } elseif ($gens.Count) { [string]$gens[0].operation_Id } else { '' }
        Test-AiRequestLogs $Data $id $operation $Check $prefix
    }
    return [pscustomobject]@{ Request = $request; Gens = $gens }
}

# A video generation: generate_video under the requests row, its phases and stages, video.export.
function Test-AiVideo($Ctx, $Data, $Exp, $Check) {
    $prefix = if ($Exp.Label) { "$($Exp.Label): " } else { '' }
    $id = $Exp.RequestId
    if (-not $id) { $Check.Issues.Add("${prefix}the response has no X-TensorSharp-Request-Id header"); return $null }
    $request = @($Data.Requests | Where-Object { [string]$_.rid -eq $id })[0]
    if ($request) { Test-AiRequestRow $request $Exp $Check $prefix }
    else { $Check.Issues.Add("${prefix}no requests row with customDimensions tensorsharp.request_id=$id$(Get-QueryNote $Data 'requests')") }
    $operation = Get-ExpectationOperation $Exp $Data.Requests $Data.Dependencies
    if (-not $operation) { $Check.Issues.Add("${prefix}no row of this request reached Application Insights"); return $null }
    $index = Get-OperationIndex $Data $operation
    $gen = @($index.Rows | Where-Object { [string]$_.name -like 'generate_video *' })[0]
    if (-not $gen) {
        $names = @($index.Rows | ForEach-Object { [string]$_.name } | Select-Object -Unique)
        $Check.Issues.Add("${prefix}no 'generate_video <model>' dependency in operation $operation (rows: $($names -join ', '))$(Get-QueryNote $Data 'dependencies')")
        return $null
    }
    if ($request -and -not (Test-AiDescendant $index $gen ([string]$request.id))) { $Check.Issues.Add("${prefix}'$($gen.name)' is not under the requests row") }
    $descendants = @(Get-AiDescendants $index $gen)
    $Check.Notes.Add("${prefix}under '$($gen.name)': $((@($descendants | ForEach-Object name | Select-Object -Unique)) -join ', ')")
    $count = ConvertTo-NullableDouble $gen.itemCount
    if ($null -ne $count -and $count -ne 1) { $Check.Issues.Add("${prefix}'$($gen.name)' itemCount is $count, expected 1") }

    $outcome = [string]$gen.outcome
    if ($Exp.Outcome -eq 'cancelled') {
        if ($outcome -ne 'cancelled' -and [string]$gen.errorType -notmatch 'Cancel') {
            $Check.Issues.Add("${prefix}'$($gen.name)' outcome is '$outcome' (error.type '$($gen.errorType)'), expected cancelled")
        }
    } elseif ($outcome -and $outcome -ne $Exp.Outcome) {
        $Check.Issues.Add("${prefix}'$($gen.name)' tensorsharp.outcome is '$outcome', expected '$($Exp.Outcome)'")
    }
    foreach ($phase in @($Exp.Phases)) {
        $phaseRows = @($descendants | Where-Object { [string]$_.name -eq $phase })
        if ($phaseRows.Count -eq 0) { $Check.Issues.Add("${prefix}no '$phase' phase dependency under '$($gen.name)'"); continue }
        if ($Exp.Stages -and $Exp.Stages.ContainsKey($phase)) {
            $stageNames = @($phaseRows | ForEach-Object { Get-AiDescendants $index $_ } | ForEach-Object { [string]$_.name } | Select-Object -Unique)
            $missing = @($Exp.Stages[$phase] | Where-Object { $stageNames -notcontains $_ })
            if ($missing) { $Check.Issues.Add("${prefix}'$phase' has no stage dependency $($missing -join ', ') (it has: $($stageNames -join ', '))") }
        }
    }
    if ($Exp.Outcome -eq 'completed' -and -not ($descendants | Where-Object { [string]$_.name -eq 'video.export' })) {
        $Check.Issues.Add("${prefix}no video.export dependency under '$($gen.name)'")
    }
    if ($Exp.ExpectLogs) { Test-AiRequestLogs $Data $id $operation $Check $prefix }
    return [pscustomobject]@{ Request = $request; Gen = $gen; Descendants = $descendants }
}

function Test-AiMiniMaxH3($Ctx, $Data, $Entry, $Check) {
    $exps = @($Entry.Expectations)
    $clip1 = if ($exps.Count -ge 1) { Test-AiVideo $Ctx $Data $exps[0] $Check } else { $null }
    $clip2 = if ($exps.Count -ge 2) { Test-AiVideo $Ctx $Data $exps[1] $Check } else { $null }
    if (-not $clip1 -or -not $clip2) { return }
    $dit1 = @($clip1.Descendants | Where-Object { $_.name -eq 'dit-load' })[0]
    $dit2 = @($clip2.Descendants | Where-Object { $_.name -eq 'dit-load' })[0]
    $cached1 = if ($dit1) { [string]$dit1.videoCached } else { '' }
    $cached2 = if ($dit2) { [string]$dit2.videoCached } else { '' }
    if (-not $dit1) { $Check.Issues.Add('clip 1 has no dit-load stage dependency') }
    elseif ($cached1 -ne 'false') { $Check.Issues.Add("clip 1 dit-load has tensorsharp.video.cached='$cached1', expected false") }
    $released = [bool]($clip1.Descendants | Where-Object { $_.name -eq 'denoiser-release' })
    if (-not $dit2) {
        $Check.Issues.Add('clip 2 has no dit-load stage dependency')
    } elseif ($cached2 -ne 'true') {
        if ($released) {
            $Check.Manual = "clip 2 reloaded the DiT (cached=$cached2) because clip 1 released it (denoiser-release), so the cached path was not exercised on this host"
        } else {
            $Check.Issues.Add("clip 2 dit-load has tensorsharp.video.cached='$cached2', expected true")
        }
    } else {
        $Check.Notes.Add('dit-load cached: clip 1 false, clip 2 true')
    }
}

function Test-AiStudio($Ctx, $Data, $Entry, $Check) {
    $queueExp = $Entry.Extra['Queue']
    $jobExp = $Entry.Extra['Job']
    $queued = if ($queueExp) { Test-AiRequest $Ctx $Data $queueExp $Check } else { $null }
    if (-not $jobExp) { return }
    $jobs = @($Data.Dependencies | Where-Object { [string]$_.jobId -eq $jobExp.JobId -and ([string]$_.name).StartsWith('studio.job') })
    if ($jobs.Count -eq 0) { $Check.Issues.Add("no studio.job dependency with tensorsharp.studio.job_id=$($jobExp.JobId)$(Get-QueryNote $Data 'dependencies')"); return }
    $job = $jobs[0]
    if ($job.name -ne 'studio.job generate') { $Check.Issues.Add("the job dependency is named '$($job.name)', expected 'studio.job generate'") }
    if ($job.operation_ParentId -and [string]$job.operation_ParentId -ne [string]$job.operation_Id) {
        $Check.Issues.Add("the job dependency has a parent ($($job.operation_ParentId)); it should start an operation of its own")
    }
    if ([string]$job.outcome -ne $jobExp.JobState) { $Check.Issues.Add("job tensorsharp.outcome is '$($job.outcome)', the job is '$($jobExp.JobState)'") }
    $count = ConvertTo-NullableDouble $job.itemCount
    if ($null -ne $count -and $count -ne 1) { $Check.Issues.Add("the job dependency's itemCount is $count, expected 1") }
    $index = Get-OperationIndex $Data ([string]$job.operation_Id)
    $children = @(Get-AiChildren $index $job | ForEach-Object { [string]$_.name })
    if ($children -notcontains 'studio.queue') { $Check.Issues.Add("the job has no studio.queue child (children: $($children -join ', '))") }
    if (-not ($children | Where-Object { $_ -like 'studio.stage *' })) { $Check.Issues.Add('the job has no studio.stage children') }
    $descendants = @(Get-AiDescendants $index $job)
    if ($jobExp.JobState -eq 'completed' -and -not ($descendants | Where-Object { [string]$_.name -like 'generate_video *' })) {
        $Check.Issues.Add('no generate_video dependency under the job')
    }
    $Check.Notes.Add("job children: $($children -join ', ')")
    if ($queued -and $queued.Request) {
        $queuedOperation = [string]$queued.Request.operation_Id
        if ($queuedOperation -eq [string]$job.operation_Id) { $Check.Issues.Add('the job is in the queuing request''s operation instead of its own') }
        $links = @(ConvertFrom-AiDynamic $job.links | Where-Object { $_ })
        if (-not ($links | Where-Object { [string]$_.operation_Id -eq $queuedOperation })) {
            $Check.Issues.Add("customDimensions[`"_MS.links`"] of the job has no link to the queuing request's operation $queuedOperation (links: $([string]$job.links))")
        }
    }
}

function Test-AiBurst($Ctx, $Data, $Entry, $Check) {
    $operations = [System.Collections.Generic.HashSet[string]]::new()
    $arrived = 0
    foreach ($exp in $Entry.Expectations) {
        $found = Test-AiRequest $Ctx $Data $exp $Check
        if ($found -and $found.Request) { $arrived++; [void]$operations.Add([string]$found.Request.operation_Id) }
    }
    if ($arrived -gt 0 -and $operations.Count -ne $arrived) { $Check.Issues.Add("$arrived requests rows are in $($operations.Count) distinct operations") }
    $Check.Notes.Add("$arrived of $($Entry.Expectations.Count) concurrent requests have a requests row (no rate-limit sampling)")
}

function Test-AiQueueStatusPoll($Ctx, $Data, $Entry, $Check) {
    foreach ($exp in $Entry.Expectations) { Test-AiRequest $Ctx $Data $exp $Check | Out-Null }
    $pollIds = @($Entry.Extra['PollIds'])
    $byId = @($Data.Requests | Where-Object { $pollIds -contains [string]$_.rid })
    $byPath = @($Data.Requests | Where-Object { (Get-UrlPath ([string]$_.url)) -eq '/api/queue/status' -or ([string]$_.name).EndsWith(' /api/queue/status') })
    if ($byId.Count -gt 0) { $Check.Issues.Add("$($byId.Count) requests row(s) carry the request id of a queue-status poll") }
    if ($byPath.Count -gt 0) { $Check.Issues.Add("$($byPath.Count) requests row(s) for /api/queue/status were exported") }
    $Check.Notes.Add("searched $($Data.Requests.Count) requests rows for $($pollIds.Count) poll ids")
}

function Test-AiContentCapture($Ctx, $Data, $Entry, $Check) {
    foreach ($exp in $Entry.Expectations) { Test-AiRequest $Ctx $Data $exp $Check | Out-Null }
    $markers = @($Entry.Expectations | ForEach-Object Marker | Where-Object { $_ })
    if ($markers.Count -eq 0) { return }
    $hits = @($Data.Markers | Where-Object { $markers -contains [string]$_.marker })
    $inTraces = @($hits | Where-Object { $_.sourceTable -eq 'traces' })
    if ($inTraces.Count -eq 0) { $Check.Issues.Add("the prompt marker $($markers -join ', ') never appeared in traces with content capture on$(Get-QueryNote $Data 'markers')") }
    else { $Check.Notes.Add("the prompt appears in $($inTraces.Count) traces row(s)") }
    $tables = @($hits | ForEach-Object sourceTable | Select-Object -Unique)
    if ($tables) { $Check.Notes.Add("marker found in: $($tables -join ', ')") }
    if ($tables -contains 'dependencies') { $Check.Notes.Add('the prompt is also on dependencies (gen_ai.input.messages)') }
}

function Test-AiEntraId($Ctx, $Data, $Entry, $Check) {
    $exps = @($Entry.Expectations)
    $arrived = @($exps | Where-Object { $id = $_.RequestId; $id -and ($Data.Requests | Where-Object { [string]$_.rid -eq $id }) })
    if ($exps.Count -eq 0) { return }
    if ($arrived.Count -eq 0) {
        $Check.Issues.Add(("no requests row arrived under $($Ctx.Service) with TENSORSHARP_AZURE_MONITOR_AAD=1$(Get-QueryNote $Data 'requests'), so token-based ingestion did not work. " +
                'A missing Monitoring Metrics Publisher role for the identity DefaultAzureCredential picked in the server, or a role assignment still propagating (it can take several minutes), looks exactly like this'))
    } elseif ($arrived.Count -lt $exps.Count) {
        $Check.Issues.Add("only $($arrived.Count) of $($exps.Count) requests arrived with Entra ID authentication")
    } else {
        $Check.Notes.Add("all $($exps.Count) requests arrived with Entra ID authentication (rows: $($Data.Requests.Count))")
    }
    # The chat request (with a chat model): its dependency tree, token counts and log rows came
    # through the same token, which is what proves the trace and log exporters used it too.
    foreach ($exp in @($exps | Where-Object { $_.Kind -eq 'gen' -and -not $_.NoGenAi })) { Test-AiRequest $Ctx $Data $exp $Check | Out-Null }
    $metricRows = @($Data.Metrics | Where-Object { [string]$_.name -like 'tensorsharp.*' -or [string]$_.name -like 'gen_ai.*' })
    # Metrics are queried only when the metrics scenario runs.
    if ($Data.Queries.ContainsKey('metrics') -and @($exps | Where-Object { $_.Kind -eq 'gen' -and -not $_.NoGenAi }).Count -gt 0) {
        if ($metricRows.Count -eq 0) { $Check.Issues.Add("no TensorSharp customMetrics rows arrived with Entra ID authentication$(Get-QueryNote $Data 'metrics')") }
        else { $Check.Notes.Add("$($metricRows.Count) TensorSharp metric names arrived with Entra ID authentication") }
    }
    $messages = @($Ctx.ExporterMessages)
    if ($messages.Count -gt 0) {
        $Check.Notes.Add("exporter/credential messages in the server output ($($messages.Count), all in exporter-messages.txt): $((@($messages | Select-Object -First 3 | ForEach-Object { if ($_.Length -gt 300) { $_.Substring(0, 300) + '...' } else { $_ } })) -join ' || ')")
    } else {
        $Check.Notes.Add('no exporter or credential warning in the server output (the exporter reports through EventSource; -ExporterDiagnostics captures it)')
    }
}

function Test-PendingEntry($Ctx, $Data, $Entry) {
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $check = $Entry.Check
    if (-not $check.Skip) {
        try {
            switch ($Entry.Name) {
                'video-minimax-h3' { Test-AiMiniMaxH3 $Ctx $Data $Entry $check }
                'studio-generate' { Test-AiStudio $Ctx $Data $Entry $check }
                'llm-burst' { Test-AiBurst $Ctx $Data $Entry $check }
                'queue-status-poll' { Test-AiQueueStatusPoll $Ctx $Data $Entry $check }
                'content-capture' { Test-AiContentCapture $Ctx $Data $Entry $check }
                'entra-id' { Test-AiEntraId $Ctx $Data $Entry $check }
                default {
                    foreach ($exp in $Entry.Expectations) {
                        if ($exp.Kind -eq 'video') { Test-AiVideo $Ctx $Data $exp $check | Out-Null }
                        else { Test-AiRequest $Ctx $Data $exp $check | Out-Null }
                    }
                }
            }
        } catch {
            $check.Issues.Add("harness error: $($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))")
        }
    }
    Complete-Check $Ctx.Name $Entry.Name $check ($Entry.Seconds + $watch.Elapsed.TotalSeconds)
}

# Every request the run drove (except the deliberately untraced ones) has a requests row.
function Test-AiRequestRows($Ctx, $Data, $Check) {
    $exps = @(Get-AllExpectations $Ctx | Where-Object { $_.Kind -ne 'studio' -and $_.RequestId })
    $missing = @($exps | Where-Object { $id = $_.RequestId; -not ($Data.Requests | Where-Object { [string]$_.rid -eq $id }) })
    if ($missing.Count -gt 0) {
        $shown = @($missing | Select-Object -First 6 | ForEach-Object { "$($_.Scenario) $($_.Method) $($_.Path) ($($_.RequestId))" })
        $Check.Issues.Add("$($missing.Count) of $($exps.Count) driven requests have no requests row: $($shown -join '; ')$(Get-QueryNote $Data 'requests')")
    }
    $routes = @($Data.Requests | Group-Object { [string]$_.name } | Sort-Object Count -Descending | ForEach-Object { "$($_.Name) x$($_.Count)" })
    $Check.Notes.Add("$($exps.Count - $missing.Count) of $($exps.Count) driven requests found; routes: $($routes -join ', ')")
}

function Test-AiUntracedPaths($Ctx, $Data, $Check) {
    if ($Data.Requests.Count -eq 0) { $Check.Issues.Add("no requests row arrived at all, so the absence of /health and /api/queue/status rows proves nothing$(Get-QueryNote $Data 'requests')"); return }
    foreach ($path in '/health', '/api/queue/status') {
        $rows = @($Data.Requests | Where-Object { (Get-UrlPath ([string]$_.url)) -eq $path -or ([string]$_.name).EndsWith(" $path") })
        if ($rows.Count -gt 0) { $Check.Issues.Add("$($rows.Count) requests row(s) for $path were exported") }
    }
    $Check.Notes.Add("the harness polled /health $($Ctx.Server.HealthPolls) and /api/queue/status $($Ctx.QueueStatusPolls) times; $($Data.Requests.Count) requests rows searched after the run's other rows arrived")
}

function Test-AiRoleInstance($Ctx, $Data, $Check) {
    $expected = "$($script:Machine):$($Ctx.Server.Pid)"
    $rows = @($Data.Instances)
    if (-not $Data.Queries['role-instance'].Ok) { $Check.Issues.Add("the query failed$(Get-QueryNote $Data 'role-instance')"); return }
    if ($rows.Count -eq 0) { $Check.Issues.Add("no row of $($Ctx.Service) arrived$(Get-QueryNote $Data 'role-instance')"); return }
    $wrong = @($rows | Where-Object { [string]$_.cloud_RoleInstance -ne $expected })
    foreach ($row in $wrong) { $Check.Issues.Add("$($row.rows) $($row.itemType) row(s) have cloud_RoleInstance '$($row.cloud_RoleInstance)', expected '$expected' (machine:pid)") }
    $breakdown = @($rows | ForEach-Object { "$($_.itemType) $($_.rows)" })
    $Check.Notes.Add("cloud_RoleInstance $expected; rows by type: $($breakdown -join ', ')")
}

# No row of a run without content capture contains a prompt marker.
function Test-AiRedaction($Ctx, $Data, $Check) {
    if (-not $Ctx.MarkerSent) { $Check.Skip = 'no request in this run carried prompt text'; return }
    $query = $Data.Queries['markers']
    if (-not $query.Ok) { $Check.Issues.Add("the marker query failed$(Get-QueryNote $Data 'markers')"); return }
    $hits = @($Data.Markers)
    if ($hits.Count -gt 0) {
        $shown = @($hits | Select-Object -First 6 | ForEach-Object { "$($_.sourceTable) '$($_.name)' ($($_.marker))" })
        $Check.Issues.Add("the prompt marker is in $($hits.Count) exported row(s): $($shown -join '; ')")
    }
    $traceRows = 0
    foreach ($group in $Data.Traces) { $traceRows += [int]$group.rows }
    if ($traceRows -eq 0) { $Check.Issues.Add("no traces row arrived for the run, so redaction was not exercised$(Get-QueryNote $Data 'traces')") }
    $Check.Notes.Add("searched name, url, data, message, exception text and customDimensions of requests, dependencies, traces and exceptions for $($Ctx.Markers.Count) markers ($script:MarkerPrefix...); $traceRows traces rows in the run")
}

function Test-AiMetrics($Ctx, $Data, $Check) {
    $expected = Get-ExpectedMetrics $Ctx
    $rows = @($Data.Metrics)
    $have = @($rows | ForEach-Object { [string]$_.name })
    $missing = @($expected | Where-Object { $have -notcontains $_ })
    if ($missing) { $Check.Issues.Add("no customMetrics rows named $($missing -join ', ')$(Get-QueryNote $Data 'metrics')") }
    $tokens = @($rows | Where-Object { $_.name -eq 'tensorsharp.tokens' })[0]
    if ($expected -contains 'tensorsharp.tokens' -and $tokens) {
        $types = Get-TokenTypes $tokens
        $missingTypes = @('input', 'output' | Where-Object { $types -notcontains $_ })
        if ($missingTypes) { $Check.Issues.Add("tensorsharp.tokens has tensorsharp.token.type [$($types -join ', ')]; missing [$($missingTypes -join ', ')]") }
        else { $Check.Notes.Add("tensorsharp.tokens types: $($types -join ', '); sum $($tokens.valueSum)") }
    }
    $shown = @($have | Select-Object -First 40)
    $Check.Notes.Add("$($have.Count) metric names from the run: $($shown -join ', ')$(if ($have.Count -gt 40) { ', ...' })")
}

# With -BothSinks: the dashboard must hold the same requests as Application Insights.
function Test-AiBothSinks($Ctx, $Check) {
    $ids = @(Get-AllExpectations $Ctx | Where-Object { $_.Kind -ne 'studio' -and $_.RequestId } | ForEach-Object RequestId | Select-Object -Unique)
    if (-not $DashboardApiKey) {
        $Check.Manual = "open $DashboardUrl, pick resource $($Ctx.Service) and confirm its spans carry the run's $($ids.Count) request ids (in runs/$($Ctx.Name)/requests-sent.json); pass -DashboardApiKey to check this automatically"
        return
    }
    $headers = @{ 'X-API-Key' = $DashboardApiKey }
    $base = $DashboardUrl.TrimEnd('/')
    $resources = Invoke-Http -Url "$base/api/telemetry/resources" -Headers $headers -TimeoutSec 60
    if ($resources.Status -ne 200) {
        $Check.Manual = "the dashboard telemetry API answered HTTP $($resources.Status) $($resources.Error); check resource $($Ctx.Service) in the dashboard UI"
        return
    }
    if (-not (@($resources.Json) | Where-Object { $_ -and $_.name -eq $Ctx.Service })) { $Check.Issues.Add("the dashboard lists no resource $($Ctx.Service): nothing arrived over OTLP"); return }
    $spans = Invoke-Http -Url "$base/api/telemetry/spans?resource=$([uri]::EscapeDataString($Ctx.Service))&limit=1000000" -Headers $headers -TimeoutSec 120
    Save-Text (Join-Path $Ctx.Dir 'otlp/spans.json') ([string]$spans.Body)
    if ($spans.Status -ne 200) { $Check.Issues.Add("GET /api/telemetry/spans answered HTTP $($spans.Status) $($spans.Error)"); return }
    $body = [string]$spans.Body
    $missing = @($ids | Where-Object { -not $body.Contains("`"$_`"") })
    if ($missing.Count -gt 0) { $Check.Issues.Add("$($missing.Count) of $($ids.Count) request ids are in Application Insights' run but not in the dashboard's spans: $((@($missing | Select-Object -First 6)) -join ', ')") }
    else { $Check.Notes.Add("all $($ids.Count) request ids are on spans in the dashboard too") }
}

function Test-RunTelemetry($Ctx) {
    $data = Get-RunTelemetry $Ctx

    $check = New-Check
    Test-AiRequestRows $Ctx $data $check
    Complete-Check $Ctx.Name 'request-rows' $check

    foreach ($entry in $Ctx.Pending) { Test-PendingEntry $Ctx $data $entry }

    $check = New-Check
    Test-AiUntracedPaths $Ctx $data $check
    Complete-Check $Ctx.Name 'untraced-paths' $check

    $check = New-Check
    Test-AiRoleInstance $Ctx $data $check
    Complete-Check $Ctx.Name 'role-instance' $check

    if (-not $Ctx.Run.Capture) {
        $check = New-Check
        Test-AiRedaction $Ctx $data $check
        Complete-Check $Ctx.Name 'content-redaction' $check
    }

    if ($Selected -contains 'metrics' -and (Get-ExpectedMetrics $Ctx).Count -gt 0) {
        $check = New-Check
        Test-AiMetrics $Ctx $data $check
        Complete-Check $Ctx.Name 'metrics' $check
        $script:MetricsChecked = $true
    }

    if ($script:Otlp.Enabled) {
        $check = New-Check
        Test-AiBothSinks $Ctx $check
        Complete-Check $Ctx.Name 'both-sinks' $check
    }
}

# ---------------------------------------------------------------------------------------------
# Checks done while the run is live, or right after it stops
# ---------------------------------------------------------------------------------------------

function Test-AiStartupLine($Ctx, $Check) {
    $state = Wait-Until -TimeoutSec 15 -IntervalSec 1 -Probe {
        $line = Get-TelemetryLine $Ctx.Server
        if ($line) { [pscustomobject]@{ Done = $true; Line = $line } }
    }
    if (-not $state) { $Check.Issues.Add("no 'Telemetry: ' startup line in the server output ($($Ctx.Server.StdoutPath))"); return }
    $line = $state.Line
    $script:RunInfo[$Ctx.Name]['telemetryLine'] = $line
    $Check.Notes.Add($line)
    if ($line -match '^Telemetry: off \(') { $Check.Issues.Add('telemetry is off although APPLICATIONINSIGHTS_CONNECTION_STRING was set'); return }
    foreach ($signal in 'traces', 'metrics', 'logs') {
        if ($line -notmatch "$signal\([^)]*azuremonitor") { $Check.Issues.Add("$signal are not exported to Azure Monitor") }
        $otlp = $line -match "$signal\([^)]*otlp"
        if ($script:Otlp.Enabled -and -not $otlp) { $Check.Issues.Add("$signal are not also exported over OTLP (-BothSinks)") }
        if (-not $script:Otlp.Enabled -and $otlp) { $Check.Issues.Add("$signal also go to OTLP although no OTLP variable was set") }
    }
    if ($Ctx.Run.EntraId) {
        if (-not $line.Contains('Azure Monitor (Entra ID)')) { $Check.Issues.Add("expected 'Azure Monitor (Entra ID)' with TENSORSHARP_AZURE_MONITOR_AAD=1") }
    } else {
        if (-not $line.Contains('Azure Monitor')) { $Check.Issues.Add("the line does not name 'Azure Monitor' as a destination") }
        if ($line.Contains('(Entra ID)')) { $Check.Issues.Add("the line says '(Entra ID)' although TENSORSHARP_AZURE_MONITOR_AAD was not set") }
    }
    if ($script:Otlp.Enabled) {
        $expected = "OTLP $OtlpEndpoint ($OtlpProtocol)"
        if (-not $line.Contains($expected)) { $Check.Issues.Add("the line does not name '$expected'") }
    }
    if (-not $line.Contains("; service $($Ctx.Service);")) { $Check.Issues.Add("the line does not name service $($Ctx.Service)") }
    $capture = if ($Ctx.Run.Capture) { 'content capture ON' } else { 'content capture off' }
    if (-not $line.Contains($capture)) { $Check.Issues.Add("expected '$capture'") }
    if (-not $line.Contains('; sampler azuremonitor fixed_percentage(1)')) { $Check.Issues.Add("expected 'sampler azuremonitor fixed_percentage(1)' (no OTEL_TRACES_SAMPLER is set)") }
    $output = (Read-SharedText $Ctx.Server.StdoutPath) + "`n" + (Read-SharedText $Ctx.Server.StderrPath)
    if ($output -match '(?i)live ?metrics') { $Check.Issues.Add('the server output mentions Live Metrics (EnableLiveMetrics should be off and warn about nothing)') }
}

function Test-ServerStop($Ctx, $Check) {
    $info = $Ctx.StopInfo
    if (-not $info) { $Check.Issues.Add('the server was not stopped by the harness'); return }
    $text = "$($info.mode); signal $($info.signal) ($($info.signalResult)); exit code $($info.exitCode); $($info.seconds) s after the signal; waited $($info.settleSec) s before it"
    if ($info.mode -ne 'graceful') {
        $Check.Issues.Add("the server did not shut down by itself ($text), so telemetry still queued in its exporters (the last batch, the last metric interval) may be missing")
    } elseif ($info.exitCode -ne 0) {
        $Check.Issues.Add("the server exited with code $($info.exitCode) after $($info.signal): $text")
    } else {
        $Check.Notes.Add($text)
    }
}

# The connection string (or its instrumentation key) must appear in none of the server's output.
function Test-ConnectionStringNotLogged($Ctx, $Check) {
    $secrets = @(
        [pscustomobject]@{ Value = $script:Ai.ConnectionString; Label = 'the connection string' },
        [pscustomobject]@{ Value = $script:Ai.InstrumentationKey; Label = 'the instrumentation key' },
        [pscustomobject]@{ Value = $OtlpApiKey; Label = 'the OTLP API key' }
    ) | Where-Object { $_.Value -and $_.Value.Length -ge 8 }
    $files = Get-ServerOutputFiles $Ctx.Server
    foreach ($file in $files) {
        $text = Read-SharedText $file
        $found = $false
        foreach ($secret in $secrets) {
            if ($text.Contains($secret.Value, [StringComparison]::OrdinalIgnoreCase)) {
                $Check.Issues.Add("$($secret.Label) appears in $(Split-Path -Leaf $file) (replaced in the evidence copy)")
                $found = $true
            }
        }
        if ($found) { [IO.File]::WriteAllText($file, (Protect-Text $text), [Text.UTF8Encoding]::new($false)) }
    }
    $Check.Notes.Add("searched $($files.Count) output files")
}

# ---------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------

function New-Run([string] $Name, $ModelArgs, [string[]] $RunScenarios) {
    [pscustomobject]@{
        Name        = $Name
        ModelArgs   = $ModelArgs
        Scenarios   = @($RunScenarios)
        Capture     = $false
        EntraId     = $false
        VideoSize   = ''
        VideoFrames = 0
    }
}

# Which server runs to make. A selected scenario whose model or switch is missing is recorded as
# SKIPPED here and left out of every run.
function Get-RunPlan($ChatArgs, $EmbeddingArgs, $LtxArgs, $H3Args) {
    $missing = @{}
    foreach ($s in @($LlmScenarios) + 'content-capture') { if (-not $ChatArgs) { $missing[$s] = 'needs -ChatModel' } }
    if (-not $EmbeddingArgs) { $missing['embeddings'] = 'needs -EmbeddingModel' }
    if (-not $LtxArgs) { $missing['video-ltx'] = 'needs -LtxConfig' }
    if (-not $H3Args) { $missing['video-minimax-h3'] = 'needs -MiniMaxH3Config' }
    if (-not $LtxArgs -and -not $H3Args) {
        $missing['video-cancel'] = 'needs -LtxConfig or -MiniMaxH3Config'
        $missing['studio-generate'] = 'needs -LtxConfig or -MiniMaxH3Config'
    }
    if (-not $EntraId) { $missing['entra-id'] = 'needs -EntraId' }
    foreach ($s in $Selected) { if ($missing.ContainsKey($s)) { Add-Result '-' $s 'SKIPPED' $missing[$s] } }
    $runnable = @($Selected | Where-Object { -not $missing.ContainsKey($_) })

    $plan = [System.Collections.Generic.List[object]]::new()
    $main = @(@($LlmScenarios) + 'queue-status-poll' | Where-Object { $runnable -contains $_ })
    if ($main.Count -gt 0) {
        $needsChat = [bool]($main | Where-Object { $_ -like 'llm-*' })
        $plan.Add((New-Run 'main' $(if ($needsChat) { $ChatArgs } else { $null }) $main))
    }
    if ($runnable -contains 'content-capture') {
        $run = New-Run 'content-capture' $ChatArgs @('content-capture')
        $run.Capture = $true
        $plan.Add($run)
    }
    if ($runnable -contains 'embeddings') { $plan.Add((New-Run 'embeddings' $EmbeddingArgs @('embeddings'))) }
    $shared = @('video-cancel', 'studio-generate' | Where-Object { $runnable -contains $_ })
    if ($LtxArgs -and ($runnable -contains 'video-ltx' -or $shared.Count)) {
        $run = New-Run 'video-ltx' $LtxArgs (@('video-ltx' | Where-Object { $runnable -contains $_ }) + $shared)
        $run.VideoSize = $LtxSize
        $run.VideoFrames = $LtxFrames
        $plan.Add($run)
        $shared = @()
    }
    if ($H3Args -and ($runnable -contains 'video-minimax-h3' -or $shared.Count)) {
        $run = New-Run 'video-minimax-h3' $H3Args (@('video-minimax-h3' | Where-Object { $runnable -contains $_ }) + $shared)
        $run.VideoSize = $MiniMaxH3Size
        $run.VideoFrames = $MiniMaxH3Frames
        $plan.Add($run)
    }
    if ($runnable -contains 'entra-id') {
        # Ingestion with a token is under test. With a chat model, one chat request makes the
        # token carry spans, logs and metrics (and gives the redaction check a prompt to look
        # for); without one, the requests rows of GET /api/models are the evidence.
        $run = New-Run 'entra-id' $ChatArgs @('entra-id')
        $run.EntraId = $true
        $plan.Add($run)
    }
    return $plan
}

function Get-RunEnvironment($Run, [string] $Service, [string] $RunDir) {
    $environment = [ordered]@{
        APPLICATIONINSIGHTS_CONNECTION_STRING = $script:Ai.ConnectionString
        OTEL_SERVICE_NAME                     = $Service
        # Short export intervals so data leaves soon after each request, not only at shutdown.
        OTEL_BSP_SCHEDULE_DELAY               = '2000'
        OTEL_BLRP_SCHEDULE_DELAY              = '2000'
        OTEL_METRIC_EXPORT_INTERVAL           = '5000'
        TENSORSHARP_LOG_DIR                   = (Join-Path $RunDir 'logs')
        TS_STUDIO_DIRECTORY                   = (Join-Path $RunDir 'studio-projects')
    }
    if ($Run.EntraId) { $environment['TENSORSHARP_AZURE_MONITOR_AAD'] = '1' }
    if ($Run.Capture) { $environment['OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'] = 'true' }
    if ($script:Otlp.Enabled) {
        $environment['OTEL_EXPORTER_OTLP_ENDPOINT'] = $OtlpEndpoint
        $environment['OTEL_EXPORTER_OTLP_PROTOCOL'] = $OtlpProtocol
        if ($OtlpApiKey) { $environment['OTEL_EXPORTER_OTLP_HEADERS'] = "x-otlp-api-key=$OtlpApiKey" }
    }
    return $environment
}

function Invoke-Run($Run) {
    $runDir = Join-Path $OutDir "runs/$($Run.Name)"
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    $service = "tensorsharp-e2e-$script:RunTag-$($Run.Name)"
    $environment = Get-RunEnvironment $Run $service $runDir
    $workingDirectory = (Get-Location).Path
    $diagnosticsDir = ''
    if ($ExporterDiagnostics) {
        # The SDK reads OTEL_DIAGNOSTICS.json from the process's working folder.
        $workingDirectory = $runDir
        $diagnosticsDir = Join-Path $runDir 'otel-diagnostics'
        New-Item -ItemType Directory -Path $diagnosticsDir -Force | Out-Null
        Save-Json (Join-Path $runDir 'OTEL_DIAGNOSTICS.json') ([ordered]@{ LogDirectory = $diagnosticsDir; FileSize = 4096; LogLevel = 'Warning' })
    }

    $serverArgs = @(@($Run.ModelArgs) + @($script:ExtraArgs) | Where-Object { $null -ne $_ -and "$_" -ne '' })
    $serverArgs += @('--host', '127.0.0.1', '--port', "$Port")

    $shownEnvironment = [ordered]@{}
    foreach ($key in $environment.Keys) {
        $shownEnvironment[$key] = switch ($key) {
            'APPLICATIONINSIGHTS_CONNECTION_STRING' { "<connection string of $($script:Ai.ResourceName), read with az; not recorded>" }
            'OTEL_EXPORTER_OTLP_HEADERS' { 'x-otlp-api-key=<redacted>' }
            default { $environment[$key] }
        }
    }
    Save-Json (Join-Path $runDir 'server-launch.json') ([ordered]@{
            file = $script:ServerHostInfo.FileName; arguments = @($script:ServerHostInfo.Prefix) + $serverArgs
            workingDirectory = $workingDirectory; environment = $shownEnvironment
            note = 'Inherited OTEL_*, TENSORSHARP_OTEL*, APPLICATIONINSIGHTS_*, TENSORSHARP_AZURE_MONITOR_AAD, PORT, HOST and ASPNETCORE_URLS were removed.'
        })
    $script:RunInfo[$Run.Name] = [ordered]@{
        serviceName = $service; entraId = $Run.EntraId; contentCapture = $Run.Capture; otlp = $script:Otlp.Enabled
        arguments = $serverArgs; scenarios = $Run.Scenarios
    }

    Write-Step "Run $($Run.Name): starting TensorSharp.Server.Host as $service"
    $server = $null
    $startUtc = [DateTime]::UtcNow
    try {
        # Something else answering on the port would be mistaken for this server.
        Wait-PortFree $Port
        if (Test-TcpPort '127.0.0.1' $Port) { throw "port $Port is already in use; stop what listens there or pass -Port" }
        $startUtc = [DateTime]::UtcNow
        $server = Start-TsServer -Dir $runDir -ServerArgs $serverArgs -Environment $environment -WorkingDirectory $workingDirectory
        $server.DiagnosticsDir = $diagnosticsDir
        $script:CurrentServer = $server
        $script:RunInfo[$Run.Name]['pid'] = $server.Pid
        $script:RunInfo[$Run.Name]['startUtc'] = $startUtc.ToString('o')
        Wait-ServerReady $server
    } catch {
        Add-Result $Run.Name 'server-start' 'FAIL' "the server did not start: $($_.Exception.Message)"
        foreach ($s in $Run.Scenarios) { Add-Result $Run.Name $s 'FAIL' 'not run: the server did not start' }
        if ($server) { [void](Stop-TsServer $server -Quick) }
        $script:CurrentServer = $null
        return
    }

    $ctx = [pscustomobject]@{
        Name             = $Run.Name
        Run              = $Run
        Base             = $server.Base
        Service          = $service
        Server           = $server
        Dir              = $runDir
        InstanceId       = "$($script:Machine):$($server.Pid)"
        Model            = $null
        MarkerSent       = $false
        Markers          = [System.Collections.Generic.List[string]]::new()
        LastMarker       = ''
        ExchangeCount    = 0
        Sent             = [System.Collections.Generic.List[object]]::new()
        CurrentScenario  = ''
        Pending          = [System.Collections.Generic.List[object]]::new()
        StartUtc         = $startUtc
        StopUtc          = $startUtc
        StopInfo         = $null
        QueryCount       = 0
        QueueStatusPolls = 0
        ExporterMessages = @()
    }
    try {
        $check = New-Check
        Test-AiStartupLine $ctx $check
        Complete-Check $Run.Name 'startup-line' $check

        $smokeEntry = New-PendingEntry 'request-row-and-logs'
        $smoke = Invoke-Http -Url "$($ctx.Base)/api/models" -TimeoutSec 60
        Save-Exchange $ctx 'api-models' $smoke
        if ($smoke.Status -ne 200) { $smokeEntry.Check.Issues.Add("GET /api/models: HTTP $($smoke.Status) $(Get-ErrorText $smoke)") }
        else { Add-Expectation $smokeEntry $smoke @{ Kind = 'gen'; NoGenAi = $true; Protocol = 'webui'; ExpectLogs = $true; Label = 'GET /api/models' } | Out-Null }
        $ctx.Pending.Add($smokeEntry)
        if ($smoke.Json) {
            $script:RunInfo[$Run.Name]['loadedModel'] = $smoke.Json.loaded
            $script:RunInfo[$Run.Name]['loadedBackend'] = $smoke.Json.loadedBackend
            $script:RunInfo[$Run.Name]['architecture'] = $smoke.Json.architecture
        }
        $ctx.Model = Get-ModelId $ctx
        $script:RunInfo[$Run.Name]['modelId'] = $ctx.Model

        foreach ($scenario in $Run.Scenarios) { Invoke-Scenario $ctx $scenario }
    } finally {
        Write-Step "Run $($Run.Name): stopping the server (exporters flush on shutdown)"
        $ctx.StopInfo = Stop-TsServer $server
        $ctx.StopUtc = [DateTime]::UtcNow
        $script:CurrentServer = $null
        $script:RunInfo[$Run.Name]['stopUtc'] = $ctx.StopUtc.ToString('o')
        $script:RunInfo[$Run.Name]['stop'] = $ctx.StopInfo
        $script:RunInfo[$Run.Name]['exitCode'] = $server.ExitCode
    }

    $check = New-Check
    Test-ServerStop $ctx $check
    Complete-Check $Run.Name 'server-stop' $check

    $check = New-Check
    Test-ConnectionStringNotLogged $ctx $check
    Complete-Check $Run.Name 'connection-string-not-logged' $check

    $ctx.ExporterMessages = Get-ExporterMessages $ctx
    $script:RunInfo[$Run.Name]['exporterMessages'] = @($ctx.ExporterMessages).Count
    $expectations = Get-AllExpectations $ctx
    $script:RunInfo[$Run.Name]['requestIds'] = @($expectations | Where-Object { $_.RequestId }).Count
    Save-Json (Join-Path $runDir 'requests-sent.json') ([ordered]@{
            serviceName  = $service
            startUtc     = $ctx.StartUtc.ToString('o')
            stopUtc      = $ctx.StopUtc.ToString('o')
            markers      = @($ctx.Markers)
            expectations = @($expectations | ForEach-Object {
                    [ordered]@{
                        scenario = $_.Scenario; label = $_.Label; kind = $_.Kind; method = $_.Method; path = $_.Path; status = $_.Status
                        requestId = $_.RequestId; marker = $_.Marker; operation = $_.Operation; outcome = $_.Outcome; jobId = $_.JobId
                    }
                })
            exchanges    = @($ctx.Sent)
        })
    $script:Contexts.Add($ctx)
}

function Initialize-Otlp {
    if (-not $BothSinks) { return }
    $uri = $null
    if (-not [uri]::TryCreate($OtlpEndpoint, [UriKind]::Absolute, [ref]$uri) -or $uri.Scheme -notin 'http', 'https') {
        throw "-OtlpEndpoint '$OtlpEndpoint' is not an absolute http:// or https:// URL."
    }
    if ($OtlpProtocol -notin 'grpc', 'http/protobuf') { throw "-OtlpProtocol must be grpc or http/protobuf." }
    $script:Otlp.Reachable = Test-TcpPort $uri.Host $uri.Port
    if ($script:Otlp.Reachable) {
        $script:Otlp.Enabled = $true
        Write-Step "-BothSinks: every run also exports OTLP to $OtlpEndpoint ($OtlpProtocol)"
    } else {
        $reason = "-BothSinks: nothing listens at $($uri.Host):$($uri.Port), so every run exports to Application Insights only"
        Add-Result '-' 'both-sinks' 'SKIPPED' $reason
        $script:Limitations.Add($reason)
    }
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

# The OpenTelemetry and Azure packages the server build resolved, from its deps.json.
function Get-PackageVersions {
    if (-not $script:ServerHostInfo) { return [ordered]@{ note = 'no server build resolved' } }
    $deps = Join-Path $script:ServerHostInfo.Directory 'TensorSharp.Server.Host.deps.json'
    if (-not (Test-Path -LiteralPath $deps)) { return [ordered]@{ note = "no $deps" } }
    $json = ConvertFrom-JsonText (Get-Content -LiteralPath $deps -Raw)
    $versions = [ordered]@{}
    if ($json -and $json.libraries) {
        foreach ($name in @($json.libraries.PSObject.Properties.Name | Sort-Object)) {
            $parts = $name -split '/', 2
            if ($parts.Count -eq 2 -and $parts[0] -match '^(OpenTelemetry|Azure\.)') { $versions[$parts[0]] = $parts[1] }
        }
    }
    return $versions
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
    [ordered]@{
        generatedAtUtc      = (Get-Date).ToUniversalTime().ToString('o')
        script              = 'eng/telemetry/appinsights-e2e.ps1'
        parameters          = [ordered]@{
            chatModel = $ChatModel; embeddingModel = $EmbeddingModel; ltxConfig = $LtxConfig; miniMaxH3Config = $MiniMaxH3Config
            scenarios = $Selected; port = $Port; extraServerArgs = $ExtraServerArgs; maxTokens = $MaxTokens
            ltx = "$LtxSize x $LtxFrames frames"; miniMaxH3 = "$MiniMaxH3Size x $MiniMaxH3Frames frames"
            ingestionTimeoutMinutes = $IngestionTimeoutMinutes; entraId = [bool]$EntraId; bothSinks = [bool]$BothSinks
            exporterDiagnostics = [bool]$ExporterDiagnostics; settleSec = $SettleSec; stopGraceSec = $StopGraceSec
        }
        promptMarker        = $script:MarkerPrefix
        tensorSharp         = Get-GitInfo $RepoRoot
        ggml                = Get-GgmlInfo
        serverHost          = $serverHost
        packages            = Get-PackageVersions
        applicationInsights = [ordered]@{
            resourceName        = $script:Ai.ResourceName
            resourceType        = 'microsoft.insights/components'
            connectionString    = 'read with az resource show at run time and passed only to the server processes; not recorded'
            queryApi            = 'POST https://api.applicationinsights.io/v1/apps/<app-id>/query through az rest'
            azureCli            = $script:Ai.AzVersion
            cliAccountType      = $script:Ai.AccountType
            # Names only: DefaultAzureCredential (the Entra ID run) reads them.
            credentialVariables = @(Get-ChildItem env: | Where-Object { $_.Name -like 'AZURE_*' } | ForEach-Object Name | Sort-Object)
        }
        otlp                = [ordered]@{
            requested = [bool]$BothSinks; enabled = $script:Otlp.Enabled; reachable = $script:Otlp.Reachable
            endpoint = if ($BothSinks) { $OtlpEndpoint } else { $null }; protocol = if ($BothSinks) { $OtlpProtocol } else { $null }
            dashboardUrl = if ($BothSinks) { $DashboardUrl } else { $null }; dashboardApiKeyGiven = [bool]$DashboardApiKey
        }
        os                  = [ordered]@{
            description  = [Runtime.InteropServices.RuntimeInformation]::OSDescription
            architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
            machine      = $script:Machine
        }
        dotnet              = Get-DotnetInfo
        gpu                 = Get-GpuInfo
        runs                = $script:RunInfo
        limitations         = @($script:Limitations)
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
    Save-Json (Join-Path $OutDir 'results.json') ([ordered]@{
            summary = $summary; resource = $script:Ai.ResourceName; promptMarker = $script:MarkerPrefix; results = $script:Results
        })

    $md = [System.Text.StringBuilder]::new()
    [void]$md.AppendLine('# Application Insights end-to-end results')
    [void]$md.AppendLine()
    $tensorSharp = $Environment.tensorSharp
    [void]$md.AppendLine("Run $($Environment.generatedAtUtc) on $($Environment.os.description) ($($Environment.os.architecture)).")
    $tensorSharpText = if ($tensorSharp.commit) { "$($tensorSharp.commit) on $($tensorSharp.branch) with $($tensorSharp.uncommittedChanges) uncommitted change(s)" } else { $tensorSharp.note }
    $ggmlText = if ($Environment.ggml.commit) { $Environment.ggml.commit } else { $Environment.ggml.note }
    [void]$md.AppendLine("TensorSharp: $tensorSharpText. ggml: $ggmlText.")
    $exporter = $Environment.packages['Azure.Monitor.OpenTelemetry.Exporter']
    [void]$md.AppendLine("Application Insights resource: $($Environment.applicationInsights.resourceName)$(if ($exporter) { "; Azure.Monitor.OpenTelemetry.Exporter $exporter" }). Rows were read back through the query API, filtered by each run's cloud_RoleName.")
    if ($Environment.otlp.requested) { [void]$md.AppendLine("OTLP alongside: $(if ($Environment.otlp.enabled) { "$($Environment.otlp.endpoint) ($($Environment.otlp.protocol))" } else { 'requested, not reachable' }).") }
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
    [void]$md.AppendLine('Evidence: `environment.json` (versions, runs, models), `runs/<run>/` (server output, file logs, every request in `requests/`, `requests-sent.json` with the request ids and prompt markers, and every query with its last result in `appinsights/`).')
    if ($Environment.limitations.Count -gt 0) {
        [void]$md.AppendLine()
        [void]$md.AppendLine('Limitations:')
        foreach ($item in $Environment.limitations) { [void]$md.AppendLine("- $item") }
    }
    Save-Text (Join-Path $OutDir 'results.md') $md.ToString()
}

# Last line of defence: replace any known secret found anywhere in the evidence.
function Protect-EvidenceFiles {
    $changed = [System.Collections.Generic.List[string]]::new()
    $binary = @('.mp4', '.webm', '.mov', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.wav', '.mp3', '.flac', '.gguf', '.bin', '.safetensors', '.onnx', '.zip')
    foreach ($file in @(Get-ChildItem -LiteralPath $OutDir -File -Recurse -Force -ErrorAction SilentlyContinue)) {
        if ($binary -contains $file.Extension.ToLowerInvariant() -or $file.Length -gt 64MB) { continue }
        $text = Read-SharedText $file.FullName
        $clean = Protect-Text $text
        if ($clean -cne $text) {
            [IO.File]::WriteAllText($file.FullName, $clean, [Text.UTF8Encoding]::new($false))
            $changed.Add([IO.Path]::GetRelativePath($OutDir, $file.FullName))
        }
    }
    return , $changed.ToArray()
}

# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

if (-not $OutDir) { $OutDir = Join-Path $RepoRoot ("docs/validation/appinsights-e2e-" + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
$OutDir = [IO.Path]::GetFullPath($OutDir)
$evidenceRoots = @('docs/validation', 'artifacts') | ForEach-Object { [IO.Path]::GetFullPath((Join-Path $RepoRoot $_)) + [IO.Path]::DirectorySeparatorChar }
if (-not ($evidenceRoots | Where-Object { $OutDir.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) })) {
    throw "-OutDir must be inside docs/validation/ or artifacts/ under $RepoRoot (both are git-ignored); got $OutDir"
}
if ((Test-Path -LiteralPath $OutDir) -and (Get-ChildItem -LiteralPath $OutDir -Force | Select-Object -First 1)) {
    throw "-OutDir $OutDir already has files; use a fresh folder for each run."
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$script:MarkerPrefix = 'TSAI' + (New-RandomHex 5).ToUpperInvariant()
$script:PromptCount = 0
$script:RunTag = New-RandomHex 3
$script:Machine = [Environment]::MachineName
$script:AzPath = ''
$script:Ai = [pscustomobject]@{
    ResourceName = ''; ConnectionString = ''; InstrumentationKey = ''; QueryUrl = ''; AzVersion = ''; AccountType = ''; Connected = $false
}
$script:Otlp = [pscustomobject]@{ Enabled = $false; Reachable = $null }
$script:RunInfo = [ordered]@{}
$script:CurrentServer = $null
$script:ServerHostInfo = $null
$script:ExtraArgs = @()
$script:Contexts = [System.Collections.Generic.List[object]]::new()
$script:MetricsChecked = $false
$script:Limitations = [System.Collections.Generic.List[string]]::new()
$script:Limitations.Add('Application Insights keeps histograms as count, sum, min and max: the metrics row checks that each instrument has rows, not their values.')
$script:Limitations.Add('Absence checks (no /health or /api/queue/status rows, no prompt marker) query once, after the run''s expected rows arrived; a row ingested later than that is not seen.')
$script:Limitations.Add('Whether the Azure Monitor exporters honour OTEL_BSP_SCHEDULE_DELAY / OTEL_BLRP_SCHEDULE_DELAY / OTEL_METRIC_EXPORT_INTERVAL is not verified; the graceful stop flushes whatever is still queued.')
$script:TempDir = Join-Path ([IO.Path]::GetTempPath()) ("tensorsharp-appinsights-e2e-" + (New-RandomHex 4))
New-Item -ItemType Directory -Path $script:TempDir -Force | Out-Null
Register-Secret $OtlpApiKey '<otlp-api-key>'
Register-Secret $DashboardApiKey '<dashboard-api-key>'

Write-Step "Application Insights end-to-end check; evidence in $OutDir; prompt marker prefix $script:MarkerPrefix; run tag $script:RunTag"
try {
    $script:ServerHostInfo = Resolve-ServerHost $ServerHost
    Write-Step "Server host: $($script:ServerHostInfo.Path)"
    $chatArgs = Resolve-ModelArgs $ChatModel 'ChatModel'
    $embeddingArgs = Resolve-ModelArgs $EmbeddingModel 'EmbeddingModel' -Embeddings
    $ltxArgs = Resolve-ModelArgs $LtxConfig 'LtxConfig'
    $h3Args = Resolve-ModelArgs $MiniMaxH3Config 'MiniMaxH3Config'
    $script:ExtraArgs = if ($ExtraServerArgs.Trim()) { @(Split-CommandLine $ExtraServerArgs.Trim()) } else { @() }
    foreach ($size in $LtxSize, $MiniMaxH3Size) { [void](ConvertTo-Size $size) }

    Connect-AppInsights
    Initialize-Otlp

    $plan = Get-RunPlan $chatArgs $embeddingArgs $ltxArgs $h3Args
    if ($plan.Count -eq 0) {
        Write-Warning 'Nothing to run: every selected scenario was skipped.'
    } else {
        Write-Step "Runs: $(($plan | ForEach-Object Name) -join ', ')"
        foreach ($run in $plan) { Invoke-Run $run }
        # Every run has stopped: read them all back, so their ingestion delays overlap.
        foreach ($ctx in $script:Contexts) {
            try {
                Test-RunTelemetry $ctx
            } catch {
                Add-Result $ctx.Name 'appinsights-read' 'FAIL' "harness error: $($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))"
            }
        }
    }
    if ($Selected -contains 'metrics' -and -not $script:MetricsChecked) { Add-Result '-' 'metrics' 'SKIPPED' 'no run sent completed LLM or video traffic' }
} catch {
    Add-Result 'harness' 'setup' 'FAIL' "$($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))"
} finally {
    if ($script:CurrentServer) { try { [void](Stop-TsServer $script:CurrentServer -Quick) } catch { Write-Warning (Protect-Text $_.Exception.Message) } }
    $environmentInfo = $null
    try {
        $environmentInfo = Get-EnvironmentInfo
        Save-Json (Join-Path $OutDir 'environment.json') $environmentInfo
    } catch {
        Write-Warning (Protect-Text "environment.json: $($_.Exception.Message)")
    }
    try {
        $scrubbed = Protect-EvidenceFiles
        if ($scrubbed.Count -gt 0) {
            Add-Result 'harness' 'evidence-scrub' 'FAIL' "a known secret was found and replaced in $($scrubbed.Count) evidence file(s): $((@($scrubbed | Select-Object -First 10)) -join ', ')"
        } else {
            Add-Result 'harness' 'evidence-scrub' 'PASS' 'no connection string, instrumentation key, AppId, resource id, subscription id or API key in the evidence'
        }
    } catch {
        Add-Result 'harness' 'evidence-scrub' 'FAIL' "could not scan the evidence: $($_.Exception.Message)"
    }
    if ($environmentInfo) { Write-ResultFiles $environmentInfo }
    else { Save-Json (Join-Path $OutDir 'results.json') ([ordered]@{ results = $script:Results }) }
    Remove-Item -LiteralPath $script:TempDir -Recurse -Force -ErrorAction SilentlyContinue
    $script:Ai.ConnectionString = ''
    $script:Http.Dispose()
}

$failedCount = @($script:Results | Where-Object { $_.status -eq 'FAIL' }).Count
$counts = $script:Results | Group-Object status | ForEach-Object { "$($_.Count) $($_.Name)" }
Write-Step "Done: $($counts -join ', '). Results: $(Join-Path $OutDir 'results.md')"
exit ([int]($failedCount -gt 0))
