<#
.SYNOPSIS
    Pretty-prints TurboQuantLoader's conversation log, grouped into sessions,
    with thinking / tool-call / response content visually separated.

.DESCRIPTION
    Reads logs/conversations.<date>.jsonl (written when config.toml has
    log_conversations = true). Each line is one completed request/response
    exchange with the full message history sent and the raw model output
    (including <think>...</think> and <tool_call>...</tool_call> blocks).

    Multi-turn conversations appear as multiple lines whose `messages` array
    grows each turn (the system prompt is shared across every session, so
    grouping by first message alone would lump unrelated sessions together).
    This script instead chains entries whose messages array is an exact
    prefix extension of an earlier entry's, and prints only the new turn
    delta each time, so you can read a session top-to-bottom instead of
    re-reading the accumulated history on every line.

.PARAMETER Date
    Log date to load, as YYYY-MM-DD. Defaults to today.

.PARAMETER Path
    Explicit path to a .jsonl file, overriding -Date.

.PARAMETER LogDir
    Directory containing conversations.<date>.jsonl. Defaults to ./logs.

.EXAMPLE
    ./scripts/view-conversations.ps1
    ./scripts/view-conversations.ps1 -Date 2026-08-18
    ./scripts/view-conversations.ps1 -Path logs/conversations.2026-08-19.jsonl
#>
param(
    [string]$Date = (Get-Date -Format "yyyy-MM-dd"),
    [string]$Path,
    [string]$LogDir = "logs"
)

if (-not $Path) {
    $Path = Join-Path $LogDir "conversations.$Date.jsonl"
}

if (-not (Test-Path $Path)) {
    Write-Host "No conversation log found at $Path" -ForegroundColor Red
    exit 1
}

function Write-Segmented {
    param([string]$Text)

    # Split on <think>, </think>, <tool_call>, </tool_call> while keeping the tags,
    # so each segment can be colorized independently.
    $pattern = '(<think>|</think>|<tool_call>|</tool_call>)'
    $parts = [regex]::Split($Text, $pattern)

    $mode = "response"
    foreach ($part in $parts) {
        if ($part -eq "") { continue }
        switch ($part) {
            "<think>"        { $mode = "thinking"; continue }
            "</think>"       { $mode = "response";  continue }
            "<tool_call>"     { $mode = "tool_call"; continue }
            "</tool_call>"    { $mode = "response";  continue }
        }
        switch ($mode) {
            "thinking"  { Write-Host $part -ForegroundColor DarkGray }
            "tool_call" { Write-Host $part -ForegroundColor Yellow }
            default     { Write-Host $part -ForegroundColor White }
        }
    }
    Write-Host ""
}

$lines = Get-Content -Path $Path -Encoding utf8
$entries = @()
foreach ($line in $lines) {
    if ($line.Trim() -eq "") { continue }
    try {
        $entries += ($line | ConvertFrom-Json)
    } catch {
        Write-Host "Skipping unparsable line: $($_.Exception.Message)" -ForegroundColor DarkRed
    }
}

if ($entries.Count -eq 0) {
    Write-Host "No entries found in $Path" -ForegroundColor Yellow
    exit 0
}

# Chain entries whose `messages` array is an exact prefix-extension of an
# earlier entry's `messages` array — that's how a growing conversation
# history looks across turns. Every session shares the same system prompt,
# so grouping by messages[0] alone (the old approach) is not distinctive
# enough; full-array prefix matching is.
function Test-IsPrefix {
    param($ShorterMessages, $LongerMessages)
    if ($ShorterMessages.Count -ge $LongerMessages.Count) { return $false }
    for ($i = 0; $i -lt $ShorterMessages.Count; $i++) {
        if ($ShorterMessages[$i].role -ne $LongerMessages[$i].role -or
            $ShorterMessages[$i].content -ne $LongerMessages[$i].content) {
            return $false
        }
    }
    return $true
}

$ordered = $entries | Sort-Object { $_.messages.Count }
$chains = @()  # each chain: @{ Turns = [System.Collections.ArrayList]; LastMessages = ... }

foreach ($entry in $ordered) {
    $bestChain = $null
    $bestMatchLen = -1
    foreach ($chain in $chains) {
        if (Test-IsPrefix -ShorterMessages $chain.LastMessages -LongerMessages $entry.messages) {
            if ($chain.LastMessages.Count -gt $bestMatchLen) {
                $bestMatchLen = $chain.LastMessages.Count
                $bestChain = $chain
            }
        }
    }
    if ($bestChain) {
        [void]$bestChain.Turns.Add($entry)
        $bestChain.LastMessages = $entry.messages
    } else {
        $newChain = @{
            Turns        = New-Object System.Collections.ArrayList
            LastMessages = $entry.messages
        }
        [void]$newChain.Turns.Add($entry)
        $chains += $newChain
    }
}

$sessions = $chains | Sort-Object { $_.Turns[0].ts }

$sessionIndex = 0
foreach ($session in $sessions) {
    $sessionIndex++
    $turns = @($session.Turns | Sort-Object { $_.messages.Count })

    $firstUserMsg = $turns[0].messages | Where-Object { $_.role -eq "user" } | Select-Object -First 1
    $previewSource = if ($firstUserMsg) { $firstUserMsg.content } else { $turns[0].messages[0].content }
    $previewSource = ($previewSource -replace '\s+', ' ')
    $preview = $previewSource.Substring(0, [Math]::Min(80, $previewSource.Length))
    Write-Host ("=" * 100) -ForegroundColor Cyan
    Write-Host "SESSION $sessionIndex  |  $($turns.Count) turn(s)  |  $preview..." -ForegroundColor Cyan
    Write-Host ("=" * 100) -ForegroundColor Cyan

    $seenCount = 0
    foreach ($turn in $turns) {
        Write-Host ""
        Write-Host "--- $($turn.ts)  model=$($turn.model)  prompt_tokens=$($turn.prompt_tokens) completion_tokens=$($turn.completion_tokens) tps=$([math]::Round($turn.tps,1)) finish=$($turn.finish_reason) ---" -ForegroundColor Magenta

        # Print only the messages new since the last turn in this session.
        $newMessages = $turn.messages | Select-Object -Skip $seenCount
        foreach ($msg in $newMessages) {
            Write-Host "[$($msg.role)]" -ForegroundColor Blue
            Write-Host $msg.content
            Write-Host ""
        }
        $seenCount = $turn.messages.Count

        Write-Host "[assistant response]" -ForegroundColor Green
        Write-Segmented -Text $turn.response
    }
    Write-Host ""
}
