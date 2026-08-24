# scripts/bench/measure_windows.ps1
#
# Measures wall-clock time and peak working set of one command on Windows --
# the native-side counterpart of `/usr/bin/time -v` in
# kmc3_fastk_comparison.sh. Peak working set is sampled by polling
# Process.PeakWorkingSet64 (a cumulative OS-maintained maximum, so polling
# cannot miss a spike between samples; the poll only needs to read it at
# least once near the end of the run, and the loop reads it every 100 ms).
#
# Usage:
#   powershell -File measure_windows.ps1 -Exe path\to\fastdna.exe -Args "-i in.fastq -o out.parquet -k 31"
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$Args,
    [string]$Label = "run"
)

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Exe
$psi.Arguments = $Args
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi

$stdout = New-Object System.Text.StringBuilder
$stderr = New-Object System.Text.StringBuilder
$outEvent = Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -Action {
    if ($EventArgs.Data) { [void]$Event.MessageData.AppendLine($EventArgs.Data) }
} -MessageData $stdout
$errEvent = Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived -Action {
    if ($EventArgs.Data) { [void]$Event.MessageData.AppendLine($EventArgs.Data) }
} -MessageData $stderr

$sw = [System.Diagnostics.Stopwatch]::StartNew()
[void]$proc.Start()
$proc.BeginOutputReadLine()
$proc.BeginErrorReadLine()

$peak = 0L
while (-not $proc.HasExited) {
    try {
        $proc.Refresh()
        if ($proc.PeakWorkingSet64 -gt $peak) { $peak = $proc.PeakWorkingSet64 }
    } catch {}
    Start-Sleep -Milliseconds 100
}
$proc.WaitForExit()
$sw.Stop()

Unregister-Event -SourceIdentifier $outEvent.Name
Unregister-Event -SourceIdentifier $errEvent.Name

$result = [PSCustomObject]@{
    label            = $Label
    exit_code        = $proc.ExitCode
    wall_seconds     = [Math]::Round($sw.Elapsed.TotalSeconds, 2)
    peak_rss_bytes   = $peak
    peak_rss_human   = "{0:N2} GB" -f ($peak / 1GB)
}
$result | ConvertTo-Json
Write-Host "--- stdout tail ---"
($stdout.ToString() -split "`n" | Select-Object -Last 8) -join "`n" | Write-Host
if ($proc.ExitCode -ne 0) {
    Write-Host "--- stderr ---"
    Write-Host $stderr.ToString()
}
