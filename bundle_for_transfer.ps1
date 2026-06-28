# bundle_for_transfer.ps1
# Portable ZIP of Cable Transcribe for another PC (code + optional meeting folders).
param(
    [string[]]$IncludeMeeting = @(),
    [string]$OutDir = "$env:USERPROFILE\Desktop",
    [switch]$IncludeAllMeetings
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Stamp = Get-Date -Format "yyyy-MM-dd"
$ZipName = "cable-transcribe-portable-$Stamp.zip"
$ZipPath = Join-Path $OutDir $ZipName
$Stage = Join-Path $env:TEMP "cable-transcribe-portable-$Stamp"

if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage | Out-Null

$CodeFiles = @(
    "cable_transcribe.py",
    "live_dictate.py",
    "transcribe_recording.py",
    "format_meeting.py",
    "_run_whatsapp_file.py",
    "chat_cleaner.py",
    "list_devices.py",
    "smoke_test_dictate.py",
    "requirements.txt",
    "README.md",
    "SETUP_OTHER_PC.md",
    "open_meeting.bat",
    "restart_dictate.bat",
    "import_recording.bat",
    "install.bat",
    "sample_chat_paste.txt"
)

foreach ($f in $CodeFiles) {
    $src = Join-Path $Root $f
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $Stage $f)
    }
}

# Empty meetings scaffold + guide
$MeetingsStage = Join-Path $Stage "meetings"
New-Item -ItemType Directory -Path $MeetingsStage | Out-Null
$guide = Join-Path $Root "meetings\SCRIBE_DOCUMENTATION_GUIDE.md"
if (Test-Path $guide) {
    Copy-Item $guide (Join-Path $MeetingsStage "SCRIBE_DOCUMENTATION_GUIDE.md")
}

function Copy-MeetingFolder {
    param([string]$Name)
    $src = Join-Path $Root "meetings\$Name"
    if (-not (Test-Path $src)) {
        Write-Warning "Meeting folder not found: $Name"
        return
    }
    $dest = Join-Path $MeetingsStage $Name
    Write-Host "Including meeting: $Name"
    robocopy $src $dest /E /NJH /NJS /NP /XD "__pycache__" | Out-Null
}

if ($IncludeAllMeetings) {
    Get-ChildItem (Join-Path $Root "meetings") -Directory |
        Where-Object { $_.Name -notlike "archived_*" } |
        ForEach-Object { Copy-MeetingFolder $_.Name }
} else {
    foreach ($m in $IncludeMeeting) {
        Copy-MeetingFolder $m
    }
}

# Current meeting pointer if that folder was included
$current = Join-Path $Root "current_meeting.json"
if (Test-Path $current) {
    $ptr = Get-Content $current -Raw | ConvertFrom-Json
    $folderName = Split-Path $ptr.folder -Leaf
    $included = Test-Path (Join-Path $MeetingsStage $folderName)
    if ($included) {
        Copy-Item $current (Join-Path $Stage "current_meeting.json")
    }
}

if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $ZipPath -Force
Remove-Item $Stage -Recurse -Force

Write-Host ""
Write-Host "Bundle ready:" -ForegroundColor Green
Write-Host "  $ZipPath"
Write-Host ""
Write-Host "On the other PC: unzip, run install.bat, read SETUP_OTHER_PC.md"
Write-Host ""
Write-Host "Examples:"
Write-Host "  .\bundle_for_transfer.ps1"
Write-Host "  .\bundle_for_transfer.ps1 -IncludeMeeting '2026-06-03_203457_whatsapp-2026-06-03-505-pm'"
