<#
.SYNOPSIS
    Builds the Fundscope installer, from a fresh checkout to dist\Fundscope-Setup-<version>.exe.

.DESCRIPTION
    1. Python environment  .venv (created with "py -3.13" and requirements-dev.txt when missing)
    2. Embedding model     verified offline; downloaded and quantized by tools\fetch_model.py when missing
    3. Icon, sample data   generated when missing
    4. Tests               pytest (skip with -SkipTests)
    5. Bundle              PyInstaller one-folder build -> dist\Fundscope
    6. Self-test           the bundled Fundscope.exe --selftest must pass
    7. Installer           Inno Setup 6 -> dist\Fundscope-Setup-<version>.exe

    Works in Windows PowerShell 5.1 and PowerShell 7:
        powershell -ExecutionPolicy Bypass -File packaging\build.ps1 [-SkipTests]

.PARAMETER SkipTests
    Skip the pytest suite.
#>
[CmdletBinding()]
param(
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Bundle = Join-Path $Root 'dist\Fundscope'
$AppExe = Join-Path $Bundle 'Fundscope.exe'
$SelftestDir = Join-Path $Root 'build\selftest'
$WarnFile = Join-Path $Root 'build\pyinstaller\semdb\warn-semdb.txt'
$Started = Get-Date

function Write-Step([string]$Number, [string]$Text) {
    Write-Host ''
    Write-Host "[$Number/7] $Text" -ForegroundColor Cyan
}

function Write-Detail([string]$Text, [string]$Color = 'Gray') {
    Write-Host "      $Text" -ForegroundColor $Color
}

function Invoke-Program([string]$FilePath, [string[]]$Arguments) {
    # Streams the program's output and returns its exit code. Windows PowerShell 5.1 turns the
    # stderr lines of native programs into terminating errors under 'Stop' (PyInstaller and pip
    # log to stderr), so the preference is relaxed while the program runs.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $FilePath @Arguments 2>&1 | ForEach-Object { Write-Detail "$_" }
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Invoke-Native([string]$FilePath, [string[]]$Arguments) {
    $code = Invoke-Program $FilePath $Arguments
    if ($code -ne 0) {
        throw "'$(Split-Path -Leaf $FilePath) $($Arguments -join ' ')' failed with exit code $code."
    }
}

function Find-InnoSetupCompiler {
    $bases = @((Join-Path $env:LOCALAPPDATA 'Programs'), ${env:ProgramFiles(x86)}, $env:ProgramFiles)
    foreach ($base in $bases) {
        if ($base) {
            $candidate = Join-Path $base 'Inno Setup 6\ISCC.exe'
            if (Test-Path $candidate) { return $candidate }
        }
    }
    throw 'Inno Setup 6 was not found. Install it (winget install JRSoftware.InnoSetup) and run this script again.'
}

Push-Location $Root
$previousUnbuffered = $env:PYTHONUNBUFFERED
$env:PYTHONUNBUFFERED = '1'  # stream Python output as it happens
try {
    $Version = (Select-String -Path 'src\semdb\__init__.py' -Pattern '__version__ = "([^"]+)"').Matches[0].Groups[1].Value
    Write-Host "Building Fundscope $Version" -ForegroundColor White

    Write-Step 1 'Python environment'
    if (-not (Test-Path $Python)) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw 'Python 3.13 with the "py" launcher is required: install it from https://www.python.org and run this script again.'
        }
        Write-Detail 'Creating .venv with Python 3.13'
        Invoke-Native 'py' @('-3.13', '-m', 'venv', '.venv')
        Invoke-Native $Python @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', 'requirements-dev.txt')
    }
    Invoke-Native $Python @('--version')

    Write-Step 2 'Embedding model'
    if ((Invoke-Program $Python @('tools\fetch_model.py', '--check')) -ne 0) {
        Write-Detail 'Model missing or invalid: downloading and quantizing it (needs internet, a few minutes)' 'Yellow'
        Invoke-Native $Python @('tools\fetch_model.py')
    }

    Write-Step 3 'App icon and sample workbook'
    if (@('icon.ico', 'icon.png', 'logo.svg' | Where-Object { -not (Test-Path "assets\$_") }).Count -gt 0) {
        Invoke-Native $Python @('tools\make_icon.py')
    } else {
        Write-Detail 'Icons present.'
    }
    $sample = 'assets\sample\sample_funds.xlsx'
    if (-not (Test-Path $sample)) {
        Invoke-Native $Python @('tools\make_sample_data.py')
        if (-not (Test-Path $sample)) { throw "tools\make_sample_data.py did not create $sample." }
    } else {
        Write-Detail 'Sample workbook present.'
    }

    if ($SkipTests) {
        Write-Step 4 'Tests skipped (-SkipTests)'
    } else {
        Write-Step 4 'Tests'
        Invoke-Native $Python @('-m', 'pytest')
    }

    Write-Step 5 'Bundling with PyInstaller'
    Invoke-Native $Python @('-m', 'PyInstaller', '--noconfirm', '--clean', '--log-level', 'WARN',
        'packaging\semdb.spec', '--distpath', 'dist', '--workpath', 'build\pyinstaller')
    # An excluded module that the app imports would only fail on the user's machine.
    $needed = @(Select-String -Path $WarnFile -Pattern '^excluded module named (\S+) - imported by .*\bsemdb\b')
    if ($needed.Count -gt 0) {
        $needed | ForEach-Object { Write-Detail $_.Line 'Red' }
        throw 'The app imports modules that packaging\semdb.spec excludes.'
    }
    $bytes = (Get-ChildItem $Bundle -Recurse -File | Measure-Object -Property Length -Sum).Sum
    Write-Detail ('{0} ({1:N0} MB)' -f $Bundle, ($bytes / 1MB))

    Write-Step 6 'Self-test of the bundled app'
    if (Test-Path $SelftestDir) { Remove-Item $SelftestDir -Recurse -Force }
    $process = Start-Process -FilePath $AppExe -ArgumentList @('--selftest', '--data-dir', "`"$SelftestDir`"") -PassThru
    $null = $process.Handle  # Windows PowerShell only reports ExitCode for processes whose handle was opened
    if (-not $process.WaitForExit(15 * 60 * 1000)) {
        $process.Kill()
        throw 'The self-test did not finish within 15 minutes.'
    }
    $log = Join-Path $SelftestDir 'logs\selftest.log'
    if (Test-Path $log) {
        Get-Content $log -Encoding UTF8 | ForEach-Object { Write-Detail $_ }
    } else {
        Write-Detail "No report was written to $log" 'Red'
    }
    if ($process.ExitCode -ne 0) {
        throw "The bundled app failed its self-test (exit code $($process.ExitCode))."
    }

    Write-Step 7 'Installer (Inno Setup)'
    Invoke-Native (Find-InnoSetupCompiler) @('/Qp', "/DAppVersion=$Version", 'packaging\installer.iss')
    $installer = Get-Item "dist\Fundscope-Setup-$Version.exe"

    Write-Host ''
    Write-Host ('Installer: {0}' -f $installer.FullName) -ForegroundColor Green
    Write-Host ('Size:      {0:N1} MB' -f ($installer.Length / 1MB)) -ForegroundColor Green
    Write-Host ('Built in {0:mm\:ss}' -f ((Get-Date) - $Started))
} catch {
    Write-Host ''
    Write-Host "BUILD FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    $env:PYTHONUNBUFFERED = $previousUnbuffered
    Pop-Location
}
