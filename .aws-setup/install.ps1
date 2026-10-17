# AWS CLI v2 install script for Windows (PowerShell 5.1+).
#
# Defaults to a user-local install under %LOCALAPPDATA%. Use -System for the
# traditional Program Files layout (requires admin). See -Help for details.

param(
    [string] $Version,  # specific version to install; empty means "latest"
    [switch] $System,   # install system-wide (Program Files); requires admin
    [switch] $Quiet,    # suppress non-essential stdout
    [switch] $Help
)

# Stop on any error so failures don't silently continue.
$ErrorActionPreference = 'Stop'
# PowerShell 5.1 defaults to TLS 1.0; force 1.2 for the CDN.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# ============================================================================
# Configuration
# ============================================================================

$DownloadBaseUrl    = 'https://awscli.amazonaws.com'
$LatestVersionUrl   = "$DownloadBaseUrl/v2/version.txt"

# The system MSI always installs to the 64-bit Program Files. Use
# ProgramW6432, which points there even from a 32-bit PowerShell host;
# $env:ProgramFiles would resolve to "Program Files (x86)" under 32-bit
# and make the post-install check look in the wrong place.
$SystemProgramFiles = $env:ProgramW6432
if (-not $SystemProgramFiles) { $SystemProgramFiles = $env:ProgramFiles }
$SystemInstallDir   = Join-Path $SystemProgramFiles 'Amazon\AWSCLIV2'
$UserInstallDir     = Join-Path $env:LOCALAPPDATA 'Programs\Amazon\AWSCLIV2'

$DistributionSource = 'script-exe'
if ($env:AWS_CLI_DISTRIBUTION_SOURCE_OVERRIDE) {
    $DistributionSource = $env:AWS_CLI_DISTRIBUTION_SOURCE_OVERRIDE
}
$DownloadQuery = "?src=$DistributionSource"

$MsiexecSuccessCodes = @(0, 1641, 3010)

# ============================================================================
# Output helpers
# ============================================================================

$EscRed    = [char]27 + '[31m'
$EscYellow = [char]27 + '[33m'
$EscGreen  = [char]27 + '[32m'
$EscReset  = [char]27 + '[0m'

function Use-Color {
    param([bool] $IsErrStream)
    if ($env:NO_COLOR) { return $false }
    if ($IsErrStream) { return -not [Console]::IsErrorRedirected }
    return -not [Console]::IsOutputRedirected
}

# Progress/info on stdout, silenced by -Quiet.
function Write-Info {
    param([string] $Message)
    if (-not $Quiet) { Write-Host $Message }
}

# Green success on stdout, silenced by -Quiet.
function Write-Success {
    param([string] $Message)
    if ($Quiet) { return }
    if (Use-Color $false) {
        Write-Host "$EscGreen$Message$EscReset"
    } else {
        Write-Host $Message
    }
}

# Yellow warning on stderr; never silenced.
function Write-Warn {
    param([string] $Message)
    if (Use-Color $true) {
        [Console]::Error.WriteLine("${EscYellow}aws-cli installer: warning:${EscReset} $Message")
    } else {
        [Console]::Error.WriteLine("aws-cli installer: warning: $Message")
    }
}

# Red error on stderr; never silenced.
function Write-Err {
    param([string] $Message)
    if (Use-Color $true) {
        [Console]::Error.WriteLine("${EscRed}aws-cli installer: error:${EscReset} $Message")
    } else {
        [Console]::Error.WriteLine("aws-cli installer: error: $Message")
    }
}

# Record the exit code and throw to unwind. We never call `exit` here: under
# `irm | iex` the script runs in the caller's session, so `exit` would close
# the user's PowerShell. The error is printed by the top-level handler (which
# also reports any other exception), so nothing is silently swallowed; the
# recorded code becomes a process exit status only when run as a file.
$Script:ExitCode = 0
function Throw-Error {
    param([int] $Code, [string] $Message)
    $Script:ExitCode = $Code
    throw $Message
}

# ============================================================================
# Argument validation
# ============================================================================

function Show-Help {
    @"
Usage: install.ps1 [-Version <X.Y.Z>] [-System] [-Quiet] [-Help]

Install or update the AWS CLI v2.

Parameters:
  -Version <X.Y.Z>  Install a specific fully-qualified version. Otherwise
                    installs the latest release. Downgrades are not
                    supported. To install a previous version, see:
                    https://docs.aws.amazon.com/cli/latest/userguide/getting-started-version.html
  -System           Install system-wide (requires admin). Otherwise installs
                    under %LOCALAPPDATA%\Programs\Amazon\AWSCLIV2\.
  -Quiet            Suppress non-essential output. Errors and warnings are
                    still printed to stderr.
  -Help             Show this help and exit.
"@ | Write-Host
}

# Accept only fully-qualified MAJOR.MINOR.PATCH.
function Test-Semver {
    param([string] $Value)
    return ($Value -match '^\d+\.\d+\.\d+$')
}

# Returns $false when the run should stop early without error.
function Validate-Args {
    if ($Help) { Show-Help; return $false }
    if ($Version -and -not (Test-Semver $Version)) {
        Throw-Error 2 "-Version must be fully-qualified semver (e.g. 2.27.41), got: $Version"
    }
    return $true
}

# ============================================================================
# Pre-install checks
# ============================================================================

function Test-Windows {
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        return $IsWindows
    }
    return $true
}

function Assert-Platform {
    if (-not (Test-Windows)) {
        Throw-Error 1 'unsupported OS. install.ps1 supports only Windows; use install.sh on Linux/macOS.'
    }
}

function Test-IsAdmin {
    $identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Assert-SystemPrivilege {
    if ($System -and -not (Test-IsAdmin)) {
        Throw-Error 1 '-System requires admin privileges. Re-run from an elevated PowerShell, or omit -System for a user-local install.'
    }
}

# ============================================================================
# Path resolution
# ============================================================================

$Script:InstallDir = $null

function Resolve-InstallPath {
    if ($System) {
        $Script:InstallDir = $SystemInstallDir
    } else {
        if (-not $env:LOCALAPPDATA) {
            Throw-Error 2 'LOCALAPPDATA is not set; cannot resolve user-local install path.'
        }
        $Script:InstallDir = $UserInstallDir
    }
}

# ============================================================================
# Download
# ============================================================================

$Script:TempDir = $null   # working dir; removed in the finally block

function New-TempDir {
    $Script:TempDir = Join-Path $env:TEMP "aws-cli-install-$([guid]::NewGuid().Guid)"
    New-Item -ItemType Directory -Path $Script:TempDir -Force | Out-Null
}

function Remove-TempDir {
    if ($Script:TempDir -and (Test-Path $Script:TempDir)) {
        try {
            Remove-Item -Recurse -Force $Script:TempDir -ErrorAction Stop
        } catch {
            Write-Warn "could not remove temp directory: $Script:TempDir"
        }
    }
}

function Get-InstallerFilename {
    if ($System) {
        if ($Script:TargetVersion) {
            return "AWSCLIV2-$($Script:TargetVersion).msi"
        }
        return 'AWSCLIV2.msi'
    }
    if ($Script:TargetVersion) {
        return "AWSCLIV2-User-$($Script:TargetVersion).msi"
    }
    return 'AWSCLIV2-User.msi'
}

function Get-InstallerUrl {
    return "$DownloadBaseUrl/$(Get-InstallerFilename)$DownloadQuery"
}

function Download-File {
    param([string] $Url, [string] $Out)
    Invoke-WebRequest -Uri $Url -OutFile $Out -UseBasicParsing -TimeoutSec 60
}

function Download-WithRetry {
    param([string] $Url, [string] $Out, [string] $Label)
    Write-Info "Downloading $Label from $Url"
    try {
        Download-File -Url $Url -Out $Out
        return
    } catch {
        Write-Warn "download of $Label failed; retrying once"
    }
    try {
        Download-File -Url $Url -Out $Out
    } catch {
        Throw-Error 1 "failed to download $Label from $Url after one retry: $($_.Exception.Message)"
    }
}

$Script:InstallerPath = $null

function Download-Installer {
    $Script:InstallerPath = Join-Path $Script:TempDir 'AWSCLIV2.msi'
    Download-WithRetry -Url (Get-InstallerUrl) -Out $Script:InstallerPath -Label 'AWS CLI installer'
}

# ============================================================================
# Signature verification
# ============================================================================

function Verify-Installer {
    $sig = Get-AuthenticodeSignature -FilePath $Script:InstallerPath
    if ($sig.Status -ne 'Valid') {
        Throw-Error 1 ("Authenticode signature verification failed for " +
            "$($Script:InstallerPath): status is '$($sig.Status)'. " +
            'Refusing to install.')
    }

    Write-Success 'MSI Authenticode signature verified.'
}

# ============================================================================
# Version resolution
# ============================================================================

# Extracts the semver from `aws --version`.
function Read-InstalledVersion {
    param([string] $AwsExe)
    if (-not (Test-Path $AwsExe)) { return $null }
    try {
        $out = & $AwsExe --version 2>$null
    } catch {
        return $null
    }
    if ($out -match '^aws-cli/(\d+\.\d+\.\d+)') {
        return $Matches[1]
    }
    return $null
}

function Get-CandidateAwsBinary {
    return Join-Path $Script:InstallDir 'aws.exe'
}

$Script:PreInstallVersion  = $null   # version present before install
$Script:TargetVersion      = $null   # version we resolved to install
$Script:PostInstallVersion = $null   # version present after install

function Capture-PreInstallVersion {
    $Script:PreInstallVersion = Read-InstalledVersion (Get-CandidateAwsBinary)
}

# Best-effort fetch of the published "latest" version; $null on any failure.
function Fetch-LatestVersion {
    try {
        $resp = Invoke-WebRequest -Uri $LatestVersionUrl -UseBasicParsing -TimeoutSec 10
        $text = ([string] $resp.Content).Trim()
        if (Test-Semver $text) { return $text }
    } catch {}
    return $null
}

function Resolve-TargetAndAnnounce {
    if ($Version) {
        $Script:TargetVersion = $Version
    } else {
        $Script:TargetVersion = Fetch-LatestVersion
    }

    if (-not $Script:TargetVersion -or -not $Script:PreInstallVersion) {
        if ($Script:TargetVersion) {
            Write-Info "Installing AWS CLI $($Script:TargetVersion)"
        }
        return $true
    }

    if ($Script:TargetVersion -eq $Script:PreInstallVersion) {
        Write-Info "AWS CLI $($Script:TargetVersion) is already installed at $($Script:InstallDir); nothing to do."
        return $false
    }

    if ([version]$Script:TargetVersion -lt [version]$Script:PreInstallVersion) {
        Write-Info "AWS CLI $($Script:PreInstallVersion) is already installed at $($Script:InstallDir). To install $($Script:TargetVersion), uninstall the existing AWS CLI at $($Script:InstallDir) and try again. See https://docs.aws.amazon.com/cli/latest/userguide/uninstall.html"
        return $false
    }

    Write-Info "Installing AWS CLI $($Script:PreInstallVersion) -> $($Script:TargetVersion)"
    return $true
}

# ============================================================================
# Install
# ============================================================================

function Run-Installer {
    $msiFilename = Split-Path $Script:InstallerPath -Leaf
    $msiDir = Split-Path $Script:InstallerPath -Parent

    $msiArgs = @('/i', $msiFilename, '/qn', '/norestart')

    Write-Info 'Running MSI installer...'
    $proc = Start-Process -FilePath 'msiexec' -ArgumentList $msiArgs `
                          -WorkingDirectory $msiDir -Wait -PassThru
    if ($MsiexecSuccessCodes -notcontains $proc.ExitCode) {
        Throw-Error 1 "msiexec failed with exit code $($proc.ExitCode)"
    }
}

# ============================================================================
# Post-install
# ============================================================================

function Verify-InstallRuns {
    $awsExe = Get-CandidateAwsBinary
    $Script:PostInstallVersion = Read-InstalledVersion $awsExe
    if (-not $Script:PostInstallVersion) {
        Throw-Error 1 "post-install check failed: '$awsExe --version' did not run successfully."
    }
}

# Writes install-time metadata next to the bundled metadata.json. `aws update`
# reads this back to learn where and how the CLI was installed. No bin_dir
# field - on Windows the MSI manages PATH, there's no symlink dir to record.
function Write-InstallJson {
    $installJson = Join-Path $Script:InstallDir 'awscli\data\install.json'
    $parent = Split-Path $installJson -Parent
    if (-not (Test-Path $parent)) { return }

    $obj = [ordered]@{
        distribution_source = $DistributionSource
        install_dir         = $Script:InstallDir
        script_install      = [ordered]@{
            system           = [bool]$System
            version_resolved = $Script:PostInstallVersion
            quiet            = [bool]$Quiet
        }
    }
    $json = $obj | ConvertTo-Json -Depth 3
    [IO.File]::WriteAllText($installJson, $json, (New-Object Text.UTF8Encoding $false))
}

function Invoke-PostInstallChecks {
    Verify-InstallRuns
    Write-InstallJson
    Write-Success "AWS CLI $($Script:PostInstallVersion) installed to $($Script:InstallDir)"
    Write-Info "When multiple installs exist, PATH order determines which 'aws' runs."
}

# ============================================================================
# Entry point
# ============================================================================

function Invoke-Main {
    try {
        # Pre-flight: validate args and environment before touching disk/network.
        if (-not (Validate-Args)) { return }
        Assert-Platform
        Assert-SystemPrivilege
        Resolve-InstallPath

        # Install: stage, resolve target (may short-circuit), download, run.
        New-TempDir
        Capture-PreInstallVersion
        if (-not (Resolve-TargetAndAnnounce)) { return }
        Download-Installer
        Verify-Installer
        Run-Installer
        Invoke-PostInstallChecks
    } finally {
        # Always remove the temp dir, even on error or early return.
        Remove-TempDir
    }
}

# Run the body inside a scriptblock and catch errors here rather than calling
# `exit` from deep in the script. Under `irm | iex` the script shares the
# caller's session, so `exit` would close the user's PowerShell window. All
# errors are printed here -- both Throw-Error (which sets $Script:ExitCode)
# and any other exception raised under $ErrorActionPreference = 'Stop' -- so
# nothing is silently swallowed.
& {
    try {
        Invoke-Main
    } catch {
        Write-Err $_.Exception.Message
        if ($Script:ExitCode -eq 0) { $Script:ExitCode = 1 }
    }
}

# Only set a real process exit status when invoked as a file (powershell
# -File, or .\install.ps1), where $PSCommandPath is populated. Under
# `irm | iex` $PSCommandPath is empty and calling `exit` would terminate the
# user's shell, so we leave it alone there.
if ($PSCommandPath) {
    exit $Script:ExitCode
}

# SIG # Begin signature block
# MIIuqQYJKoZIhvcNAQcCoIIumjCCLpYCAQExDzANBglghkgBZQMEAgEFADB5Bgor
# BgEEAYI3AgEEoGswaTA0BgorBgEEAYI3AgEeMCYCAwEAAAQQH8w7YFlLCE63JNLG
# KX7zUQIBAAIBAAIBAAIBAAIBADAxMA0GCWCGSAFlAwQCAQUABCAiHpMnZP1SUvbs
# 9uFFAb6fnPVwW5unYwUVeW/m7eBVTKCCE98wggXAMIIEqKADAgECAhAP0bvKeWvX
# +N1MguEKmpYxMA0GCSqGSIb3DQEBCwUAMGwxCzAJBgNVBAYTAlVTMRUwEwYDVQQK
# EwxEaWdpQ2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5jb20xKzApBgNV
# BAMTIkRpZ2lDZXJ0IEhpZ2ggQXNzdXJhbmNlIEVWIFJvb3QgQ0EwHhcNMjIwMTEz
# MDAwMDAwWhcNMzExMTA5MjM1OTU5WjBiMQswCQYDVQQGEwJVUzEVMBMGA1UEChMM
# RGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3d3cuZGlnaWNlcnQuY29tMSEwHwYDVQQD
# ExhEaWdpQ2VydCBUcnVzdGVkIFJvb3QgRzQwggIiMA0GCSqGSIb3DQEBAQUAA4IC
# DwAwggIKAoICAQC/5pBzaN675F1KPDAiMGkz7MKnJS7JIT3yithZwuEppz1Yq3aa
# za57G4QNxDAf8xukOBbrVsaXbR2rsnnyyhHS5F/WBTxSD1Ifxp4VpX6+n6lXFllV
# cq9ok3DCsrp1mWpzMpTREEQQLt+C8weE5nQ7bXHiLQwb7iDVySAdYyktzuxeTsiT
# +CFhmzTrBcZe7FsavOvJz82sNEBfsXpm7nfISKhmV1efVFiODCu3T6cw2Vbuyntd
# 463JT17lNecxy9qTXtyOj4DatpGYQJB5w3jHtrHEtWoYOAMQjdjUN6QuBX2I9YI+
# EJFwq1WCQTLX2wRzKm6RAXwhTNS8rhsDdV14Ztk6MUSaM0C/CNdaSaTC5qmgZ92k
# J7yhTzm1EVgX9yRcRo9k98FpiHaYdj1ZXUJ2h4mXaXpI8OCiEhtmmnTK3kse5w5j
# rubU75KSOp493ADkRSWJtppEGSt+wJS00mFt6zPZxd9LBADMfRyVw4/3IbKyEbe7
# f/LVjHAsQWCqsWMYRJUadmJ+9oCw++hkpjPRiQfhvbfmQ6QYuKZ3AeEPlAwhHbJU
# KSWJbOUOUlFHdL4mrLZBdd56rF+NP8m800ERElvlEFDrMcXKchYiCd98THU/Y+wh
# X8QgUWtvsauGi0/C1kVfnSD8oR7FwI+isX4KJpn15GkvmB0t9dmpsh3lGwIDAQAB
# o4IBZjCCAWIwDwYDVR0TAQH/BAUwAwEB/zAdBgNVHQ4EFgQU7NfjgtJxXWRM3y5n
# P+e6mK4cD08wHwYDVR0jBBgwFoAUsT7DaQP4v0cB1JgmGggC72NkK8MwDgYDVR0P
# AQH/BAQDAgGGMBMGA1UdJQQMMAoGCCsGAQUFBwMDMH8GCCsGAQUFBwEBBHMwcTAk
# BggrBgEFBQcwAYYYaHR0cDovL29jc3AuZGlnaWNlcnQuY29tMEkGCCsGAQUFBzAC
# hj1odHRwOi8vY2FjZXJ0cy5kaWdpY2VydC5jb20vRGlnaUNlcnRIaWdoQXNzdXJh
# bmNlRVZSb290Q0EuY3J0MEsGA1UdHwREMEIwQKA+oDyGOmh0dHA6Ly9jcmwzLmRp
# Z2ljZXJ0LmNvbS9EaWdpQ2VydEhpZ2hBc3N1cmFuY2VFVlJvb3RDQS5jcmwwHAYD
# VR0gBBUwEzAHBgVngQwBAzAIBgZngQwBBAEwDQYJKoZIhvcNAQELBQADggEBAEHx
# qRH0DxNHecllao3A7pgEpMbjDPKisedfYk/ak1k2zfIe4R7sD+EbP5HU5A/C5pg0
# /xkPZigfT2IxpCrhKhO61z7H0ZL+q93fqpgzRh9Onr3g7QdG64AupP2uU7SkwaT1
# IY1rzAGt9Rnu15ClMlIr28xzDxj4+87eg3Gn77tRWwR2L62t0+od/P1Tk+WMieNg
# GbngLyOOLFxJy34riDkruQZhiPOuAnZ2dMFkkbiJUZflhX0901emWG4f7vtpYeJa
# 3Cgh6GO6Ps9W7Zrk9wXqyvPsEt84zdp7PiuTUy9cUQBY3pBIowrHC/Q7bVUx8ALM
# R3eWUaNetbxcyEMRoacwggawMIIEmKADAgECAhAIrUCyYNKcTJ9ezam9k67ZMA0G
# CSqGSIb3DQEBDAUAMGIxCzAJBgNVBAYTAlVTMRUwEwYDVQQKEwxEaWdpQ2VydCBJ
# bmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5jb20xITAfBgNVBAMTGERpZ2lDZXJ0
# IFRydXN0ZWQgUm9vdCBHNDAeFw0yMTA0MjkwMDAwMDBaFw0zNjA0MjgyMzU5NTla
# MGkxCzAJBgNVBAYTAlVTMRcwFQYDVQQKEw5EaWdpQ2VydCwgSW5jLjFBMD8GA1UE
# AxM4RGlnaUNlcnQgVHJ1c3RlZCBHNCBDb2RlIFNpZ25pbmcgUlNBNDA5NiBTSEEz
# ODQgMjAyMSBDQTEwggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQDVtC9C
# 0CiteLdd1TlZG7GIQvUzjOs9gZdwxbvEhSYwn6SOaNhc9es0JAfhS0/TeEP0F9ce
# 2vnS1WcaUk8OoVf8iJnBkcyBAz5NcCRks43iCH00fUyAVxJrQ5qZ8sU7H/Lvy0da
# E6ZMswEgJfMQ04uy+wjwiuCdCcBlp/qYgEk1hz1RGeiQIXhFLqGfLOEYwhrMxe6T
# SXBCMo/7xuoc82VokaJNTIIRSFJo3hC9FFdd6BgTZcV/sk+FLEikVoQ11vkunKoA
# FdE3/hoGlMJ8yOobMubKwvSnowMOdKWvObarYBLj6Na59zHh3K3kGKDYwSNHR7Oh
# D26jq22YBoMbt2pnLdK9RBqSEIGPsDsJ18ebMlrC/2pgVItJwZPt4bRc4G/rJvmM
# 1bL5OBDm6s6R9b7T+2+TYTRcvJNFKIM2KmYoX7BzzosmJQayg9Rc9hUZTO1i4F4z
# 8ujo7AqnsAMrkbI2eb73rQgedaZlzLvjSFDzd5Ea/ttQokbIYViY9XwCFjyDKK05
# huzUtw1T0PhH5nUwjewwk3YUpltLXXRhTT8SkXbev1jLchApQfDVxW0mdmgRQRNY
# mtwmKwH0iU1Z23jPgUo+QEdfyYFQc4UQIyFZYIpkVMHMIRroOBl8ZhzNeDhFMJlP
# /2NPTLuqDQhTQXxYPUez+rbsjDIJAsxsPAxWEQIDAQABo4IBWTCCAVUwEgYDVR0T
# AQH/BAgwBgEB/wIBADAdBgNVHQ4EFgQUaDfg67Y7+F8Rhvv+YXsIiGX0TkIwHwYD
# VR0jBBgwFoAU7NfjgtJxXWRM3y5nP+e6mK4cD08wDgYDVR0PAQH/BAQDAgGGMBMG
# A1UdJQQMMAoGCCsGAQUFBwMDMHcGCCsGAQUFBwEBBGswaTAkBggrBgEFBQcwAYYY
# aHR0cDovL29jc3AuZGlnaWNlcnQuY29tMEEGCCsGAQUFBzAChjVodHRwOi8vY2Fj
# ZXJ0cy5kaWdpY2VydC5jb20vRGlnaUNlcnRUcnVzdGVkUm9vdEc0LmNydDBDBgNV
# HR8EPDA6MDigNqA0hjJodHRwOi8vY3JsMy5kaWdpY2VydC5jb20vRGlnaUNlcnRU
# cnVzdGVkUm9vdEc0LmNybDAcBgNVHSAEFTATMAcGBWeBDAEDMAgGBmeBDAEEATAN
# BgkqhkiG9w0BAQwFAAOCAgEAOiNEPY0Idu6PvDqZ01bgAhql+Eg08yy25nRm95Ry
# sQDKr2wwJxMSnpBEn0v9nqN8JtU3vDpdSG2V1T9J9Ce7FoFFUP2cvbaF4HZ+N3HL
# IvdaqpDP9ZNq4+sg0dVQeYiaiorBtr2hSBh+3NiAGhEZGM1hmYFW9snjdufE5Btf
# Q/g+lP92OT2e1JnPSt0o618moZVYSNUa/tcnP/2Q0XaG3RywYFzzDaju4ImhvTnh
# OE7abrs2nfvlIVNaw8rpavGiPttDuDPITzgUkpn13c5UbdldAhQfQDN8A+KVssIh
# dXNSy0bYxDQcoqVLjc1vdjcshT8azibpGL6QB7BDf5WIIIJw8MzK7/0pNVwfiThV
# 9zeKiwmhywvpMRr/LhlcOXHhvpynCgbWJme3kuZOX956rEnPLqR0kq3bPKSchh/j
# wVYbKyP/j7XqiHtwa+aguv06P0WmxOgWkVKLQcBIhEuWTatEQOON8BUozu3xGFYH
# Ki8QxAwIZDwzj64ojDzLj4gLDb879M4ee47vtevLt/B3E+bnKD+sEq6lLyJsQfmC
# XBVmzGwOysWGw/YmMwwHS6DTBwJqakAwSEs0qFEgu60bhQjiWQ1tygVQK+pKHJ6l
# /aCnHwZ05/LWUpD9r4VIIflXO7ScA+2GRfS0YW6/aOImYIbqyK+p/pQd52MbOoZW
# eE4wggdjMIIFS6ADAgECAhAFkMTPDbAUGZlTpoz86bDnMA0GCSqGSIb3DQEBCwUA
# MGkxCzAJBgNVBAYTAlVTMRcwFQYDVQQKEw5EaWdpQ2VydCwgSW5jLjFBMD8GA1UE
# AxM4RGlnaUNlcnQgVHJ1c3RlZCBHNCBDb2RlIFNpZ25pbmcgUlNBNDA5NiBTSEEz
# ODQgMjAyMSBDQTEwHhcNMjYwNDEzMDAwMDAwWhcNMjcwNDEyMjM1OTU5WjCB6zET
# MBEGCysGAQQBgjc8AgEDEwJVUzEZMBcGCysGAQQBgjc8AgECEwhEZWxhd2FyZTEd
# MBsGA1UEDwwUUHJpdmF0ZSBPcmdhbml6YXRpb24xEDAOBgNVBAUTBzQxNTI5NTQx
# CzAJBgNVBAYTAlVTMRMwEQYDVQQIEwpXYXNoaW5ndG9uMRAwDgYDVQQHEwdTZWF0
# dGxlMSIwIAYDVQQKExlBbWF6b24gV2ViIFNlcnZpY2VzLCBJbmMuMQwwCgYDVQQL
# EwNBV1MxIjAgBgNVBAMTGUFtYXpvbiBXZWIgU2VydmljZXMsIEluYy4wggGiMA0G
# CSqGSIb3DQEBAQUAA4IBjwAwggGKAoIBgQCdnV6w7tV727PaCeTlAamSDuzpMtRI
# BOhJ12XroPRlI0DEX54YlUOG98dBqsMrLyKn/Ftc/z00ZjOB1syVpnaCDdr3whoB
# 2lQMIVblLSqoSJjCkVmJX8aNqLuwJZZ8kPIMoXrPifCQR7jMJfp2JCaFHCXVCU3r
# JULv7XkMBOev/W0fGxNzAOJJ0irO53zTexSADYUDWYuJ48MANwbHe8e8nvo50M0L
# /YWPWHy1s5rpUtnBKPmuC1U5Uc/ooZQa16SAkM3IR6zqGTF+7JC+1NMipPrIvmPe
# +uB39PzRf+orNiSvy7GC3eAnHH7hKsErQOYcdCu6eQjeul86Qyzb7/lLLLyxCli1
# qivpjvPTt9/rkUVMVaTPRDZzZJonNebs7VBFckHJO6SehjP/WYG2Fw0582XeBgBx
# +k9qWvkhLcKWy5rxSZV4P1b+3TBpdzOoJDlWJwnVIjXGlgZtpnbcYLPOaHy4Z5j2
# lMoWsbEXromK9xQLlhLMYNpB0OZEo9PYVdUCAwEAAaOCAgIwggH+MB8GA1UdIwQY
# MBaAFGg34Ou2O/hfEYb7/mF7CIhl9E5CMB0GA1UdDgQWBBTjYxPRDvWBv/mKUWry
# YB/WRh7UtTA9BgNVHSAENjA0MDIGBWeBDAEDMCkwJwYIKwYBBQUHAgEWG2h0dHA6
# Ly93d3cuZGlnaWNlcnQuY29tL0NQUzAOBgNVHQ8BAf8EBAMCB4AwEwYDVR0lBAww
# CgYIKwYBBQUHAwMwgbUGA1UdHwSBrTCBqjBToFGgT4ZNaHR0cDovL2NybDMuZGln
# aWNlcnQuY29tL0RpZ2lDZXJ0VHJ1c3RlZEc0Q29kZVNpZ25pbmdSU0E0MDk2U0hB
# Mzg0MjAyMUNBMS5jcmwwU6BRoE+GTWh0dHA6Ly9jcmw0LmRpZ2ljZXJ0LmNvbS9E
# aWdpQ2VydFRydXN0ZWRHNENvZGVTaWduaW5nUlNBNDA5NlNIQTM4NDIwMjFDQTEu
# Y3JsMIGUBggrBgEFBQcBAQSBhzCBhDAkBggrBgEFBQcwAYYYaHR0cDovL29jc3Au
# ZGlnaWNlcnQuY29tMFwGCCsGAQUFBzAChlBodHRwOi8vY2FjZXJ0cy5kaWdpY2Vy
# dC5jb20vRGlnaUNlcnRUcnVzdGVkRzRDb2RlU2lnbmluZ1JTQTQwOTZTSEEzODQy
# MDIxQ0ExLmNydDAJBgNVHRMEAjAAMA0GCSqGSIb3DQEBCwUAA4ICAQCrPzJbznZM
# nFDrHJU/sZ3iXchw0zJZ5PXJSf8+w8/LH2aCp87tJCqRKUe2VWvsUD2cD3QP3aT+
# wdMDxlkyeKquSD3XN8InQcvbR57+bmJDuD2Rpa9NNdRUeafVkHuiQDOUpEzWdH22
# h7SjaCwokTe+0de5opUDFak1XqvT5sMgItkf6vZ0oObTUTlPYhfsfjJllF1x+AAR
# hQRHx/1dfl/xh6rmGVUhHPb7qcq4D7BxRmpTGG1yeopdMukxi9zQf/XdfT03RSKt
# VAGZEeXEoyFMz10L+Pwz5IG4RZDOjXCZ4WEH4INZ5k5ttNGrg2NBd2VGxPfiKz3N
# v0H2uC1NJLXFD6mTp/x6yo4pwyEolbMWUVnQS27laaGtBgrjNRjGv6sotLJzyXzJ
# dwYEGZ0eEfhaveIYVHB9K3nnu4qko4JjaKdNkw6JtpU/mS2urxqfUBYcfcmTC72M
# XUeM2W5xUN3W1GX/kh3GDy8lCEci/tr0gRfw/0R4MDD+BsqoRyDuiour/eFR6xpB
# FO23RdAZWGfY7jLKZ4WvKbqslQgamlXPE98mdfMSmgRWaKPNWujCDzTa5K8IJArV
# PmLL9Z3prTyUsdLL/9HSe/Q5/mFWvrO9wcCxK8JP34lIGRKAxdSZff3lxvITQ7WS
# 7dZSFgJwhzLW/YqlPzSiS3+CA9/EQzQi6DGCGiAwghocAgEBMH0waTELMAkGA1UE
# BhMCVVMxFzAVBgNVBAoTDkRpZ2lDZXJ0LCBJbmMuMUEwPwYDVQQDEzhEaWdpQ2Vy
# dCBUcnVzdGVkIEc0IENvZGUgU2lnbmluZyBSU0E0MDk2IFNIQTM4NCAyMDIxIENB
# MQIQBZDEzw2wFBmZU6aM/Omw5zANBglghkgBZQMEAgEFAKB8MBAGCisGAQQBgjcC
# AQwxAjAAMBkGCSqGSIb3DQEJAzEMBgorBgEEAYI3AgEEMBwGCisGAQQBgjcCAQsx
# DjAMBgorBgEEAYI3AgEVMC8GCSqGSIb3DQEJBDEiBCCdqk18NUEp0ip5QVCGZiPX
# kvbzWBnh2NR4dC43ImBqnDANBgkqhkiG9w0BAQEFAASCAYBotgEJ5acCdBtjfgU4
# Ky6Q7xtyjxw047QvTC2lF8JMkAF8K50IZibxIk3F3McKuJAjpPCy4IvdkfP98luu
# EAm22qG/Udv098f9s9m2vJ3bBvTan9k2tvIRNjxqvOejrO9Cikn4n0y4WOnzZUrP
# QJDilYGRshCObPkhEp7ODM596oQ89UvV6SswoFjVXYqXv6g1y5pRr0VXDqxLbo6X
# Ww5N4n7i/N2be/K2MRnU/LEoSX0k2u2Jxq13KC4tPDcf6w4Hh1uWpyBvdnGcb5aI
# 8rJab8Mf1a733pobzt2NUfknN/RuX8CGI1eTnBtIeYMYMfmTW2pI5HcvlKMpYWtr
# LrOdvcgCk2m4XNZIFYCGmXknHwO70qYVnXZL056otZa8KQGSw1O/i32GQHnnZCp2
# TqFe2XvqJ1kKH9QJ0+gZe9Cq8mkKc9wfxJeWngrd5rKMZrnWQ4lifeYirK3pPsCo
# Z8qUIhF+yCsQTDtfinBqlt2uKUG/WuAjjRe44YOzpSFgXW+hghd2MIIXcgYKKwYB
# BAGCNwMDATGCF2IwghdeBgkqhkiG9w0BBwKgghdPMIIXSwIBAzEPMA0GCWCGSAFl
# AwQCAQUAMHcGCyqGSIb3DQEJEAEEoGgEZjBkAgEBBglghkgBhv1sBwEwMTANBglg
# hkgBZQMEAgEFAAQgwXXc+P48/HWOPCiJmor+x3Nw4666TcpmDaSQohlNTGICEClA
# TqZgD6k7u1Eqqr8Te0MYDzIwMjYxMDE0MTgxMjU0WqCCEzowggbtMIIE1aADAgEC
# AhAIT9wzT35FTtvDD4/5khg1MA0GCSqGSIb3DQEBCwUAMGkxCzAJBgNVBAYTAlVT
# MRcwFQYDVQQKEw5EaWdpQ2VydCwgSW5jLjFBMD8GA1UEAxM4RGlnaUNlcnQgVHJ1
# c3RlZCBHNCBUaW1lU3RhbXBpbmcgUlNBNDA5NiBTSEEyNTYgMjAyNSBDQTEwHhcN
# MjYwODA1MDAwMDAwWhcNMzcxMTA0MjM1OTU5WjBjMQswCQYDVQQGEwJVUzEXMBUG
# A1UEChMORGlnaUNlcnQsIEluYy4xOzA5BgNVBAMTMkRpZ2lDZXJ0IFNIQTI1NiBS
# U0E0MDk2IFRpbWVzdGFtcCBSZXNwb25kZXIgMjAyNiAxMIICIjANBgkqhkiG9w0B
# AQEFAAOCAg8AMIICCgKCAgEAtnum8sn+zUr41JtMZbP9OMYw+HwJDpG5xkIu/lqc
# fNYmMX81YmsUiHLbh9ykpeWBGKTLhYBrAN9Tdg/QEzG32XcObmgIblnr0CoQ3WSA
# eDZ6nH6X6VkFyYkJw3QBJREwvm4UhLzSxmwPA7cFKRTEOMsmEEj6qJk/dqLEAL+o
# QYuOwE2UuiX1Vnul8YReIyWd4kgLn9gq6LNXM0UplkR6jL/QHxmb6fMoGBJYbnaU
# I7XD6cKDpekK2SVMld4iDbzeHDtOaaxldH5IxuNusQ69nd8/ZXEiB5Hbxj3RlK13
# cX1W4DlFXKdv/CEhM8Cj1vvlmvhNroyPdRGbbpBlgyf8Wdu5N6ByhFwURn0U6ozl
# PoxN22v+fviUhP+6DR547OZnpBMWDfei1f5sVGwiiW/KQTWOK97g+4RJpPzPNV4V
# YMAwO2jM2Aty2QYPVmOQTJm0msuXnJrSbl2gf9JylpkJlWXqk1Q4LJsxz+TELoQC
# ZIljbgvTJgoPU2R12ydv8i1UqL/adelA0y7U9Pmmtbze9Xx3rtajC5SzQd1jgfwA
# wsa90v9YcSPdmeoyoBBA/27cCL237l5DTYYPDLQ4ON3OLTGWnvRb6jDrf/T75gMR
# fUzSLCBQfBusm9+mSWRlC/Df6S/e9Q8i13CuhzOT2Jx+V/nlbXM4QoBwlUAhelww
# JT0CAwEAAaOCAZUwggGRMAwGA1UdEwEB/wQCMAAwHQYDVR0OBBYEFBTJY4owLtRK
# +26U8+bjQH717M3iMB8GA1UdIwQYMBaAFO9vU0rp5AZ8esrikFb2L9RJ7MtOMA4G
# A1UdDwEB/wQEAwIHgDAWBgNVHSUBAf8EDDAKBggrBgEFBQcDCDCBlQYIKwYBBQUH
# AQEEgYgwgYUwJAYIKwYBBQUHMAGGGGh0dHA6Ly9vY3NwLmRpZ2ljZXJ0LmNvbTBd
# BggrBgEFBQcwAoZRaHR0cDovL2NhY2VydHMuZGlnaWNlcnQuY29tL0RpZ2lDZXJ0
# VHJ1c3RlZEc0VGltZVN0YW1waW5nUlNBNDA5NlNIQTI1NjIwMjVDQTEuY3J0MF8G
# A1UdHwRYMFYwVKBSoFCGTmh0dHA6Ly9jcmwzLmRpZ2ljZXJ0LmNvbS9EaWdpQ2Vy
# dFRydXN0ZWRHNFRpbWVTdGFtcGluZ1JTQTQwOTZTSEEyNTYyMDI1Q0ExLmNybDAg
# BgNVHSAEGTAXMAgGBmeBDAEEAjALBglghkgBhv1sBwEwDQYJKoZIhvcNAQELBQAD
# ggIBAI3FOmEenVIK35msCYB+fShAsWvSYvLBItoNdAgQ2jIqrGsVsluXMJU/+mRe
# bBc52s6lbKAvOVPXaizmKkMLLflEEKDZQx4CkS2t8aHPjkXha3hYZ010htFa3dhN
# gmalH5vuWvh3tTCf4frTS7gPtGc4Z/xaPhQ2AB1mR8eEe/WbH0RWHvVIl6VwQ3+g
# 5FKNfN2N/DWJkf13w2H+2GfqEfbd35Ww8CvoYBjLNIDTadcPWdgsjsiOaK/7EsKJ
# gLjUNIVgvcaFOLLQ/GlrA+0ZHJoFUbOr5SJN8zykPspXIXlpDJY/gqFUZRROeab9
# GVgmhbdOJcD/63RhxPahFUGbckRONqMe6DYAv6/mOG0pWd3cPStsdcS7buj5Dyni
# wRY8yooMH6ptx5vpP/pZzBPBeZD2U4IsthyxB5Jaa8qrOkB5z160TXiM5ADMspZ0
# TfD9MJoq0tFpFPssKRFhWeEDYPvcUuN7U7lvcdHl4ezQ3NT/7Ffs1sR1yh/LRbdZ
# 3B3Vc6q2WmD8mDC0p9kzl2o73iVtS946IkEj7FkRsZGww1teYxERROC745xrtjvc
# w9ZyyUjHZWGRIpJeMNsPquCDf0fkyHtB+J4AiNZqCQk23rxh+KbpyMTNVKItJ5l9
# 2Svl20U9NbqMBOVYl1h54NEYLJq1/xHWFKPNK903zJZA9P2DMIIGtDCCBJygAwIB
# AgIQDcesVwX/IZkuQEMiDDpJhjANBgkqhkiG9w0BAQsFADBiMQswCQYDVQQGEwJV
# UzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3d3cuZGlnaWNlcnQu
# Y29tMSEwHwYDVQQDExhEaWdpQ2VydCBUcnVzdGVkIFJvb3QgRzQwHhcNMjUwNTA3
# MDAwMDAwWhcNMzgwMTE0MjM1OTU5WjBpMQswCQYDVQQGEwJVUzEXMBUGA1UEChMO
# RGlnaUNlcnQsIEluYy4xQTA/BgNVBAMTOERpZ2lDZXJ0IFRydXN0ZWQgRzQgVGlt
# ZVN0YW1waW5nIFJTQTQwOTYgU0hBMjU2IDIwMjUgQ0ExMIICIjANBgkqhkiG9w0B
# AQEFAAOCAg8AMIICCgKCAgEAtHgx0wqYQXK+PEbAHKx126NGaHS0URedTa2NDZS1
# mZaDLFTtQ2oRjzUXMmxCqvkbsDpz4aH+qbxeLho8I6jY3xL1IusLopuW2qftJYJa
# DNs1+JH7Z+QdSKWM06qchUP+AbdJgMQB3h2DZ0Mal5kYp77jYMVQXSZH++0trj6A
# o+xh/AS7sQRuQL37QXbDhAktVJMQbzIBHYJBYgzWIjk8eDrYhXDEpKk7RdoX0M98
# 0EpLtlrNyHw0Xm+nt5pnYJU3Gmq6bNMI1I7Gb5IBZK4ivbVCiZv7PNBYqHEpNVWC
# 2ZQ8BbfnFRQVESYOszFI2Wv82wnJRfN20VRS3hpLgIR4hjzL0hpoYGk81coWJ+Kd
# PvMvaB0WkE/2qHxJ0ucS638ZxqU14lDnki7CcoKCz6eum5A19WZQHkqUJfdkDjHk
# ccpL6uoG8pbF0LJAQQZxst7VvwDDjAmSFTUms+wV/FbWBqi7fTJnjq3hj0XbQcd8
# hjj/q8d6ylgxCZSKi17yVp2NL+cnT6Toy+rN+nM8M7LnLqCrO2JP3oW//1sfuZDK
# iDEb1AQ8es9Xr/u6bDTnYCTKIsDq1BtmXUqEG1NqzJKS4kOmxkYp2WyODi7vQTCB
# ZtVFJfVZ3j7OgWmnhFr4yUozZtqgPrHRVHhGNKlYzyjlroPxul+bgIspzOwbtmsg
# Y1MCAwEAAaOCAV0wggFZMBIGA1UdEwEB/wQIMAYBAf8CAQAwHQYDVR0OBBYEFO9v
# U0rp5AZ8esrikFb2L9RJ7MtOMB8GA1UdIwQYMBaAFOzX44LScV1kTN8uZz/nupiu
# HA9PMA4GA1UdDwEB/wQEAwIBhjATBgNVHSUEDDAKBggrBgEFBQcDCDB3BggrBgEF
# BQcBAQRrMGkwJAYIKwYBBQUHMAGGGGh0dHA6Ly9vY3NwLmRpZ2ljZXJ0LmNvbTBB
# BggrBgEFBQcwAoY1aHR0cDovL2NhY2VydHMuZGlnaWNlcnQuY29tL0RpZ2lDZXJ0
# VHJ1c3RlZFJvb3RHNC5jcnQwQwYDVR0fBDwwOjA4oDagNIYyaHR0cDovL2NybDMu
# ZGlnaWNlcnQuY29tL0RpZ2lDZXJ0VHJ1c3RlZFJvb3RHNC5jcmwwIAYDVR0gBBkw
# FzAIBgZngQwBBAIwCwYJYIZIAYb9bAcBMA0GCSqGSIb3DQEBCwUAA4ICAQAXzvsW
# gBz+Bz0RdnEwvb4LyLU0pn/N0IfFiBowf0/Dm1wGc/Do7oVMY2mhXZXjDNJQa8j0
# 0DNqhCT3t+s8G0iP5kvN2n7Jd2E4/iEIUBO41P5F448rSYJ59Ib61eoalhnd6ywF
# LerycvZTAz40y8S4F3/a+Z1jEMK/DMm/axFSgoR8n6c3nuZB9BfBwAQYK9FHaoq2
# e26MHvVY9gCDA/JYsq7pGdogP8HRtrYfctSLANEBfHU16r3J05qX3kId+ZOczgj5
# kjatVB+NdADVZKON/gnZruMvNYY2o1f4MXRJDMdTSlOLh0HCn2cQLwQCqjFbqrXu
# vTPSegOOzr4EWj7PtspIHBldNE2K9i697cvaiIo2p61Ed2p8xMJb82Yosn0z4y25
# xUbI7GIN/TpVfHIqQ6Ku/qjTY6hc3hsXMrS+U0yy+GWqAXam4ToWd2UQ1KYT70kZ
# jE4YtL8Pbzg0c1ugMZyZZd/BdHLiRu7hAWE6bTEm4XYRkA6Tl4KSFLFk43esaUeq
# GkH/wyW4N7OigizwJWeukcyIPbAvjSabnf7+Pu0VrFgoiovRDiyx3zEdmcif/sYQ
# sfch28bZeUz2rtY/9TCA6TD8dC3JE3rYkrhLULy7Dc90G6e8BlqmyIjlgp2+VqsS
# 9/wQD7yFylIz0scmbKvFoW2jNrbM1pD2T7m3XDCCBY0wggR1oAMCAQICEA6bGI75
# 0C3n79tQ4ghAGFowDQYJKoZIhvcNAQEMBQAwZTELMAkGA1UEBhMCVVMxFTATBgNV
# BAoTDERpZ2lDZXJ0IEluYzEZMBcGA1UECxMQd3d3LmRpZ2ljZXJ0LmNvbTEkMCIG
# A1UEAxMbRGlnaUNlcnQgQXNzdXJlZCBJRCBSb290IENBMB4XDTIyMDgwMTAwMDAw
# MFoXDTMxMTEwOTIzNTk1OVowYjELMAkGA1UEBhMCVVMxFTATBgNVBAoTDERpZ2lD
# ZXJ0IEluYzEZMBcGA1UECxMQd3d3LmRpZ2ljZXJ0LmNvbTEhMB8GA1UEAxMYRGln
# aUNlcnQgVHJ1c3RlZCBSb290IEc0MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIIC
# CgKCAgEAv+aQc2jeu+RdSjwwIjBpM+zCpyUuySE98orYWcLhKac9WKt2ms2uexuE
# DcQwH/MbpDgW61bGl20dq7J58soR0uRf1gU8Ug9SH8aeFaV+vp+pVxZZVXKvaJNw
# wrK6dZlqczKU0RBEEC7fgvMHhOZ0O21x4i0MG+4g1ckgHWMpLc7sXk7Ik/ghYZs0
# 6wXGXuxbGrzryc/NrDRAX7F6Zu53yEioZldXn1RYjgwrt0+nMNlW7sp7XeOtyU9e
# 5TXnMcvak17cjo+A2raRmECQecN4x7axxLVqGDgDEI3Y1DekLgV9iPWCPhCRcKtV
# gkEy19sEcypukQF8IUzUvK4bA3VdeGbZOjFEmjNAvwjXWkmkwuapoGfdpCe8oU85
# tRFYF/ckXEaPZPfBaYh2mHY9WV1CdoeJl2l6SPDgohIbZpp0yt5LHucOY67m1O+S
# kjqePdwA5EUlibaaRBkrfsCUtNJhbesz2cXfSwQAzH0clcOP9yGyshG3u3/y1Yxw
# LEFgqrFjGESVGnZifvaAsPvoZKYz0YkH4b235kOkGLimdwHhD5QMIR2yVCkliWzl
# DlJRR3S+Jqy2QXXeeqxfjT/JvNNBERJb5RBQ6zHFynIWIgnffEx1P2PsIV/EIFFr
# b7GrhotPwtZFX50g/KEexcCPorF+CiaZ9eRpL5gdLfXZqbId5RsCAwEAAaOCATow
# ggE2MA8GA1UdEwEB/wQFMAMBAf8wHQYDVR0OBBYEFOzX44LScV1kTN8uZz/nupiu
# HA9PMB8GA1UdIwQYMBaAFEXroq/0ksuCMS1Ri6enIZ3zbcgPMA4GA1UdDwEB/wQE
# AwIBhjB5BggrBgEFBQcBAQRtMGswJAYIKwYBBQUHMAGGGGh0dHA6Ly9vY3NwLmRp
# Z2ljZXJ0LmNvbTBDBggrBgEFBQcwAoY3aHR0cDovL2NhY2VydHMuZGlnaWNlcnQu
# Y29tL0RpZ2lDZXJ0QXNzdXJlZElEUm9vdENBLmNydDBFBgNVHR8EPjA8MDqgOKA2
# hjRodHRwOi8vY3JsMy5kaWdpY2VydC5jb20vRGlnaUNlcnRBc3N1cmVkSURSb290
# Q0EuY3JsMBEGA1UdIAQKMAgwBgYEVR0gADANBgkqhkiG9w0BAQwFAAOCAQEAcKC/
# Q1xV5zhfoKN0Gz22Ftf3v1cHvZqsoYcs7IVeqRq7IviHGmlUIu2kiHdtvRoU9BNK
# ei8ttzjv9P+Aufih9/Jy3iS8UgPITtAq3votVs/59PesMHqai7Je1M/RQ0SbQyHr
# lnKhSLSZy51PpwYDE3cnRNTnf+hZqPC/Lwum6fI0POz3A8eHqNJMQBk1RmppVLC4
# oVaO7KTVPeix3P0c2PR3WlxUjG/voVA9/HYJaISfb8rbII01YBwCA8sgsKxYoA5A
# Y8WYIsGyWfVVa88nq2x2zm8jLfR+cWojayL/ErhULSd+2DrZ8LaHlv1b0VysGMNN
# n3O3AamfV6peKOK5lDGCA3wwggN4AgEBMH0waTELMAkGA1UEBhMCVVMxFzAVBgNV
# BAoTDkRpZ2lDZXJ0LCBJbmMuMUEwPwYDVQQDEzhEaWdpQ2VydCBUcnVzdGVkIEc0
# IFRpbWVTdGFtcGluZyBSU0E0MDk2IFNIQTI1NiAyMDI1IENBMQIQCE/cM09+RU7b
# ww+P+ZIYNTANBglghkgBZQMEAgEFAKCB0TAaBgkqhkiG9w0BCQMxDQYLKoZIhvcN
# AQkQAQQwHAYJKoZIhvcNAQkFMQ8XDTI2MTAxNDE4MTI1NFowKwYLKoZIhvcNAQkQ
# AgwxHDAaMBgwFgQUUdmr2gNJc9hPQmaspIJI5rNpxDkwLwYJKoZIhvcNAQkEMSIE
# IMHAzoW/PmFo6Jf9AuQ2GX/3TlD9oVH5dBevmQo/f2rYMDcGCyqGSIb3DQEJEAIv
# MSgwJjAkMCIEIC2gnaf0Ex+f5y22xebpyWVnVa8EPx6nQswNISDhQev8MA0GCSqG
# SIb3DQEBAQUABIICAKsGfUo4SSQM1LocG/FcoXZm4XcZad65BgxwP83cvr8yM/iP
# oJ2soT3wKGEBCbjnimZaBhFKAlXHBHtxJAbkJBGETZjY47OahS8vBHSi4+EZdyJ5
# RUeaOCc0FKeZhe6ueWpvRJRwPGtFuFQ8+mGqTbFCjU+QiBYSur6rnYrR46TEYrCW
# mr1RHwzsSy/9vr1pwg/yYrx2vqOaNiGF4gpHuRquLhXTG+z3lWHHfJH3+yKLR8QT
# TvgNNtdVKXzpzjKHMk06tOt6BXRKbeZoLlvuJpYd2U0GNUz6b+kbiNqpe9ZAKeW7
# EUZkzmQ1CuhMRDreo+hLQKE65lMXRLkqNmR7256CjquV93WY8W10bFSrSgkflIgU
# qnsEYdQj1zx+sCgXVJzFaxxb4fBuqebmY5d8lz8H+E6I+pZWjTki+DrWh6pnKaYc
# 1upiQpLDYr8tn+8xhI0WFjgCH0TzHKs30CDvjO+Clua0f21K1jvJCUXMKWhdowP1
# w4FJba9tHl5DhEr55QLzCWzv24u6Xd8M3uj4QP3pAXwApNmpOi4o8zlNMdbYqXgY
# g0HaNXpeiEpF4vU39FUfBcXfvhmKDioiNS79yaXChUGLX2oXuoAUQQc+eUa0jXY/
# FbshO5TLKMRidFmGrmsNPUimMoIVrUrc4FkX3Km/Zm72w4BYKCcw+GPebC9W
# SIG # End signature block
