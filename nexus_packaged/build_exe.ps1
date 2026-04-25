Param(
    [switch]$SkipCompile,
    [switch]$FastBuild,
    [switch]$VerboseBuild
)

$ErrorActionPreference = "Stop"

function Replace-ImportText {
    param(
        [string]$FilePath,
        [string]$From,
        [string]$To
    )
    $content = Get-Content -Path $FilePath -Raw -Encoding UTF8
    $content = $content -replace [Regex]::Escape($From), $To
    Set-Content -Path $FilePath -Value $content -Encoding UTF8
}

function Ensure-Protection-Normalized {
    param(
        [string]$EncryptorPath,
        [string]$IntegrityPath,
        [string]$EncryptorObfPath,
        [string]$IntegrityObfPath,
        [string]$ProtectionInit
    )
    if ((Test-Path $EncryptorObfPath) -and -not (Test-Path $EncryptorPath)) {
        Move-Item -LiteralPath $EncryptorObfPath -Destination $EncryptorPath -Force
    }
    if ((Test-Path $IntegrityObfPath) -and -not (Test-Path $IntegrityPath)) {
        Move-Item -LiteralPath $IntegrityObfPath -Destination $IntegrityPath -Force
    }
    if (Test-Path $ProtectionInit) {
        Replace-ImportText -FilePath $ProtectionInit -From "from nexus_packaged.protection._e7f3" -To "from nexus_packaged.protection.encryptor"
        Replace-ImportText -FilePath $ProtectionInit -From "from nexus_packaged.protection._b2a1" -To "from nexus_packaged.protection.integrity"
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$encryptorPath = "nexus_packaged/protection/encryptor.py"
$integrityPath = "nexus_packaged/protection/integrity.py"
$encryptorObfPath = "nexus_packaged/protection/_e7f3.py"
$integrityObfPath = "nexus_packaged/protection/_b2a1.py"
$protectionInit = "nexus_packaged/protection/__init__.py"

if (!(Test-Path "nexus_packaged/dist")) {
    New-Item -ItemType Directory -Path "nexus_packaged/dist" | Out-Null
}
if (!(Test-Path "nexus_packaged/logs")) {
    New-Item -ItemType Directory -Path "nexus_packaged/logs" | Out-Null
}

# Preflight checks for active Python environment.
& python -c "import nuitka" 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'Nuitka is not installed in the active Python environment. Run: python -m pip install "nuitka[onefile]" zstandard'
}

# Recover safely from interrupted previous build runs.
Ensure-Protection-Normalized `
    -EncryptorPath $encryptorPath `
    -IntegrityPath $integrityPath `
    -EncryptorObfPath $encryptorObfPath `
    -IntegrityObfPath $integrityObfPath `
    -ProtectionInit $protectionInit

try {
    Write-Host "[1/5] Renaming sensitive modules for obfuscation..."
    if (-not (Test-Path $encryptorPath)) {
        throw "Missing source file for obfuscation: $encryptorPath"
    }
    if (-not (Test-Path $integrityPath)) {
        throw "Missing source file for obfuscation: $integrityPath"
    }
    Move-Item -LiteralPath $encryptorPath -Destination $encryptorObfPath -Force
    Move-Item -LiteralPath $integrityPath -Destination $integrityObfPath -Force
    Replace-ImportText -FilePath $protectionInit -From "from nexus_packaged.protection.encryptor" -To "from nexus_packaged.protection._e7f3"
    Replace-ImportText -FilePath $protectionInit -From "from nexus_packaged.protection.integrity" -To "from nexus_packaged.protection._b2a1"

    if ($SkipCompile) {
        Write-Host "[2/5] SkipCompile set, skipping Nuitka build."
    } else {
        Write-Host "[2/5] Running Nuitka compilation..."
        $env:NUITKA_CACHE_DIR = Join-Path $projectRoot "nexus_packaged/dist/.nuitka_cache"
        $ltoFlag = if ($FastBuild) { "--lto=no" } else { "--lto=yes" }
        $nuitkaArgs = @(
            "-m", "nuitka", "main.py",
            "--standalone",
            "--onefile",
            $ltoFlag,
            "--follow-imports",
            "--python-flag=no_annotations",
            "--python-flag=no_docstrings",
            "--include-data-dir=nexus_packaged/data=nexus_packaged/data",
            "--include-data-dir=nexus_packaged/config=nexus_packaged/config",
            "--include-data-dir=nexus_packaged/protection=nexus_packaged/protection",
            "--output-dir=dist",
            "--output-filename=nexus_trader.exe",
            "--assume-yes-for-downloads"
        )
        if ($FastBuild) {
            # Practical speed profile for constrained CI/dev environments.
            # These optional trees are not used by nexus_packaged runtime paths.
            $nuitkaArgs += @(
                "--nofollow-import-to=sympy",
                "--nofollow-import-to=scipy",
                "--nofollow-import-to=matplotlib",
                "--nofollow-import-to=IPython",
                "--nofollow-import-to=jupyter",
                "--nofollow-import-to=notebook"
            )
        }
        if ($VerboseBuild) {
            $nuitkaArgs += @(
                "--show-scons",
                "--show-progress",
                "--show-memory",
                "--verbose"
            )
        }
        python @nuitkaArgs
    }

    if (Test-Path "dist/nexus_trader.exe") {
        Write-Host "[3/5] Computing executable hash..."
        $hash = Get-FileHash -Path "dist/nexus_trader.exe" -Algorithm SHA256
        "$($hash.Hash.ToLower())  dist/nexus_trader.exe" | Set-Content -Path "dist/nexus_trader.exe.sha256" -Encoding ASCII
        Write-Host "Hash stored at dist/nexus_trader.exe.sha256"
    } else {
        Write-Host "[3/5] Skipping hash: dist/nexus_trader.exe not found."
    }
}
finally {
    Write-Host "[4/5] Restoring original module names..."
    if (Test-Path $encryptorObfPath) {
        Move-Item -LiteralPath $encryptorObfPath -Destination $encryptorPath -Force
    }
    if (Test-Path $integrityObfPath) {
        Move-Item -LiteralPath $integrityObfPath -Destination $integrityPath -Force
    }
    if (Test-Path $protectionInit) {
        Replace-ImportText -FilePath $protectionInit -From "from nexus_packaged.protection._e7f3" -To "from nexus_packaged.protection.encryptor"
        Replace-ImportText -FilePath $protectionInit -From "from nexus_packaged.protection._b2a1" -To "from nexus_packaged.protection.integrity"
    }
}

Write-Host "[5/5] Build routine complete."
