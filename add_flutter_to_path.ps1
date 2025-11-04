# PowerShell script to add Flutter to PATH permanently
# Right-click and "Run as Administrator"

$flutterPath = "C:\Users\HP\flutter\bin"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($currentPath -notlike "*$flutterPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$flutterPath", "User")
    Write-Host "✅ Flutter added to PATH successfully!" -ForegroundColor Green
    Write-Host "⚠️  Please restart your terminal/PowerShell for changes to take effect." -ForegroundColor Yellow
} else {
    Write-Host "✅ Flutter is already in PATH!" -ForegroundColor Green
}

Write-Host "`nTo verify, run: flutter --version" -ForegroundColor Cyan
Read-Host "Press Enter to exit"

