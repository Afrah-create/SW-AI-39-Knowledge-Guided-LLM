@echo off
REM Flutter Setup and Build Script for Android App
REM Run this script from the android-app directory

echo ========================================
echo Flutter Android App Setup
echo ========================================
echo.

REM Add Flutter to PATH for this session
set PATH=%PATH%;C:\Users\HP\flutter\bin

REM Check Flutter installation
echo Checking Flutter installation...
flutter --version
echo.

REM Navigate to android-app directory
cd /d "%~dp0android-app" 2>nul || cd /d "D:\deployment\android-app"
echo Current directory: %CD%
echo.

REM Get Flutter dependencies
echo Installing Flutter dependencies...
flutter pub get
echo.

REM Check Flutter doctor
echo Running Flutter doctor...
flutter doctor
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Ensure Android Studio is installed
echo 2. Connect an Android device or start an emulator
echo 3. Run: flutter run
echo 4. Or build APK: flutter build apk --release
echo.
pause

