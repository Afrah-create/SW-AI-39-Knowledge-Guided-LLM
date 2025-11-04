# Android App - Agricultural Recommendation System

This directory contains the Android mobile application for offline crop recommendations.

## Project Structure

```
android-app/
├── lib/
│   ├── main.dart
│   ├── models/
│   │   ├── crop_model.dart
│   │   ├── soil_properties.dart
│   │   └── climate_conditions.dart
│   ├── services/
│   │   ├── recommendation_service.dart
│   │   ├── constraint_engine.dart
│   │   ├── model_loader.dart
│   │   └── local_storage.dart
│   ├── screens/
│   │   ├── home_screen.dart
│   │   ├── input_screen.dart
│   │   └── results_screen.dart
│   └── widgets/
│       ├── crop_card.dart
│       └── input_field.dart
├── assets/
│   ├── models/
│   │   └── (TFLite models will be added here)
│   └── data/
│       └── crop_constraints.json
├── pubspec.yaml
└── README.md
```

## Getting Started

### Prerequisites

- Flutter SDK 3.0+
- Android Studio / VS Code
- Android SDK (API 24+)

### Setup

```bash
# Install Flutter
flutter doctor

# Create Flutter project (if not exists)
flutter create agricultural_app

# Install dependencies
cd agricultural_app
flutter pub get
```

### Run App

```bash
# Debug mode
flutter run

# Release build
flutter build apk --release
```

## Features

- ✅ Offline crop recommendations
- ✅ Soil and climate input
- ✅ ML-powered recommendations
- ✅ Local data storage
- ✅ History of recommendations

