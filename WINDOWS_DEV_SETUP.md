# Windows Development Setup for LiteRT

## Prerequisites

### 1. Install Visual Studio 2019 or 2022
- Download Visual Studio Community from https://visualstudio.microsoft.com/
- During installation, select:
  - "Desktop development with C++"
  - Windows 10/11 SDK
  - MSVC v142 or v143 (C++ compiler)
  - C++ CMake tools for Windows (optional but useful)

### 2. Install Python 3.9-3.11
```powershell
# Download from python.org or use Windows Store
python --version  # Verify installation
```

### 3. Install Bazel
```powershell
# Download Bazelisk (recommended)
# From https://github.com/bazelbuild/bazelisk/releases
# Rename to bazel.exe and add to PATH

# Or use Chocolatey:
choco install bazel

# Verify
bazel --version
```

### 4. Install Git
```powershell
# Download from https://git-scm.com/download/win
# Or use Chocolatey:
choco install git
```

## Building LiteRT on Windows

### 1. Clone the Repository
```powershell
git clone https://github.com/google-ai-edge/LiteRT.git
cd LiteRT
```

### 2. Configure Environment
```powershell
# Set up Visual Studio environment
# For VS 2022:
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

# For VS 2019:
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
```

### 3. Build the Shared Library
```powershell
# Build the specific target that was failing
bazel build //litert/runtime:litert_runtime_c_api_shared_lib

# Or build with specific Windows flags
bazel build --config=windows //litert/runtime:litert_runtime_c_api_shared_lib

# For debug build
bazel build --config=windows --compilation_mode=dbg //litert/runtime:litert_runtime_c_api_shared_lib
```

### 4. Expected Output
After successful build, you should find:
- `bazel-bin/litert/runtime/litert_runtime_c_api_shared_lib.dll`
- `bazel-bin/litert/runtime/litert_runtime_c_api_shared_lib.lib` (import library)

## Troubleshooting

### Common Issues

1. **Missing includes**
   ```
   error C1083: Cannot open include file: 'dlfcn.h'
   ```
   This should now be fixed with our changes.

2. **Bazel cache issues**
   ```powershell
   bazel clean --expunge
   bazel shutdown
   ```

3. **Python version conflicts**
   Ensure Python 3.9-3.11 is in PATH and is the default version.

4. **Long path issues on Windows**
   Enable long paths in Windows:
   ```powershell
   # Run as Administrator
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```

## Verifying the Fix

After building successfully, test dynamic loading:

```powershell
# Create a simple test
bazel test //litert/core:dynamic_loading_test
bazel test //litert/cc:litert_shared_library_test
```

## Alternative: Using WSL2

If native Windows build has issues, WSL2 provides a Linux environment:

```powershell
# Install WSL2
wsl --install

# In WSL2 Ubuntu:
sudo apt update
sudo apt install build-essential python3 python3-pip
# Install Bazel...
# Build normally as on Linux
```

However, this would test the Linux code path, not the Windows-specific code.