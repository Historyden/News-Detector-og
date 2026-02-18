import os
import sys

# -----------------------------
# Required files
# -----------------------------
required_files = ["vectorizer.pkl", "fake_news_model.pkl"]
missing_files = [f for f in required_files if not os.path.isfile(f)]

# -----------------------------
# Required packages
# -----------------------------
required_packages = ["streamlit", "scikit-learn", "numpy", "pandas"]
missing_packages = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing_packages.append(pkg)

# -----------------------------
# Results
# -----------------------------
if missing_files:
    print("❌ Missing pickle files:")
    for f in missing_files:
        print(f"  - {f}")
else:
    print("✅ All pickle files are present.")

if missing_packages:
    print("❌ Missing Python packages:")
    for p in missing_packages:
        print(f"  - {p}")
    print("\nInstall missing packages with:")
    print("  pip install " + " ".join(missing_packages))
else:
    print("✅ All required Python packages are installed.")

if not missing_files and not missing_packages:
    print("\n🎉 Environment is ready! You can run:")
    print("  streamlit run opd_god_mode.py")
