import sys

# Map user library names to their Python import names and display helpers
libraries = {
    "numpy": ("numpy", lambda m: f"Version: {m.__version__}"),
    "pandas": ("pandas", lambda m: f"Version: {m.__version__}"),
    "nltk": ("nltk", lambda m: f"Version: {m.__version__}"),
    "scikit-learn": ("sklearn", lambda m: f"Version: {m.__version__}"),
    "streamlit": ("streamlit", lambda m: f"Version: {m.__version__}"),
}

def main():
    print(f"Python Version: {sys.version}\n")
    print("Checking library installations:")
    print("-" * 60)

    all_ok = True
    for name, (import_name, get_info) in libraries.items():
        try:
            # Import the module dynamically
            module = __import__(import_name)
            # Try to get the version or metadata
            info = get_info(module)
            
            # Perform a very basic sanity check on functionality
            if import_name == "numpy":
                a = module.array([1, 2, 3])
                assert a.sum() == 6
            elif import_name == "pandas":
                df = module.DataFrame({"a": [1, 2]})
                assert len(df) == 2
            elif import_name == "sklearn":
                # Check for a simple estimator creation
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
            elif import_name == "nltk":
                # Check basic word tokenize imports or similar
                from nltk.tokenize import word_tokenize
            elif import_name == "streamlit":
                # Simple check for streamlit layout or text function presence
                assert hasattr(module, "title")

            print(f"[ OK ]   {name:<15} -> Imported successfully. {info}")
        except ImportError as e:
            print(f"[FAIL]   {name:<15} -> Failed to import! Error: {e}")
            all_ok = False
        except Exception as e:
            print(f"[WARN]   {name:<15} -> Imported but basic check failed. Error: {e}")
            all_ok = False

    print("-" * 60)
    if all_ok:
        print("All specified libraries are installed and verified successfully!")
    else:
        print("Some libraries are missing or did not pass the basic check. Please install or re-install them.")

if __name__ == "__main__":
    main()
