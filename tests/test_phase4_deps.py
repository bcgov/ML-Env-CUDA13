try:
    import gguf
    import sentencepiece
    import dotenv
    print(f"gguf version: {getattr(gguf, '__version__', 'unknown')}")
    # sentencepiece doesn't always have __version__ in some builds, check if it exists
    print(f"sentencepiece OK")
    print(f"python-dotenv OK")
    print("\nPhase 4 & 5 dependencies OK")
except Exception as e:
    print(f"Phase 4 & 5 dependencies FAILED: {e}")
    exit(1)
