try:
    import llama_cpp
    print('llama_cpp import OK')
    from llama_cpp import Llama
    print('Llama class available')
except Exception as e:
    print('llama-cpp-python import failed:', e)
    raise
