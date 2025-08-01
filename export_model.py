from pathlib import Path

required_files = [
    "config.json",
    "generation_config.json",
    "model-00001-of-00004.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "preprocessor_config.json",
]

missing = []
saved_model_dir = Path("saved_model")

for file in required_files:
    if not (saved_model_dir / file).exists():
        missing.append(file)

if not missing:
    print("[✓] All required files are present in 'saved_model/'")
else:
    print("[!] Missing files:")
    for f in missing:
        print(f"  - {f}")
