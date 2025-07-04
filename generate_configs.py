"""Generate pyproject.toml for different build variants of pylate-rs."""

import argparse
import sys # Import sys module

CONFIGS = {
    "default": {
        "package_name_suffix": "",
        "features": ["tokenizers/onig", "hf-hub", "python"],
    },
    "metal": {
        "package_name_suffix": "-metal",
        "features": ["tokenizers/onig", "hf-hub", "python", "metal", "accelerate"],
    },
    "accelerate": {
        "package_name_suffix": "-accelerate",
        "features": ["tokenizers/onig", "hf-hub", "python", "accelerate"],
    },
    "cuda": {
        "package_name_suffix": "-cuda",
        "features": ["tokenizers/onig", "hf-hub", "python", "cuda"],
    },
    "mkl": {
        "package_name_suffix": "-mkl",
        "features": ["tokenizers/onig", "hf-hub", "python", "mkl"],
    },
}


def generate_config(variant_name: str, base_name="pylate-rs"):
    """Generate a pyproject.toml file for a specific build variant."""
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    config = CONFIGS.get(variant_name)
    if not config:
        error = f"Unknown variant '{variant_name}'. Available: {list(CONFIGS.keys())}"
        raise ValueError(error)

    full_package_name = f"{base_name}{config['package_name_suffix']}"
    template_path = "pyproject.toml.template"

    with open(template_path) as f:
        template_content = f.read()

    features_list = config["features"]
    formatted_features = (
        "[" + ", ".join(f'"{feature}"' for feature in features_list) + "]"
    )

    new_content = template_content.replace(
        "{{ package_name }}", f'"{full_package_name}"'
    )
    new_content = new_content.replace("{{ features }}", formatted_features)

    with open("pyproject.toml", "w") as f:
        f.write(new_content)

    print(f"✅ Successfully generated pyproject.toml for '{full_package_name}'")
    print(f"   Features: {features_list}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pyproject.toml for a specific build variant."
    )
    parser.add_argument(
        "variant",
        choices=CONFIGS.keys(),
        help="The build variant to generate.",
    )
    args = parser.parse_args()
    generate_config(variant_name=args.variant)
