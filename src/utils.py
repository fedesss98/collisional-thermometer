import toml
from pathlib import Path


def read_configuration(args=None):
    """Read the configuration file."""
    with open('config.toml', 'r') as f:
        config = toml.load(f)

    # Overwrite the configuration with the command line arguments
    if args is not None and args.chains:
        print(args.chains)
        config['ancilla']['chains'] = args.chains
    if args is not None and args.depth:
        print(args.depth)
        config['ancilla']['n'] = args.depth

    # Create the folder to store the results
    results_folder = Path(config['folder'])
    results_folder.mkdir(parents=True, exist_ok=True)
    # Create the subfolder for the images
    images_folder = results_folder / 'img'
    images_folder.mkdir(parents=True, exist_ok=True)
    # Save the configuration file
    with open(results_folder / 'config.toml', 'w') as f:
        toml.dump(config, f)

    return config, results_folder