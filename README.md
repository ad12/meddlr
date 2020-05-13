# dl-ss-recon
Deep-learning based semi-supervised MRI reconstruction


### Setup

#### Setting Up Symlinks
To simplify path logic, all paths will be relative to the base repository directory.

All data will be stored in a local directory `./datasets/data`.
If you would like to store data at a different path, symlink the local directory `./datasets/data` 
to the desired output directory:
```bash
# Run the following command from the base repository directory.

# Linux
ln -s /PATH/TO/DESIRED/DATASET/DIR ./datasets/data
```

Similarly, all results will be stored in the `./results` folder. Use symlinking to output
to different folder.

### Contributing
Please run `./dev/linter.sh` from the base repository directory before committing any code.

You may need to install the following libraries:
```bash
pip install black==19.3b0 isort flake8 flake8-comprehensions
```