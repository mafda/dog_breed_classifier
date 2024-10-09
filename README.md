# Dog Breed Classifier

In this project, given an image of a dog, the algorithm will identify an
estimate of the canine’s breed.  If supplied an image of a human, the code will
identify the resembling dog breed.


## Project Setup

### Clone this repository

```shell
(base)$: git@github.com:mafda/dog_breed_classifier.git
(base)$: cd dog_breed_classifier
```

### Configure environment

- Create the conda environment

    ```shell
    (base)$: conda env create -f environment.yml
    ```

- Activate the environment

    ```shell
    (base)$: conda activate dog_breed
    ```

- Download the [dog
  dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).
  Unzip the folder and place it in the repo, at location
  `path/to/dog_breed_classifier/data/dog_images`.

- Download the [human
  dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).
  Unzip the folder and place it in the repo, at location
  `path/to/dog_breed_classifier/data/lfw`.

## Project Structure

```shell
├── README.md
├── data
│   ├── dog_images
│   ├── haarcascades
│   └── lfw
├── environment.yml
└── src
    └── dog_app.ipynb
```

## References

- [Data Scientist Nanodegree
  Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025)

---

made with 💙 by [mafda](https://mafda.github.io/)
