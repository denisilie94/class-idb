# Classification with dictionary learning and a distance barrier promoting incoherence (class-idb)

We present a new approach to the incoherent dictionary learning problem using a barrier function that promotes incoherence.
This function has a context-dependent quadratic term and a distance barrier term which can be used in both local and global structures.
This strategy achieves better results in terms of error representation and incoherence of the dictionary, compared with the standard problem.
We demonstrate on several datasets that this function can improve the performance of dictionaries in classification problems.

## Citing Us

This represents the main resources necessary for reproducing the results presented in “Classification with dictionary learning and a distance barrier promoting incoherence”, by Denis C. Ilie Ablachim, and Bogdan Dumitrescu. If you use our work please cite the following paper

```
@inproceedings{ilie2023sparse,
  title={Classification with dictionary learning and a distance barrier promoting incoherence},
  author={Ilie-Ablachim, Denis C and Dumitrescu, Bogdan},
  booktitle={33rd IEEE International Workshop on Machine Learning for Signal Processing (MLSP 2023)},
  year={2023},
  organization={IEEE}
}
```

## Datasets

Most of the datasets are available [here](http://www.zhuolin.umiacs.io/projectlcksvd.html).

The action bank features can be downloaded from [here](https://cse.buffalo.edu/~jcorso/r/actionbank/).

We provide some extractors to organize the *HMDB51* and *UCF50* datasets as mat files efficiently.

## Reproducing the results

The proposed methods are available under the following scripts: *aksvd_idb_test.m* and *aksvd_itdb_test.m*.

## Funding

This work is supported by a grant of the Ministry of Research, Innovation and Digitization, CNCS – UEFISCDI, project number PN-III-P4-PCE-2021-0154, within PNCDI III.
