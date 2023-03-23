
# filter_design

This repository contains the codes and dataset used to obtain the result of the paper _Design of non-Gaussian Multispectral SWIR Filters for Assessment of ECOSTRESS Library_.

## Requirements
```
numpy
pandas
scipy
sklearn
matplotlib
```

## Dataset
 
The dataset was extract from the ECOSTRESS Library. The file "reflet_amostras.csv" contains the extracted data, which is the reflectance of 1942 samples of materials. The values of reflectance are in the range from 0 to 1. These values were interpolated so that the wavelenght ranges from 900nm to 1700nm with 1nm resolution. The csv file has no header and uses the comma ',' as delimiter. Each column correspond to an SSR, and each row of this file correspond to a wavelength, starting from 900nm.
The Illuminant was obtained from [ASTM](astm.org), and the file "ASTMG173.csv" remains as it was provided.

## Filter Design script
The file "codigo.py" caontains the program. You can run the code by typing
```shell
$ python codigo.py
```

## Contact

E-mail: germano.fonseca@coppe.ufrj.br or germanosfonseca@yahoo.com.br

## References

- [*The ECOSTRESS spectral library version 1.0*](doi.org/10.1016/j.rse.2019.05.015)

## Citing

Please kindly cite our paper and the code when you use it. Thanks!

```
@article{Fonseca2023,
author = {Germano S. Fonseca, Leonardo B. de Sá and José Gabriel},
journal = {J. Opt. Soc. Am. A},
keywords = {},
number = {4},
pages = {},
publisher = {Optica Publishing Group},
title = {Design of non-Gaussian Multispectral SWIR Filters for Assessment of ECOSTRESS Library},
volume = {40},
month = {Apr},
year = {2023},
url = {},
doi = {}
}
```

```
@misc{fonseca_code_2023,
title={Python Code to Design non-Gaussian SWIR Multispectral Filter},
url={https://figshare.com/articles/software/_/22298743/0},
DOI={10.6084/m9.figshare.22298743},
abstractNote={<p>Python code to design non-Gaussian SWIR multispectral filter.</p>
<p>This code was used to obtain the results of the paper "Design of non-Gaussian Multispectral SWIR Filters for Assessment of ECOSTRESS Library".</p>
<p>More info in the README.md.</p>},
publisher={figshare},
author={Fonseca, Germano},
year={2023},
month={Mar}
}
```
