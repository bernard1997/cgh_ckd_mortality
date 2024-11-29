import numpy as np 
import pandas as pd 
# Logistic Regression coefficients
continuous_log_reg_coefficients = {
    "Intercept": -5.992801108,
    "ADL.dependent": 0.804769756,
    "AF": 0.107704878,
    "age": 0.065961244,
    "albumin": -0.139847448,
    "calcium.total.serum": 1.487481591,
    "CCI": 0.080843731,
    "CHF.merge": 0.306259817,
    "CRRT.given": 0.123355331,
    "CVA": 0.900118855,
    "dementia": 0.445034263,
    "eGFR": 0.032825585,
    "haemoglobin": -0.166093241,
    "liver.disease": 0.700835815,
    "MI.NSTEMI": 0.382964116,
    "phosphate.inorganic.serum": 0.650928039,
    "PVD": 0.763039388,
    "raceIndian": -0.777507358,
    "raceMalay": 0.194981343,
    "raceOthers": -0.136938792,
}


categorical_log_reg_coefficients = {
    "Intercept": -2.177676078,
    "ADL.dependentTRUE": 0.787787679,
    "age70_74": -0.067285564,
    "age75_79": 0.680830001,
    "age80_84": 1.015714979,
    "age_abv_85": 1.19695128,
    "albumin25_29": -1.044455171,
    "albumin30_34": -1.431304702,
    "albumin_35_abv": -2.017900634,
    "CCI3_4": 0.151252194,
    "CCI_abv_5": 0.612888178,
    "CRRT.givenTRUE": 0.340059758,
    "CVATRUE": 0.942599405,
    "dementiaTRUE": 0.520906507,
    "eGFR10_14": 0.464167663,
    "eGFR_15_abv": 0.864709402,
    "liver.diseaseTRUE": 0.664697819,
    "MI.NSTEMITRUE": 0.423873051,
    "phosphate.inorganic.serumTRUE": 0.776649496,
    "PVDTRUE": 0.750710325,
    "sexMALE": 0.346948671
}






