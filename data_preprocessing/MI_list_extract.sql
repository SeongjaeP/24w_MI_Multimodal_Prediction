SELECT di.subject_id, di.hadm_id, di.icd_code, dd.long_title
FROM mimiciv_hosp.diagnoses_icd di
JOIN mimiciv_hosp.d_icd_diagnoses dd ON di.icd_code = dd.icd_code
WHERE dd.long_title ILIKE '%Myocardial infarction%'