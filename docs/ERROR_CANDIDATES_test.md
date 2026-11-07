# Error-analysis candidates (test; 51 label errors of 206; tags auto-computed)

Choose >= 8 DISTINCT errors and write the cause hypothesis yourself. Gold ids marked G, predicted ids marked P.


## annotation 161 | sigir-201410 | NCT02097186 | inclusion | picked for `false_nei`
- Criterion: Patients undergoing elective carotid endarterectomy
- Gold: **not included** evidence [] | Predicted: **not enough information** evidence [0, 1, 2] | valid=True
- Tags: false_nei, rare_class_missed, evidence_spurious
- Raw output: `{"label": "not enough information", "evidence_ids": [0, 1, 2]}`
- Note:
  - [0] (-P) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
  - [1] (-P) She is now complaining of a cool right foot.
  - [2] (-P) Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
  - [3] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 162 | sigir-201410 | NCT02097186 | inclusion | picked for `rare_class_missed`
- Criterion: Patients undergoing open abdominal aortic aneurysm repair
- Gold: **not included** evidence [] | Predicted: **not enough information** evidence [0, 1, 2] | valid=True
- Tags: false_nei, rare_class_missed, evidence_spurious
- Raw output: `{"label": "not enough information", "evidence_ids": [0, 1, 2]}`
- Note:
  - [0] (-P) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
  - [1] (-P) She is now complaining of a cool right foot.
  - [2] (-P) Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
  - [3] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 163 | sigir-201410 | NCT02097186 | inclusion | picked for `evidence_spurious`
- Criterion: Patients undergoing endovascular abdominal aneurysm repair
- Gold: **not included** evidence [] | Predicted: **not enough information** evidence [0, 1, 2] | valid=True
- Tags: false_nei, rare_class_missed, evidence_spurious
- Raw output: `{"label": "not enough information", "evidence_ids": [0, 1, 2]}`
- Note:
  - [0] (-P) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
  - [1] (-P) She is now complaining of a cool right foot.
  - [2] (-P) Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
  - [3] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 174 | sigir-201410 | NCT02097186 | exclusion | picked for `false_confident`
- Criterion: Patients with previous axillary surgery
- Gold: **not enough information** evidence [] | Predicted: **not excluded** evidence [] | valid=True
- Tags: false_confident
- Raw output: `{"label": "not excluded", "evidence_ids": []}`
- Note:
  - [0] (--) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
  - [1] (--) She is now complaining of a cool right foot.
  - [2] (--) Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
  - [3] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 176 | sigir-201410 | NCT02264964 | inclusion | picked for `evidence_missing`
- Criterion: No medical history of central vena catheterization.
- Gold: **included** evidence [0] | Predicted: **not enough information** evidence [] | valid=True
- Tags: false_nei, evidence_missing
- Raw output: `{"label": "not enough information", "evidence_ids": []}`
- Note:
  - [0] (G-) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
  - [1] (--) She is now complaining of a cool right foot.
  - [2] (--) Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
  - [3] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 185 | sigir-201411 | NCT01526382 | inclusion | picked for `false_nei`
- Criterion: Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both.
- Gold: **included** evidence [2, 3] | Predicted: **not enough information** evidence [] | valid=True
- Tags: false_nei, evidence_missing
- Raw output: `{"label": "not enough information", "evidence_ids": []}`
- Note:
  - [0] (--) A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
  - [1] (--) She denies trauma.
  - [2] (G-) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
  - [3] (G-) Her body temperature is normal and her blood pressure is 80/60.
  - [4] (--) Her right arm has no discoloration or movement limitation.
  - [5] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 187 | sigir-201411 | NCT01526382 | inclusion | picked for `evidence_partial`
- Criterion: A respiratory rate of at least 20/min or a PaCO2 of 32mmHg or lower or the use of mechanical ventilation;
- Gold: **included** evidence [2] | Predicted: **not enough information** evidence [2, 3] | valid=True
- Tags: false_nei, evidence_partial
- Raw output: `{"label": "not enough information", "evidence_ids": [2, 3]}`
- Note:
  - [0] (--) A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
  - [1] (--) She denies trauma.
  - [2] (GP) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
  - [3] (-P) Her body temperature is normal and her blood pressure is 80/60.
  - [4] (--) Her right arm has no discoloration or movement limitation.
  - [5] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 188 | sigir-201411 | NCT01526382 | inclusion | picked for `rare_class_missed`
- Criterion: The use of vasopressors to maintain a systolic blood pressure of at least 90mmHg despite fluid resuscitation, low dose of dopamine (≤ 5 μg/kg per minute), or dobutamine;
- Gold: **not included** evidence [3] | Predicted: **not enough information** evidence [] | valid=True
- Tags: false_nei, rare_class_missed, evidence_missing
- Raw output: `{"label": "not enough information", "evidence_ids": []}`
- Note:
  - [0] (--) A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
  - [1] (--) She denies trauma.
  - [2] (--) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
  - [3] (G-) Her body temperature is normal and her blood pressure is 80/60.
  - [4] (--) Her right arm has no discoloration or movement limitation.
  - [5] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 189 | sigir-201411 | NCT01526382 | inclusion | picked for `evidence_spurious`
- Criterion: at least 1 of 3 signs of hypoperfusion (urine output < 0.5mL/kg of body weight per hour for 1 hour or more; neurologic dysfunction defined by confusion, psychosis, or a Glasgow coma scale score of ≤ 6; plasma lactate higher than the upper limit of the normal value).
- Gold: **not enough information** evidence [] | Predicted: **not enough information** evidence [2, 3] | valid=True
- Tags: evidence_spurious
- Raw output: `{"label": "not enough information", "evidence_ids": [2, 3]}`
- Note:
  - [0] (--) A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
  - [1] (--) She denies trauma.
  - [2] (-P) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
  - [3] (-P) Her body temperature is normal and her blood pressure is 80/60.
  - [4] (--) Her right arm has no discoloration or movement limitation.
  - [5] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 193 | sigir-201411 | NCT01526382 | exclusion | picked for `false_confident`
- Criterion: Patients were moribund.
- Gold: **not enough information** evidence [0, 1, 2, 3, 4] | Predicted: **not excluded** evidence [] | valid=True
- Tags: false_confident, evidence_missing
- Raw output: `{"label": "not excluded", "evidence_ids": []}`
- Note:
  - [0] (G-) A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
  - [1] (G-) She denies trauma.
  - [2] (G-) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
  - [3] (G-) Her body temperature is normal and her blood pressure is 80/60.
  - [4] (G-) Her right arm has no discoloration or movement limitation.
  - [5] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 196 | sigir-201411 | NCT01723137 | exclusion | picked for `evidence_missing`
- Criterion: Less than 18 years of age
- Gold: **not excluded** evidence [0] | Predicted: **not excluded** evidence [] | valid=True
- Tags: evidence_missing
- Raw output: `{"label": "not excluded", "evidence_ids": []}`
- Note:
  - [0] (G-) A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
  - [1] (--) She denies trauma.
  - [2] (--) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
  - [3] (--) Her body temperature is normal and her blood pressure is 80/60.
  - [4] (--) Her right arm has no discoloration or movement limitation.
  - [5] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 307 | sigir-201417 | NCT01745731 | inclusion | picked for `evidence_partial`
- Criterion: Patient selection should be cautious, covering basically 5 types of liver damage which must be submitted prior to liver volumetry: Metastatic Disease subsidiary right hepatectomy extended to segment IV Metastatic Disease subsidiary right hepatectomy with suspected diseased liver (neoadjuvant chemotherapy) (in cases of doubt may be used liver function test indocyanine green) Bilobar liver metastases with multiple nodules in the right lobe and more than 3 nodules greater than 30 mm in the left hepatic lobe (LHI) will perform lumpectomies the LHI + right portal branch ligation (or postoperative percutaneous embolization) in order to make right hepatectomy 4-6 weeks (two stage surgery) Subsidiary Hepatocarcinoma extended right hepatectomy Liver Injury benign / malignant (Hemangiomas, hydatid cysts or liver tumors / primary bile hepatoblastoma), which by extension threatens the viability of the remaining liver tissue.
- Gold: **not enough information** evidence [3] | Predicted: **not enough information** evidence [0, 1, 2, 3, 4] | valid=True
- Tags: evidence_partial
- Raw output: `{"label": "not enough information", "evidence_ids": [0, 1, 2, 3, 4]}`
- Note:
  - [0] (-P) A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
  - [1] (-P) The physical examination is remarkable for tenderness and positive Murphy sign.
  - [2] (-P) Abdominal ultrasound shows hepatomegaly and abundant free intraperitoneal fluid.
  - [3] (GP) Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.
  - [4] (-P) After surgery, the patient is taken to the ICU.
  - [5] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 309 | sigir-201417 | NCT01745731 | exclusion | picked for `evidence_spurious`
- Criterion: Different tumor records current disease or any disease hematologic.
- Gold: **not excluded** evidence [] | Predicted: **not excluded** evidence [0, 1, 2, 3, 4] | valid=True
- Tags: evidence_spurious
- Raw output: `{"label": "not excluded", "evidence_ids": [0, 1, 2, 3, 4]}`
- Note:
  - [0] (-P) A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
  - [1] (-P) The physical examination is remarkable for tenderness and positive Murphy sign.
  - [2] (-P) Abdominal ultrasound shows hepatomegaly and abundant free intraperitoneal fluid.
  - [3] (-P) Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.
  - [4] (-P) After surgery, the patient is taken to the ICU.
  - [5] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 319 | sigir-201417 | NCT01745731 | exclusion | picked for `evidence_missing`
- Criterion: Difficulty in monitoring.
- Gold: **not excluded** evidence [5] | Predicted: **not excluded** evidence [] | valid=True
- Tags: evidence_missing
- Raw output: `{"label": "not excluded", "evidence_ids": []}`
- Note:
  - [0] (--) A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
  - [1] (--) The physical examination is remarkable for tenderness and positive Murphy sign.
  - [2] (--) Abdominal ultrasound shows hepatomegaly and abundant free intraperitoneal fluid.
  - [3] (--) Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.
  - [4] (--) After surgery, the patient is taken to the ICU.
  - [5] (G-) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 322 | sigir-201417 | NCT01745731 | exclusion | picked for `na_mistake`
- Criterion: Pregnant women or women of childbearing age who do not have adequate contraception.
- Gold: **not excluded** evidence [0] | Predicted: **not applicable** evidence [] | valid=True
- Tags: na_mistake, evidence_missing
- Raw output: `{"label": "not applicable", "evidence_ids": []}`
- Note:
  - [0] (G-) A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
  - [1] (--) The physical examination is remarkable for tenderness and positive Murphy sign.
  - [2] (--) Abdominal ultrasound shows hepatomegaly and abundant free intraperitoneal fluid.
  - [3] (--) Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.
  - [4] (--) After surgery, the patient is taken to the ICU.
  - [5] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 381 | sigir-201421 | NCT00997100 | inclusion | picked for `evidence_wrong_ids`
- Criterion: Age > 18 years at the time of signing the informed consent form
- Gold: **included** evidence [0] | Predicted: **included** evidence [4] | valid=True
- Tags: evidence_wrong_ids
- Raw output: `{"label": "included", "evidence_ids": [4]}`
- Note:
  - [0] (G-) A 21-year-old female is evaluated for progressive arthralgias and malaise.
  - [1] (--) On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles.
  - [2] (--) Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA.
  - [3] (--) Her urine is positive for protein and RBC casts.
  - [4] (-P) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 382 | sigir-201421 | NCT00997100 | inclusion | picked for `evidence_partial`
- Criterion: Fulfil at least 4 criteria for SLE as defined by the American College of Rheumatology (ACR)
- Gold: **included** evidence [0, 1, 2] | Predicted: **included** evidence [0, 1, 2, 3] | valid=True
- Tags: evidence_partial
- Raw output: `{"label": "included", "evidence_ids": [0, 1, 2, 3]}`
- Note:
  - [0] (GP) A 21-year-old female is evaluated for progressive arthralgias and malaise.
  - [1] (GP) On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles.
  - [2] (GP) Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA.
  - [3] (-P) Her urine is positive for protein and RBC casts.
  - [4] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 385 | sigir-201421 | NCT00997100 | inclusion | picked for `false_nei`
- Criterion: Ability to take and retain oral medication
- Gold: **included** evidence [4] | Predicted: **not enough information** evidence [4] | valid=True
- Tags: false_nei
- Raw output: `{"label": "not enough information", "evidence_ids": [4]}`
- Note:
  - [0] (--) A 21-year-old female is evaluated for progressive arthralgias and malaise.
  - [1] (--) On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles.
  - [2] (--) Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA.
  - [3] (--) Her urine is positive for protein and RBC casts.
  - [4] (GP) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 389 | sigir-201421 | NCT00997100 | exclusion | picked for `false_confident`
- Criterion: Severe renal impairment (estimated or measured GFR <50%)
- Gold: **not enough information** evidence [] | Predicted: **not excluded** evidence [] | valid=True
- Tags: false_confident
- Raw output: `{"label": "not excluded", "evidence_ids": []}`
- Note:
  - [0] (--) A 21-year-old female is evaluated for progressive arthralgias and malaise.
  - [1] (--) On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles.
  - [2] (--) Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA.
  - [3] (--) Her urine is positive for protein and RBC casts.
  - [4] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 422 | sigir-201422 | NCT00236912 | inclusion | picked for `na_mistake`
- Criterion: If female, using birth control
- Gold: **not applicable** evidence [3] | Predicted: **not enough information** evidence [3] | valid=True
- Tags: na_mistake
- Raw output: `{"label": "not enough information", "evidence_ids": [3]}`
- Note:
  - [0] (--) A 15-year-old girl presents to the ER with abdominal pain.
  - [1] (--) The pain appeared gradually and was periumbilical at first, localizing to the right lower quadrant over hours.
  - [2] (--) She has had no appetite since yesterday but denies diarrhea.
  - [3] (GP) She has had no sexual partners and her menses are regular.
  - [4] (--) On examination, she has localized rebound tenderness over the right lower quadrant.
  - [5] (--) On an abdominal ultrasound, a markedly edematous appendix is seen.
  - [6] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 683 | sigir-20158 | NCT02562040 | exclusion | picked for `na_mistake`
- Criterion: Previous tonsillectomy, including partial tonsillectomy
- Gold: **not applicable** evidence [] | Predicted: **not excluded** evidence [] | valid=True
- Tags: na_mistake
- Raw output: `{"label": "not excluded", "evidence_ids": []}`
- Note:
  - [0] (--) A 10 yo boy with nighttime snoring, pauses in breathing, and restlessness with nighttime awakenings.
  - [1] (--) No history of headache or night terrors.
  - [2] (--) The boy's teacher recently contacted his parents because she was concerned about his declining grades, lack of attention, and excessive sleepiness during class.
  - [3] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO

## annotation 836 | sigir-201516 | NCT00839124 | inclusion | picked for `rare_class_missed`
- Criterion: History of episodic wheezing, chest tightness, or shortness of breath after age of 6 years consistent with asthma, or physician diagnosed asthma after age of 6 years.
- Gold: **not included** evidence [0, 1] | Predicted: **not enough information** evidence [1, 3, 4] | valid=True
- Tags: false_nei, rare_class_missed, evidence_partial
- Raw output: `{"label": "not enough information", "evidence_ids": [1, 3, 4]}`
- Note:
  - [0] (G-) A 4 year old boy presents to the emergency room with wheezing.
  - [1] (GP) He has had a history of allergic rhinitis, but no history of wheezing.
  - [2] (--) His mother reports that 5 hours ago patient was playing in the backyard sandbox when she heard him suddenly start coughing.
  - [3] (-P) The coughing lasted only moments, but he has been audibly wheezing since.
  - [4] (-P) Mother was concerned, because his breathing has not returned to normal, so she brought him to the ED.
  - [5] (--) On exam, the child is playful and well appearing.
  - [6] (--) Wheezing is heard in the mid-right chest area.
  - [7] (--) O2 sats are 100% on room air.
  - [8] (--) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- Cause hypothesis: TODO
