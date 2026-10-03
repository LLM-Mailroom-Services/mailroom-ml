# TEST-EVAL — Arm A (`m9a-local-gpu1-armA-1ep-20260927-025236`)

**Eval role:** authoritative held-out test for M9a **#112** gates and Hub release  
**Run tag:** `m9a-local-gpu1-armA-1ep-20260927-025236`  
**Arm:** A  
**Checkpoint:** `data/modernbert_training/runs/m9a-local-gpu1-armA-1ep-20260927-025236/latest`  
**Eval JSON (this dir):** `reports/TEST-EVAL/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json`  
**Eval JSON (canonical):** `reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json`  
**Artifact SHA:** `5ecea781599be293dcc80f32ac4b57b478d6787aed2c7bbf2cc2a01858b60aae`  
**Training log:** `logs/m9a-local-gpu1-armA-1ep-20260927-025236.log`  
**Training report:** `reports/M9a-REPORT-m9a-local-gpu1-armA-1ep-20260927-025236.md`

## Test harness protocol

| Parameter | Value |
| --- | --- |
| Command surface | `training/eval_modernbert.py` |
| `--subset` | `test` |
| Documents | **323** (`--sample 0`) |
| `--max-length` | 8192 |
| `--seed` | 42 |
| Windows evaluated | 541 (`window_calibration.n_windows`) |

The test split is isolated from training and validation checkpoint selection. Report-only **P0** thresholds in `recorded_gates` are diagnostic; **#112** gates below are the release bar.

## Headline test metrics (eval harness)

| Metric | Value |
| --- | ---: |
| doc_type_accuracy | **0.8638** (279/323) |
| subclass_accuracy_conditional | **0.4839** (135/279 scorable) |
| window_calibration ECE | **0.0397** |
| window_calibration band ECE | 0.0574 |
| fast_path_rate | 0.0372 |

## Cohorts (#104 single vs multi-window)

| cohort | n_docs | doc_type_accuracy | mean_agreement | window ECE |
| --- | ---: | ---: | ---: | ---: |
| single-window | 274 | 0.8723 | 1.0000 | 0.1179 |
| multi-window | 49 | 0.8163 | 0.8020 | 0.1024 |

## Selective-risk sweep (global)

| Field | Value |
| --- | --- |
| budget_met | **None** |
| recommended deployment threshold | **None** |
| error_budget | None |
| coverage at pick | ~— |

Per-cohort selective-risk on multi-window reports `budget_met: None` with `recommended_threshold: None`.

## Per-head test macro-F1 (observed)

Support counts are per-class **test** support from `per_head.<head>.support`.

| head | macro-F1 | notes |
| --- | ---: | --- |
| contract | 0.0039 | majority-prior collapse persists |
| corporate_record | 0.0212 |  |
| correspondence | 0.0944 | below #112 floor |
| insurance_claim | 0.7552 | partial learning |
| merger_agreement | 0.1600 |  |

## M9a #112 gates (held-out test)

Verified with `training/check_m9a_gates.py` → `reports/TEST-EVAL/gates-check-armA.txt`.

| Gate | Actual | Threshold | Status |
| --- | ---: | ---: | --- |
| contract test macro-F1 | 0.0039 | ≥ 0.2 | **NOT MET** |
| correspondence test macro-F1 | 0.0944 | ≥ 0.25 | **NOT MET** |
| doc_type test accuracy | 0.8638 | ≥ 0.89 | **NOT MET** |
| window ECE (doc_type calibrated) | 0.0397 | ≤ 0.05 | MET |

**Overall #112 gate verdict:** **FAIL** (3/4 gates not met).

### Report-only P0 (eval harness)

| Gate | Actual | Threshold | met |
| --- | ---: | ---: | --- |
| P0 doc type | 0.8638 | 0.95 | False |
| P0 subclass | 0.4839 | 0.75 | False |

## Baseline comparison (doc_type test accuracy)

| source | doc_type_accuracy |
| ---: | ---: |
| run-3 (`eval_run3_20260921.json`) | 0.8947 |
| **this run (A)** | 0.8638 |
| Arm A smoke | 0.8638 |

Δ vs run-3: **-3.09 pp** doc_type; conditional subclass 0.4839 on this checkpoint.

## Hub release

| Field | Value |
| --- | --- |
| Model repo | `Lucius-Morningstar/mailroom-modernbert-classifier` |
| Release tag | `m9a-local-gpu1-armA-1ep-20260927-025236` |
| Eval on Hub | `eval_report_m9a-local-gpu1-armA-1ep-20260927-025236.json` |

Publish via `./training/complete_run.sh --run-tag m9a-local-gpu1-armA-1ep-20260927-025236 --publish` (requires all #112 gates MET, or `--force-publish`).

## Analyst insights & findings

- **Doc_type routing:** best `merger_agreement` 1.000 acc (n=17); weakest `contract` 0.683 acc (n=60).
- **Subclass bottleneck:** lowest test macro-F1 head `contract` at 0.0039 (#112 floors: contract ≥0.20, correspondence ≥0.25).
- **Window cohort (#104):** multi-window doc_type acc 0.8163 vs single-window 0.8723 (n=49/274).
- **Fast-path rate:** 0.0372 of docs would route on classifier gate.

## TEST-EVAL verdict

- **#112 gates:** **FAIL** on held-out test.
- **doc_type_accuracy:** 0.8638; **window ECE:** 0.0397.
- **Subclass conditional:** 0.4839.

## Per-document scores

| filename | gt doc_type | pred | dt ok | sc ok | n_win | fast_path |
| --- | --- | --- | --- | --- | ---: | --- |
| `0000899243-02-000452_dex211.txt` | corporate_record | correspondence | False | False | 1 | False |
| `0000950123-11-000444_y86334a4exv24w3.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0000950134-08-011905_f38510a4exv3w2w2.htm` | corporate_record | corporate_record | True | False | 2 | False |
| `0000950137-05-012449_i98663a1exv4w1.txt` | corporate_record | corporate_record | True | True | 9 | False |
| `0000950137-06-011083_n08078exv3w03.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0000950137-07-001855_x09498a3exv3w03.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0000950172-01-501237_s558625.txt` | corporate_record | corporate_record | True | True | 12 | False |
| `0001047469-05-004497_a2151900zex-3_1.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001047469-06-009032_a2170944zex-4_32.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001047469-15-005538_a2225135zex-10_19.htm` | contract | merger_agreement | False | False | 2 | False |
| `0001047469-17-007858_a2234150zex-3_1.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001104659-07-060356_a07-15464_2ex3d1.htm` | corporate_record | corporate_record | True | False | 2 | False |
| `0001144204-13-006917_v333161_ex10-32.htm` | contract | corporate_record | False | False | 1 | False |
| `0001144204-14-036356_v380646_ex3-3.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001144204-15-070401_v426454_ex3-2.htm` | corporate_record | corporate_record | True | False | 2 | False |
| `0001193125-04-096445_dex211.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001193125-06-074669_dex31a2.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-06-112108_dex415.htm` | corporate_record | corporate_record | True | True | 1 | False |
| `0001193125-07-204979_dex211.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-08-114390_dex3144.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-09-259347_dex21.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001193125-10-059057_dex211.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001193125-10-283621_dex242.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-11-064249_dex31.htm` | corporate_record | corporate_record | True | False | 3 | False |
| `0001193125-11-095556_dex211.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001193125-11-277426_d207460dex211.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001193125-11-336311_d230618dex43.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-12-191562_d267959dex242.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-12-428863_d388365dex211.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-14-174317_d658316dex32.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-14-352440_d779077dex410.htm` | corporate_record | corporate_record | True | True | 1 | False |
| `0001193125-14-382241_d744509dex36.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-18-229893_d522375dex43.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-21-169895_d155073dex34.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-22-167822_d271308dex34.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001193125-24-221320_d829976dex21.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001193125-25-240868_d28749dex31.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001213900-15-009308_fs12015a3ex21i_shimmick.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001213900-17-006771_fs12017a2ex3ix_inpixon.htm` | corporate_record | corporate_record | True | False | 2 | False |
| `0001213900-18-010616_f8k041017a1ex3-1_anutra.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001437749-18-018936_ex_126721.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001437749-20-019373_ex_203002.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001437749-20-020141_ex_204901.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001493152-16-006927_ex10-23.htm` | contract | contract | True | True | 1 | False |
| `0001493152-17-002036_ex3-1.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001493152-17-007882_ex3-3.htm` | corporate_record | corporate_record | True | False | 2 | False |
| `0001493152-22-019482_ex10-10.htm` | contract | corporate_record | False | False | 1 | False |
| `0001547903-13-000006_a42registrationrightsagree.htm` | corporate_record | corporate_record | True | False | 3 | False |
| `0001567619-13-000087_s000086x3_ex10-3.htm` | contract | corporate_record | False | False | 1 | False |
| `0001628280-18-007944_exhibit1013s-1a.htm` | contract | corporate_record | False | False | 1 | False |
| `0001683168-22-000662_gamingtech_ex0303.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001683168-23-005255_cardiff_ex0309.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001753926-23-001338_g083766_ex3-5.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agreement.PDF` | contract | contract | True | False | 2 | False |
| `AgapeAtpCorp_20191202_10-KA_EX-10.1_11911128_EX-10.1_Supply Agreement.pdf` | contract | contract | True | False | 1 | False |
| `ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship Agreement.pdf` | contract | contract | True | False | 1 | False |
| `ArcaUsTreasuryFund_20200207_N-2_EX-99.K5_11971930_EX-99.K5_Development Agreement.pdf` | contract | contract | True | False | 1 | False |
| `ArconicRolledProductsCorp_20191217_10-12B_EX-2.7_11923804_EX-2.7_Trademark License Agreement.pdf` | contract | contract | True | False | 1 | False |
| `Array BioPharma Inc. - LICENSE, DEVELOPMENT AND COMMERCIALIZATION AGREEMENT.PDF` | contract | merger_agreement | False | False | 8 | False |
| `AzulSa_20170303_F-1A_EX-10.3_9943903_EX-10.3_Maintenance Agreement2.pdf` | contract | merger_agreement | False | False | 1 | False |
| `BEYONDCOMCORP_08_03_2000-EX-10.2-CO-HOSTING AGREEMENT.PDF` | contract | contract | True | False | 2 | False |
| `BICYCLETHERAPEUTICSPLC_03_10_2020-EX-10.11-SERVICE AGREEMENT.PDF` | contract | contract | True | False | 2 | False |
| `BLACKBOXSTOCKSINC_08_05_2014-EX-10.1-DISTRIBUTOR AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `BORROWMONEYCOM,INC_06_11_2020-EX-10.1-JOINT VENTURE AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `COOLTECHNOLOGIES,INC_10_25_2017-EX-10.71-Strategic Alliance Agreement.PDF` | contract | corporate_record | False | False | 1 | False |
| `CURAEGISTECHNOLOGIES,INC_05_26_2010-EX-1-CORPORATE SPONSORSHIP AGREEMENT.PDF` | contract | corporate_record | False | False | 1 | False |
| `CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance Agreement2.pdf` | contract | correspondence | False | False | 1 | False |
| `ClickstreamCorp_20200330_1-A_EX1A-6 MAT CTRCT_12089935_EX1A-6 MAT CTRCT_Development Agreement.pdf` | contract | contract | True | False | 1 | False |
| `Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT.PDF` | contract | contract | True | False | 2 | False |
| `CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf` | contract | contract | True | False | 1 | False |
| `DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.PDF` | contract | contract | True | False | 2 | False |
| `EbixInc_20010515_10-Q_EX-10.3_4049767_EX-10.3_Co-Branding Agreement.pdf` | contract | contract | True | False | 2 | False |
| `EdietsComInc_20001030_10QSB_EX-10.4_2606646_EX-10.4_Co-Branding Agreement.pdf` | contract | contract | True | False | 2 | False |
| `FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_AMENDMENT.pdf` | contract | contract | True | False | 1 | False |
| `FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_POWEROF.pdf` | contract | contract | True | False | 1 | False |
| `FIDELITYNATIONALINFORMATIONSERVICES,INC_08_05_2009-EX-10.3-INTELLECTUAL PROPERTY AGREEMENT.PDF` | contract | merger_agreement | False | False | 4 | False |
| `GWG HOLDINGS, INC. - ORDERLY MARKETING AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `IMAGEWARESYSTEMSINC_12_20_1999-EX-10.22-MAINTENANCE AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `INKTOMICORP_06_08_1998-EX-10.14-SOFTWARE HOSTING AGREEMENT.PDF` | contract | contract | True | False | 2 | False |
| `IntegrityMediaInc_20010329_10-K405_EX-10.17_2373875_EX-10.17_Co-Branding Agreement.pdf` | contract | contract | True | False | 1 | False |
| `JINGWEIINTERNATIONALLTD_10_04_2007-EX-10.7-INTELLECTUAL PROPERTY AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `KENTUCKYUTILITIESCO_03_25_2003-EX-10.65-TRANSPORTATION AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `KUBIENT,INC_07_02_2020-EX-10.14-MASTER SERVICES AGREEMENT_Part1.pdf` | contract | contract | True | False | 1 | False |
| `KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing Agreement.pdf` | contract | contract | True | False | 2 | False |
| `LEJUHOLDINGSLTD_03_12_2014-EX-10.34-INTERNET CHANNEL COOPERATION AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `LUCIDINC_04_15_2011-EX-10.9-DISTRIBUTOR AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `LejuHoldingsLtd_20140121_DRS (on F-1)_EX-10.26_8473102_EX-10.26_Content License Agreement1.pdf` | contract | contract | True | False | 2 | False |
| `LinkPlusCorp_20050802_8-K_EX-10_3240252_EX-10_Affiliate Agreement.pdf` | contract | corporate_record | False | False | 1 | False |
| `LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement.pdf` | contract | contract | True | False | 1 | False |
| `MRSFIELDSORIGINALCOOKIESINC_01_29_1998-EX-10-FRANCHISE AGREEMENT.PDF` | contract | merger_agreement | False | False | 7 | False |
| `OAKTREECAPITALGROUP,LLC_03_02_2020-EX-10.8-Services Agreement.PDF` | contract | contract | True | False | 1 | False |
| `PACIFICSYSTEMSCONTROLTECHNOLOGYINC_08_24_2000-EX-10.53-SPONSORSHIP AGREEMENT.PDF` | contract | corporate_record | False | False | 1 | False |
| `PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT .PDF` | contract | contract | True | False | 4 | False |
| `PcquoteComInc_19990721_S-1A_EX-10.11_6377149_EX-10.11_Co-Branding Agreement3.pdf` | contract | contract | True | False | 1 | False |
| `PelicanDeliversInc_20200211_S-1_EX-10.3_11975895_EX-10.3_Development Agreement1.pdf` | contract | contract | True | False | 1 | False |
| `REWALKROBOTICSLTD_07_10_2014-EX-10.2-STRATEGIC ALLIANCE AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `RandWorldwideInc_20010402_8-KA_EX-10.2_2102464_EX-10.2_Co-Branding Agreement.pdf` | contract | contract | True | False | 2 | False |
| `RangeResourcesLouisianaInc_20150417_8-K_EX-10.5_9045501_EX-10.5_Transportation Agreement.pdf` | contract | merger_agreement | False | False | 3 | False |
| `SLOVAKWIRELESSFINANCECOBV_03_28_2001-EX-4.(B)(II).3-Maintenance and support contract for SICAP(R) modules.PDF` | contract | contract | True | False | 1 | False |
| `SOLUTIONSVENDINGINTERNATIONAL,INC_03_31_2020-EX1A-1 UNDR AGMT-SERVICES AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `SmartRxSystemsInc_20180914_1-A_EX1A-6 MAT CTRCT_11351705_EX1A-6 MAT CTRCT_Distributor Agreement.pdf` | contract | merger_agreement | False | False | 2 | False |
| `TRANSPHORM,INC_02_14_2020-EX-10.12(1)-JOINT VENTURE AGREEMENT.PDF` | contract | contract | True | False | 3 | False |
| `URSCORPNEW_03_17_2014-EX-99-COOPERATION AGREEMENT.PDF` | contract | corporate_record | False | False | 1 | False |
| `WPPPLC_04_30_2020-EX-4.28-SERVICE AGREEMENT.PDF` | contract | merger_agreement | False | False | 2 | False |
| `WestPharmaceuticalServicesInc_20200116_8-K_EX-10.1_11947529_EX-10.1_Supply Agreement.pdf` | contract | merger_agreement | False | False | 3 | False |
| `XLITECHNOLOGIES,INC_12_02_2015-EX-10.02-STRATEGIC ALLIANCE AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor Agreement.pdf` | contract | contract | True | False | 6 | False |
| `auto:CLM-000049.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000058.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000178.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000228.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000267.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000269.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000333.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000365.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000381.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000483.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000500.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000535.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000583.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000663.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000803.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `auto:CLM-000973.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `bailey-s/deleted_items/253.` | correspondence | correspondence | True | False | 1 | False |
| `bass-e/_sent_mail/1208.` | correspondence | correspondence | True | True | 1 | False |
| `beck-s/hpl/9.` | correspondence | correspondence | True | True | 1 | False |
| `beck-s/inbox/509.` | correspondence | correspondence | True | False | 1 | False |
| `carrier:887093388938040.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887103388874951.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887143385158118.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887203387940481.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887253387243387.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887263387629219.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `carrier:887273385086013.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887423388592720.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `carrier:887433385493589.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887433387270005.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887473387486812.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887533386749226.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887593385441561.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887603388968925.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887673388415242.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887683388914415.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887793386319985.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887843389522899.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887903385471960.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `carrier:887923388522232.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `cash-m/all_documents/148.` | correspondence | correspondence | True | False | 1 | False |
| `cash-m/all_documents/482.` | correspondence | correspondence | True | False | 1 | False |
| `contract_104_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 9 | True |
| `contract_105_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 13 | True |
| `contract_106_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 7 | True |
| `contract_107_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 9 | True |
| `contract_108_merger_agreement.txt` | merger_agreement | merger_agreement | True | True | 12 | True |
| `contract_109_merger_agreement.txt` | merger_agreement | merger_agreement | True | True | 11 | False |
| `contract_110_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 9 | False |
| `contract_123_merger_agreement.txt` | merger_agreement | merger_agreement | True | True | 11 | True |
| `contract_135_merger_agreement.txt` | merger_agreement | merger_agreement | True | True | 10 | False |
| `contract_139_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 8 | True |
| `contract_16_merger_agreement.txt` | merger_agreement | merger_agreement | True | True | 8 | True |
| `contract_18_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 9 | True |
| `contract_29_merger_agreement.txt` | merger_agreement | merger_agreement | True | True | 9 | True |
| `contract_48_merger_agreement.txt` | merger_agreement | merger_agreement | True | True | 9 | False |
| `contract_67_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 8 | True |
| `contract_74_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 9 | False |
| `contract_91_merger_agreement.txt` | merger_agreement | merger_agreement | True | True | 11 | True |
| `cuilla-m/inbox/79.` | correspondence | corporate_record | False | False | 1 | False |
| `dasovich-j/all_documents/10778.` | correspondence | contract | False | False | 1 | False |
| `dasovich-j/all_documents/28272.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/28798.` | correspondence | correspondence | True | True | 1 | False |
| `dasovich-j/all_documents/3687.` | correspondence | corporate_record | False | False | 1 | False |
| `dasovich-j/deleted_items/1254.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/sent_items/418.` | correspondence | correspondence | True | False | 1 | False |
| `ermis-f/all_documents/303.` | correspondence | correspondence | True | False | 1 | False |
| `farmer-d/all_documents/991.` | correspondence | correspondence | True | True | 1 | False |
| `farmer-d/pan_energy_swap/10.` | correspondence | correspondence | True | True | 1 | False |
| `fischer-m/all_documents/71.` | correspondence | correspondence | True | False | 1 | False |
| `fossum-d/all_documents/643.` | correspondence | contract | False | False | 1 | False |
| `giron-d/_sent_mail/29.` | correspondence | correspondence | True | True | 1 | False |
| `grigsby-m/_sent_mail/79.` | correspondence | correspondence | True | True | 1 | False |
| `grigsby-m/deleted_items/244.` | correspondence | contract | False | False | 1 | False |
| `haedicke-m/all_documents/4683.` | correspondence | correspondence | True | True | 1 | False |
| `hayslett-r/all_documents/217.` | correspondence | correspondence | True | True | 1 | False |
| `hayslett-r/inbox/216.` | correspondence | correspondence | True | False | 1 | False |
| `hernandez-j/all_documents/39.` | correspondence | correspondence | True | True | 1 | False |
| `horton-s/all_documents/162.` | correspondence | correspondence | True | False | 1 | False |
| `horton-s/all_documents/93.` | correspondence | correspondence | True | False | 1 | False |
| `hyatt-k/deleted_items/553.` | correspondence | correspondence | True | True | 1 | False |
| `inpatient:196071176982440:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `inpatient:196121176993224:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `inpatient:196231176979546:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `inpatient:196261176970939:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `inpatient:196291176990524:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `inpatient:196351177024814:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `inpatient:196361177004078:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `inpatient:196371176965639:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `inpatient:196491176981796:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `inpatient:196521176961719:1.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `inpatient:196581176967508:1.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `inpatient:196711176997740:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `inpatient:196941177012227:1.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `inpatient:196971176984049:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `inpatient:196991176984856:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-1049.txt` | insurance_claim | correspondence | False | False | 1 | False |
| `insurbias-1075.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-1180.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-1224.txt` | insurance_claim | correspondence | False | False | 1 | False |
| `insurbias-1310.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-1328.txt` | insurance_claim | correspondence | False | False | 1 | False |
| `insurbias-149.txt` | insurance_claim | correspondence | False | False | 1 | False |
| `insurbias-242.txt` | insurance_claim | correspondence | False | False | 1 | False |
| `insurbias-282.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-329.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-354.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-356.txt` | insurance_claim | correspondence | False | False | 1 | False |
| `insurbias-436.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-491.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-701.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-809.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-859.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `insurbias-886.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `jones-t/all_documents/10358.` | correspondence | correspondence | True | False | 1 | False |
| `jones-t/all_documents/1795.` | correspondence | correspondence | True | True | 1 | False |
| `jones-t/all_documents/3795.` | correspondence | correspondence | True | True | 1 | False |
| `kaminski-v/all_documents/2165.` | correspondence | correspondence | True | True | 1 | False |
| `kaminski-v/conferences/310.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/deleted_items/82.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/inbox/653.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/inbox/81.` | correspondence | correspondence | True | False | 1 | False |
| `kean-s/all_documents/1802.` | correspondence | correspondence | True | False | 1 | False |
| `keavey-p/all_documents/433.` | correspondence | correspondence | True | False | 1 | False |
| `kitchen-l/_americas/finance/29.` | correspondence | correspondence | True | True | 1 | False |
| `kitchen-l/_americas/mrha/ooc/175.` | correspondence | correspondence | True | False | 1 | False |
| `lay-k/inbox/62.` | correspondence | correspondence | True | True | 1 | False |
| `lay-k/inbox/732.` | correspondence | correspondence | True | False | 1 | False |
| `lay-k/inbox/852.` | correspondence | correspondence | True | True | 1 | False |
| `lenhart-m/sent_items/346.` | correspondence | correspondence | True | True | 1 | False |
| `lokay-m/hea_nesa/75.` | correspondence | correspondence | True | True | 1 | False |
| `lokay-m/tw_commercial_group/1663.` | correspondence | correspondence | True | True | 1 | False |
| `love-p/_sent_mail/682.` | correspondence | correspondence | True | True | 1 | False |
| `love-p/deleted_items/127.` | correspondence | correspondence | True | True | 1 | False |
| `mann-k/_sent_mail/2929.` | correspondence | correspondence | True | True | 1 | False |
| `mann-k/all_documents/1165.` | correspondence | correspondence | True | True | 1 | False |
| `mann-k/all_documents/1951.` | correspondence | correspondence | True | False | 1 | False |
| `mccarty-d/inbox/91.` | correspondence | correspondence | True | False | 1 | False |
| `mcconnell-m/_sent_mail/651.` | correspondence | correspondence | True | False | 1 | False |
| `mckay-b/deleted_items/137.` | correspondence | corporate_record | False | False | 1 | False |
| `meyers-a/deleted_items/800.` | correspondence | correspondence | True | True | 1 | False |
| `nemec-g/all_documents/6413.` | correspondence | correspondence | True | False | 1 | False |
| `outpatient:542232281499617:1.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `outpatient:542342281167843:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542342281644955:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542352281466028:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542362281309404:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542412281513838:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542472280879535:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542632281150539:1.txt` | insurance_claim | insurance_claim | True | False | 1 | False |
| `outpatient:542692281071580:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542722281184051:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542792281348241:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542882281481266:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542902281420106:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542912280980211:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `outpatient:542952281224872:1.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233144493674571.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233174490714832.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233354491440443.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233444490300134.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233474492850363.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233514490718493.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233554489194247.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233644494037809.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233724490153267.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233844488740197.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233884488751635.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233934493455308.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233974490586206.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `pde:233994489555124.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `perlingiere-d/all_documents/5.` | correspondence | correspondence | True | True | 1 | False |
| `perlingiere-d/deleted_items/116.` | correspondence | insurance_claim | False | False | 1 | False |
| `perlingiere-d/sent_items/196.` | correspondence | correspondence | True | True | 1 | False |
| `property:261137610.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:261163075.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:261261502.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:261371580.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:261823130.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:262921804.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:263239641.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:263349821.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:263478129.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:264043643.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:264211938.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:266174390.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:266258735.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:266459530.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:267387009.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `property:268734494.txt` | insurance_claim | insurance_claim | True | True | 1 | False |
| `quigley-d/deleted_items/110.` | correspondence | correspondence | True | False | 1 | False |
| `rapp-b/deleted_items/170.` | correspondence | correspondence | True | False | 1 | False |
| `rogers-b/inbox/288.` | correspondence | correspondence | True | True | 1 | False |
| `sager-e/all_documents/1519.` | correspondence | correspondence | True | False | 1 | False |
| `sager-e/all_documents/1942.` | correspondence | correspondence | True | False | 1 | False |
| `sager-e/all_documents/254.` | correspondence | correspondence | True | False | 1 | False |
| `sanders-r/all_documents/3602.` | correspondence | correspondence | True | False | 1 | False |
| `sanders-r/all_documents/759.` | correspondence | correspondence | True | False | 1 | False |
| `sanders-r/deleted_items/182.` | correspondence | correspondence | True | False | 1 | False |
| `sanders-r/deleted_items/542.` | correspondence | contract | False | False | 2 | False |
| `schoolcraft-d/inbox/junk/212.` | correspondence | correspondence | True | True | 1 | False |
| `shackleton-s/all_documents/2755.` | correspondence | contract | False | False | 1 | False |
| `shackleton-s/all_documents/3429.` | correspondence | correspondence | True | True | 1 | False |
| `shackleton-s/deleted_items/589.` | correspondence | correspondence | True | False | 1 | False |
| `shackleton-s/sent_items/715.` | correspondence | correspondence | True | True | 1 | False |
| `skilling-j/all_documents/1390.` | correspondence | correspondence | True | False | 1 | False |
| `skilling-j/all_documents/1522.` | correspondence | correspondence | True | True | 1 | False |
| `skilling-j/inbox/1577.` | correspondence | correspondence | True | False | 1 | False |
| `smith-m/deleted_items/84.` | correspondence | insurance_claim | False | False | 1 | False |
| `solberg-g/inbox/102.` | correspondence | correspondence | True | True | 1 | False |
| `stokley-c/chris_stokley/sent/487.` | correspondence | correspondence | True | False | 1 | False |
| `symes-k/_sent_mail/1100.` | correspondence | correspondence | True | True | 1 | False |
| `symes-k/all_documents/3111.` | correspondence | correspondence | True | False | 1 | False |
| `symes-k/all_documents/799.` | correspondence | correspondence | True | True | 1 | False |
| `tholt-j/_sent_mail/105.` | correspondence | correspondence | True | True | 1 | False |
| `zipper-a/deleted_items/135.` | correspondence | correspondence | True | True | 1 | False |

## Reproduce

```bash
uv run python training/eval_modernbert.py --checkpoint data/modernbert_training/runs/m9a-local-gpu1-armA-1ep-20260927-025236/latest \
  --stage data/modernbert_training/stage --sample 0 --selective-risk \
  --write-routing-thresholds "${CKPT}/routing_thresholds.json" --json \
  > reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json
uv run python training/check_m9a_gates.py reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json
uv run python training/write_eval_report.py test --run-tag m9a-local-gpu1-armA-1ep-20260927-025236 \
  --eval-json reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json
```

## Artifacts

| path | role |
| --- | --- |
| `reports/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json` | canonical eval JSON |
| `reports/TEST-EVAL/eval_m9a-local-gpu1-armA-1ep-20260927-025236.json` | TEST-EVAL copy |
| `reports/M9a-REPORT-m9a-local-gpu1-armA-1ep-20260927-025236.md` | training-focused report |
| `reports/TEST-EVAL/TEST-EVAL-REPORT-m9a-local-gpu1-armA-1ep-20260927-025236.md` | held-out test harness report |
| `reports/TEST-EVAL/MANIFEST.json` | run index + gate snapshot |
