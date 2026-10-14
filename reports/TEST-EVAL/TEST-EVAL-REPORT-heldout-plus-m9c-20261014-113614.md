# TEST-EVAL — held-out-plus (`m9c-20261014-113614`)

**Eval role:** extended monitoring pool (1,323 docs); **#112 gates remain on canonical 323 only** (`--subset test`).
**Run tag:** `m9c-20261014-113614`
**Eval JSON:** `eval_m9c-20261014-113614-heldout-plus.json`
**Checkpoint:** `/checkpoints/runs/20261014-113614`
**Artifact SHA:** `05301a83bba45c8fab557db014555d3045f1b4a44444fed3d9e8d477a2a71136`

## Harness protocol

| Parameter | Value |
| --- | --- |
| CLI | `training/eval/eval_modernbert.py` |
| `--subset` | `heldout-plus` |
| Documents | **1323** (`--sample 0`) |
| Windows | **1554** |
| `--max-length` | 8192 |
| seed | 42 |

## Headline metrics (full 1,323)

| Metric | Value |
| --- | ---: |
| doc_type_accuracy | **0.9766** |
| subclass_accuracy_conditional | **0.8011** |
| window ECE | 0.0148 |
| fast_path_rate | 0.1005 |

## Slices (canonical 323 vs Enron plus 1,000)

| slice | n | doc_type_acc | subclass (cond.) |
| --- | ---: | ---: | ---: |
| canonical test | 323 | 0.9598 | 0.7387 |
| plus v1 (Enron) | 1000 | 0.982 | 0.8208 |

### Per doc_type (plus slice)

| doc_type | n | doc_type_acc |
| --- | ---: | ---: |
| correspondence | 1000 | 0.982 |

## Per-head test macro-F1 (observed)

Support counts are per-class **test** support from `per_head.<head>.support`.

| head | macro-F1 | notes |
| --- | ---: | --- |
| contract | 0.3589 | partial learning |
| corporate_record | 0.7733 | partial learning |
| correspondence | 0.3174 | partial learning |
| insurance_claim | 1.0000 | near-balanced control head |
| merger_agreement | 0.1774 |  |

## LLM sorter comparable metrics

Schema `mailroom-ml/comparable-metrics/v1` in eval JSON.

| Metric | Value |
| --- | ---: |
| doc_type_accuracy | 0.9766 |
| subclass_accuracy_conditional | 0.8011 |
| window_ece | 0.0148 |
| fast_path_rate | 0.1005 |

## Analyst insights & findings

- **Doc_type routing:** best `insurance_claim` 1.000 acc (n=114); weakest `corporate_record` 0.894 acc (n=47).
- **Subclass bottleneck:** lowest test macro-F1 head `merger_agreement` at 0.1774 (#112 floors: contract ≥0.20, correspondence ≥0.25).
- **Window cohort (#104):** multi-window doc_type acc 0.9074 vs single-window 0.9795 (n=54/1269).
- **Selective-risk:** global budget_met with deployment threshold 0.89 at coverage ~0.911.
- **Fast-path rate:** 0.1005 of docs would route on classifier gate.

## Per-document errors (288 of 1323 docs)

_Full per-doc table omitted (n=1323 > 500); see `per_doc` in eval JSON._

| filename | gt doc_type | pred | dt ok | sc ok | n_win | fast_path |
| --- | --- | --- | --- | --- | ---: | --- |
| `0000950123-11-000444_y86334a4exv24w3.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0000950134-08-011905_f38510a4exv3w2w2.htm` | corporate_record | corporate_record | True | False | 2 | False |
| `0001047469-15-005538_a2225135zex-10_19.htm` | contract | corporate_record | False | False | 2 | False |
| `0001104659-07-060356_a07-15464_2ex3d1.htm` | corporate_record | corporate_record | True | False | 2 | False |
| `0001144204-13-006917_v333161_ex10-32.htm` | contract | corporate_record | False | False | 1 | False |
| `0001193125-09-259347_dex21.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001193125-10-059057_dex211.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001193125-11-064249_dex31.htm` | corporate_record | corporate_record | True | False | 3 | False |
| `0001193125-11-095556_dex211.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001193125-24-221320_d829976dex21.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001193125-25-240868_d28749dex31.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001213900-15-009308_fs12015a3ex21i_shimmick.htm` | corporate_record | correspondence | False | False | 1 | False |
| `0001437749-18-018936_ex_126721.htm` | corporate_record | corporate_record | True | False | 1 | False |
| `0001493152-16-006927_ex10-23.htm` | contract | contract | True | False | 1 | False |
| `0001567619-13-000087_s000086x3_ex10-3.htm` | contract | corporate_record | False | False | 1 | False |
| `AgapeAtpCorp_20191202_10-KA_EX-10.1_11911128_EX-10.1_Supply Agreement.pdf` | contract | contract | True | False | 1 | False |
| `ArcGroupInc_20171211_8-K_EX-10.1_10976103_EX-10.1_Sponsorship Agreement.pdf` | contract | contract | True | False | 1 | False |
| `ArcaUsTreasuryFund_20200207_N-2_EX-99.K5_11971930_EX-99.K5_Development Agreement.pdf` | contract | contract | True | False | 1 | False |
| `AzulSa_20170303_F-1A_EX-10.3_9943903_EX-10.3_Maintenance Agreement2.pdf` | contract | contract | True | False | 1 | False |
| `BICYCLETHERAPEUTICSPLC_03_10_2020-EX-10.11-SERVICE AGREEMENT.PDF` | contract | contract | True | False | 2 | False |
| `BORROWMONEYCOM,INC_06_11_2020-EX-10.1-JOINT VENTURE AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `COOLTECHNOLOGIES,INC_10_25_2017-EX-10.71-Strategic Alliance Agreement.PDF` | contract | contract | True | False | 1 | False |
| `CURAEGISTECHNOLOGIES,INC_05_26_2010-EX-1-CORPORATE SPONSORSHIP AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `CardlyticsInc_20180112_S-1_EX-10.16_11002987_EX-10.16_Maintenance Agreement2.pdf` | contract | contract | True | False | 1 | False |
| `ClickstreamCorp_20200330_1-A_EX1A-6 MAT CTRCT_12089935_EX1A-6 MAT CTRCT_Development Agreement.pdf` | contract | contract | True | False | 1 | False |
| `Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT.PDF` | contract | contract | True | False | 2 | False |
| `CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf` | contract | contract | True | False | 1 | False |
| `FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_AMENDMENT.pdf` | contract | contract | True | False | 1 | False |
| `FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_POWEROF.pdf` | contract | contract | True | False | 1 | False |
| `FIDELITYNATIONALINFORMATIONSERVICES,INC_08_05_2009-EX-10.3-INTELLECTUAL PROPERTY AGREEMENT.PDF` | contract | contract | True | False | 4 | False |
| `GWG HOLDINGS, INC. - ORDERLY MARKETING AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `INKTOMICORP_06_08_1998-EX-10.14-SOFTWARE HOSTING AGREEMENT.PDF` | contract | contract | True | False | 2 | False |
| `IntegrityMediaInc_20010329_10-K405_EX-10.17_2373875_EX-10.17_Co-Branding Agreement.pdf` | contract | contract | True | False | 1 | False |
| `JINGWEIINTERNATIONALLTD_10_04_2007-EX-10.7-INTELLECTUAL PROPERTY AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `KUBIENT,INC_07_02_2020-EX-10.14-MASTER SERVICES AGREEMENT_Part1.pdf` | contract | contract | True | False | 1 | False |
| `KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing Agreement.pdf` | contract | contract | True | False | 2 | False |
| `LEJUHOLDINGSLTD_03_12_2014-EX-10.34-INTERNET CHANNEL COOPERATION AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `LUCIDINC_04_15_2011-EX-10.9-DISTRIBUTOR AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `LinkPlusCorp_20050802_8-K_EX-10_3240252_EX-10_Affiliate Agreement.pdf` | contract | contract | True | False | 1 | False |
| `LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement.pdf` | contract | contract | True | False | 1 | False |
| `OAKTREECAPITALGROUP,LLC_03_02_2020-EX-10.8-Services Agreement.PDF` | contract | contract | True | False | 1 | False |
| `PACIFICSYSTEMSCONTROLTECHNOLOGYINC_08_24_2000-EX-10.53-SPONSORSHIP AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT .PDF` | contract | contract | True | False | 4 | False |
| `PelicanDeliversInc_20200211_S-1_EX-10.3_11975895_EX-10.3_Development Agreement1.pdf` | contract | contract | True | False | 1 | False |
| `RandWorldwideInc_20010402_8-KA_EX-10.2_2102464_EX-10.2_Co-Branding Agreement.pdf` | contract | contract | True | False | 2 | False |
| `SLOVAKWIRELESSFINANCECOBV_03_28_2001-EX-4.(B)(II).3-Maintenance and support contract for SICAP(R) modules.PDF` | contract | contract | True | False | 1 | False |
| `SmartRxSystemsInc_20180914_1-A_EX1A-6 MAT CTRCT_11351705_EX1A-6 MAT CTRCT_Distributor Agreement.pdf` | contract | contract | True | False | 2 | False |
| `URSCORPNEW_03_17_2014-EX-99-COOPERATION AGREEMENT.PDF` | contract | corporate_record | False | False | 1 | False |
| `WPPPLC_04_30_2020-EX-4.28-SERVICE AGREEMENT.PDF` | contract | contract | True | False | 2 | False |
| `WestPharmaceuticalServicesInc_20200116_8-K_EX-10.1_11947529_EX-10.1_Supply Agreement.pdf` | contract | contract | True | False | 3 | False |
| `XLITECHNOLOGIES,INC_12_02_2015-EX-10.02-STRATEGIC ALLIANCE AGREEMENT.PDF` | contract | contract | True | False | 1 | False |
| `ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor Agreement.pdf` | contract | contract | True | False | 6 | False |
| `cash-m/all_documents/482.` | correspondence | correspondence | True | False | 1 | False |
| `contract_105_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 13 | False |
| `contract_106_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 7 | False |
| `contract_108_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 12 | False |
| `contract_109_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 11 | False |
| `contract_123_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 11 | False |
| `contract_135_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 10 | False |
| `contract_16_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 8 | False |
| `contract_18_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 9 | False |
| `contract_48_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 9 | False |
| `contract_74_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 9 | False |
| `contract_91_merger_agreement.txt` | merger_agreement | merger_agreement | True | False | 11 | False |
| `cuilla-m/inbox/79.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/10778.` | correspondence | contract | False | False | 1 | False |
| `dasovich-j/all_documents/28272.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/3687.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/deleted_items/1254.` | correspondence | correspondence | True | False | 1 | False |
| `ermis-f/all_documents/303.` | correspondence | correspondence | True | False | 1 | False |
| `fischer-m/all_documents/71.` | correspondence | correspondence | True | False | 1 | False |
| `fossum-d/all_documents/643.` | correspondence | correspondence | True | False | 1 | False |
| `giron-d/_sent_mail/29.` | correspondence | correspondence | True | False | 1 | False |
| `grigsby-m/deleted_items/244.` | correspondence | corporate_record | False | False | 1 | False |
| `hayslett-r/all_documents/217.` | correspondence | correspondence | True | False | 1 | False |
| `hayslett-r/inbox/216.` | correspondence | correspondence | True | False | 1 | False |
| `horton-s/all_documents/93.` | correspondence | correspondence | True | False | 1 | False |
| `jones-t/all_documents/3795.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/deleted_items/82.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/inbox/653.` | correspondence | correspondence | True | False | 1 | False |
| `kean-s/all_documents/1802.` | correspondence | correspondence | True | False | 1 | False |
| `kitchen-l/_americas/mrha/ooc/175.` | correspondence | correspondence | True | False | 1 | False |
| `lokay-m/hea_nesa/75.` | correspondence | correspondence | True | False | 1 | False |
| `love-p/deleted_items/127.` | correspondence | correspondence | True | False | 1 | False |
| `mann-k/all_documents/1951.` | correspondence | correspondence | True | False | 1 | False |
| `mccarty-d/inbox/91.` | correspondence | correspondence | True | False | 1 | False |
| `nemec-g/all_documents/6413.` | correspondence | correspondence | True | False | 1 | False |
| `sager-e/all_documents/1519.` | correspondence | correspondence | True | False | 1 | False |
| `sager-e/all_documents/1942.` | correspondence | correspondence | True | False | 1 | False |
| `sager-e/all_documents/254.` | correspondence | correspondence | True | False | 1 | False |
| `sanders-r/deleted_items/542.` | correspondence | contract | False | False | 2 | False |
| `shackleton-s/all_documents/2755.` | correspondence | contract | False | False | 1 | False |
| `shackleton-s/deleted_items/589.` | correspondence | correspondence | True | False | 1 | False |
| `skilling-j/all_documents/1522.` | correspondence | correspondence | True | False | 1 | False |
| `allen-p/inbox/5.` | correspondence | correspondence | True | False | 1 | False |
| `arnold-j/deleted_items/266.` | correspondence | contract | False | False | 1 | False |
| `arnold-j/deleted_items/560.` | correspondence | correspondence | True | False | 1 | False |
| `arnold-j/deleted_items/574.` | correspondence | correspondence | True | False | 1 | False |
| `arnold-j/inbox/85.` | correspondence | correspondence | True | False | 1 | False |
| `bass-e/all_documents/37.` | correspondence | correspondence | True | False | 1 | False |
| `beck-s/_sent_mail/668.` | correspondence | correspondence | True | False | 1 | False |
| `beck-s/_sent_mail/91.` | correspondence | correspondence | True | False | 1 | False |
| `blair-l/customer___virginia_power_dominion/10.` | correspondence | correspondence | True | False | 1 | False |
| `buy-r/all_documents/32.` | correspondence | correspondence | True | False | 1 | False |
| `campbell-l/all_documents/470.` | correspondence | correspondence | True | False | 1 | False |
| `campbell-l/inbox/284.` | correspondence | correspondence | True | False | 1 | False |
| `campbell-l/inbox/745.` | correspondence | correspondence | True | False | 1 | False |
| `cash-m/deleted_items/72.` | correspondence | correspondence | True | False | 1 | False |
| `cash-m/sent_items/208.` | correspondence | correspondence | True | False | 1 | False |
| `corman-s/deleted_items/52.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/11275.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/11730.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/13490.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/1558.` | correspondence | corporate_record | False | False | 1 | False |
| `dasovich-j/all_documents/1714.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/18.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/27709.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/29358.` | correspondence | insurance_claim | False | False | 1 | False |
| `dasovich-j/all_documents/3567.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/3974.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/4456.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/675.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/8096.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/868.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/9476.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/9763.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/all_documents/9865.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/deleted_items/1528.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/deleted_items/1908.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/deleted_items/2005.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/deleted_items/572.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/deleted_items/95.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/sent_items/139.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/sent_items/3.` | correspondence | correspondence | True | False | 1 | False |
| `dasovich-j/sent_items/368.` | correspondence | correspondence | True | False | 1 | False |
| `davis-d/all_documents/310.` | correspondence | correspondence | True | False | 1 | False |
| `delainey-d/_sent_mail/847.` | correspondence | correspondence | True | False | 1 | False |
| `derrick-j/all_documents/53.` | correspondence | correspondence | True | False | 1 | False |
| `derrick-j/deleted_items/69.` | correspondence | correspondence | True | False | 1 | False |
| `donoho-l/inbox/junk_file/338.` | correspondence | correspondence | True | False | 1 | False |
| `dorland-c/sent_items/511.` | correspondence | corporate_record | False | False | 1 | False |
| `ermis-f/inbox/60.` | correspondence | correspondence | True | False | 1 | False |
| `farmer-d/all_documents/2150.` | correspondence | correspondence | True | False | 1 | False |
| `farmer-d/all_documents/608.` | correspondence | correspondence | True | False | 1 | False |
| `fossum-d/_sent_mail/108.` | correspondence | correspondence | True | False | 1 | False |
| `gang-l/deleted_items/148.` | correspondence | contract | False | False | 1 | False |
| `germany-c/bankrupt/transport/43.` | correspondence | correspondence | True | False | 1 | False |
| `germany-c/bankrupt/transport/dominion/29.` | correspondence | correspondence | True | False | 1 | False |
| `giron-d/inbox/180.` | correspondence | correspondence | True | False | 1 | False |
| `giron-d/inbox/38.` | correspondence | correspondence | True | False | 1 | False |
| `guzman-m/all_documents/399.` | correspondence | correspondence | True | False | 1 | False |
| `haedicke-m/all_documents/856.` | correspondence | correspondence | True | False | 1 | False |
| `haedicke-m/all_documents/859.` | correspondence | correspondence | True | False | 1 | False |
| `haedicke-m/inbox/499.` | correspondence | correspondence | True | False | 1 | False |
| `hain-m/all_documents/233.` | correspondence | correspondence | True | False | 1 | False |
| `heard-m/inbox/master_netting/214.` | correspondence | correspondence | True | False | 1 | False |
| `heard-m/sent_items/295.` | correspondence | correspondence | True | False | 1 | False |
| `hendrickson-s/all_documents/24.` | correspondence | correspondence | True | False | 1 | False |
| `holst-k/deleted_items/47.` | correspondence | correspondence | True | False | 1 | False |
| `holst-k/inbox/81.` | correspondence | correspondence | True | False | 1 | False |
| `hyatt-k/inbox/21.` | correspondence | correspondence | True | False | 1 | False |
| `hyatt-k/sent_items/91.` | correspondence | correspondence | True | False | 1 | False |
| `hyvl-d/all_documents/556.` | correspondence | correspondence | True | False | 1 | False |
| `hyvl-d/all_documents/932.` | correspondence | correspondence | True | False | 1 | False |
| `jones-t/all_documents/10949.` | correspondence | contract | False | False | 1 | False |
| `jones-t/all_documents/12028.` | correspondence | correspondence | True | False | 1 | False |
| `jones-t/all_documents/1894.` | correspondence | correspondence | True | False | 1 | False |
| `jones-t/all_documents/2599.` | correspondence | correspondence | True | False | 1 | False |
| `jones-t/all_documents/3515.` | correspondence | correspondence | True | False | 1 | False |
| `jones-t/inbox/562.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/_sent_mail/1648.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/_sent_mail/4542.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/_sent_mail/4697.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/_sent_mail/773.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/all_documents/2922.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/all_documents/5650.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/all_documents/9931.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/deleted_items/1282.` | correspondence | correspondence | True | False | 1 | False |
| `kaminski-v/sent_items/961.` | correspondence | correspondence | True | False | 1 | False |
| `kean-s/all_documents/1191.` | correspondence | correspondence | True | False | 1 | False |
| `kean-s/all_documents/207.` | correspondence | correspondence | True | False | 1 | False |
| `kean-s/all_documents/466.` | correspondence | correspondence | True | False | 1 | False |
| `kean-s/all_documents/4945.` | correspondence | correspondence | True | False | 1 | False |
| `kean-s/all_documents/6019.` | correspondence | correspondence | True | False | 1 | False |
| `kean-s/all_documents/6528.` | correspondence | correspondence | True | False | 1 | False |
| `kitchen-l/_americas/east_power/124.` | correspondence | correspondence | True | False | 1 | False |
| `kitchen-l/_americas/hr/79.` | correspondence | correspondence | True | False | 1 | False |
| `kitchen-l/_americas/mrha/ooc/249.` | correspondence | correspondence | True | False | 1 | False |
| `kitchen-l/_americas/regulatory/29.` | correspondence | correspondence | True | False | 1 | False |
| `kitchen-l/_americas/sec/19.` | correspondence | contract | False | False | 3 | False |
| `kitchen-l/_americas/south_america/5.` | correspondence | contract | False | False | 1 | False |
| `kitchen-l/_americas/turbines/trading/5.` | correspondence | correspondence | True | False | 1 | False |
| `lay-k/inbox/1785.` | correspondence | correspondence | True | False | 1 | False |
| `lay-k/inbox/305.` | correspondence | correspondence | True | False | 1 | False |
| `lay-k/inbox/451.` | correspondence | correspondence | True | False | 1 | False |
| `lay-k/inbox/641.` | correspondence | correspondence | True | False | 1 | False |
| `lewis-a/deleted_items/810.` | correspondence | contract | False | False | 1 | False |
| `lewis-a/inbox/128.` | correspondence | correspondence | True | False | 1 | False |
| `lewis-a/inbox/146.` | correspondence | correspondence | True | False | 1 | False |
| `lokay-m/all_documents/2079.` | correspondence | correspondence | True | False | 1 | False |
| `lokay-m/articles/336.` | correspondence | correspondence | True | False | 1 | False |
| `lokay-m/articles/351.` | correspondence | correspondence | True | False | 1 | False |
| `lokey-t/inbox/202.` | correspondence | corporate_record | False | False | 1 | False |
| `lokey-t/inbox/643.` | correspondence | correspondence | True | False | 1 | False |
| `lokey-t/inbox/880.` | correspondence | correspondence | True | False | 1 | False |
| `love-p/deleted_items/416.` | correspondence | correspondence | True | False | 1 | False |
| `mann-k/_sent_mail/2980.` | correspondence | correspondence | True | False | 1 | False |
| `mann-k/_sent_mail/3801.` | correspondence | correspondence | True | False | 1 | False |
| `mann-k/_sent_mail/3904.` | correspondence | correspondence | True | False | 1 | False |
| `mann-k/all_documents/546.` | correspondence | correspondence | True | False | 1 | False |
| `mann-k/all_documents/6055.` | correspondence | correspondence | True | False | 1 | False |
| `mann-k/all_documents/6156.` | correspondence | correspondence | True | False | 1 | False |
| `mann-k/all_documents/6561.` | correspondence | correspondence | True | False | 1 | False |
| `martin-t/inbox/106.` | correspondence | contract | False | False | 5 | False |
| `martin-t/inbox/40.` | correspondence | contract | False | False | 6 | False |
| `mccarty-d/inbox/alaska_gas/4.` | correspondence | correspondence | True | False | 1 | False |
| `mcconnell-m/_sent_mail/551.` | correspondence | corporate_record | False | False | 1 | False |
| `mckay-j/sent_items/85.` | correspondence | correspondence | True | False | 1 | False |
| `mims-thurston-p/deleted_items/556.` | correspondence | correspondence | True | False | 1 | False |
| `mims-thurston-p/inbox/126.` | correspondence | correspondence | True | False | 1 | False |
| `mims-thurston-p/inbox/225.` | correspondence | correspondence | True | False | 1 | False |
| `motley-m/inbox/ene_portland/administrative/48.` | correspondence | corporate_record | False | False | 1 | False |
| `nemec-g/all_documents/1437.` | correspondence | correspondence | True | False | 1 | False |
| `nemec-g/all_documents/5487.` | correspondence | correspondence | True | False | 1 | False |
| `nemec-g/inbox/1433.` | correspondence | correspondence | True | False | 1 | False |
| `nemec-g/inbox/1574.` | correspondence | correspondence | True | False | 1 | False |
| `nemec-g/inbox/809.` | correspondence | correspondence | True | False | 1 | False |
| `perlingiere-d/all_documents/130.` | correspondence | correspondence | True | False | 1 | False |
| `perlingiere-d/sent_items/20.` | correspondence | correspondence | True | False | 1 | False |
| `rapp-b/inbox/206.` | correspondence | correspondence | True | False | 1 | False |
| `rapp-b/inbox/332.` | correspondence | correspondence | True | False | 1 | False |
| `rodrique-r/_sent_mail/423.` | correspondence | correspondence | True | False | 1 | False |
| `rogers-b/deleted_items/171.` | correspondence | correspondence | True | False | 1 | False |
| `sager-e/all_documents/1821.` | correspondence | correspondence | True | False | 1 | False |
| `sager-e/inbox/1049.` | correspondence | correspondence | True | False | 1 | False |
| `sager-e/inbox/117.` | correspondence | correspondence | True | False | 1 | False |
| `salisbury-h/inbox/1159.` | correspondence | correspondence | True | False | 1 | False |
| `sanders-r/all_documents/606.` | correspondence | correspondence | True | False | 1 | False |
| `sanders-r/all_documents/7556.` | correspondence | correspondence | True | False | 1 | False |
| `sanders-r/senator_dunn_inv_/51.` | correspondence | corporate_record | False | False | 1 | False |
| `scholtes-d/diana_scholtes/saved_sent/4.` | correspondence | correspondence | True | False | 1 | False |
| `scholtes-d/stf/current_issues/24.` | correspondence | correspondence | True | False | 1 | False |
| `scott-s/inbox/157.` | correspondence | correspondence | True | False | 1 | False |
| `scott-s/inbox/393.` | correspondence | correspondence | True | False | 1 | False |
| `scott-s/sent_items/555.` | correspondence | correspondence | True | False | 1 | False |
| `shackleton-s/all_documents/11002.` | correspondence | correspondence | True | False | 1 | False |
| `shackleton-s/all_documents/5218.` | correspondence | correspondence | True | False | 1 | False |
| `shackleton-s/all_documents/9286.` | correspondence | correspondence | True | False | 1 | False |
| `shackleton-s/inbox/319.` | correspondence | correspondence | True | False | 1 | False |
| `shackleton-s/inbox/374.` | correspondence | correspondence | True | False | 1 | False |
| `shackleton-s/inbox/396.` | correspondence | contract | False | False | 1 | False |
| `shackleton-s/inbox/846.` | correspondence | correspondence | True | False | 1 | False |
| `shackleton-s/inbox/982.` | correspondence | correspondence | True | False | 1 | False |
| `shankman-j/all_documents/637.` | correspondence | correspondence | True | False | 1 | False |
| `shankman-j/deleted_items/229.` | correspondence | correspondence | True | False | 1 | False |
| `shankman-j/deleted_items/280.` | correspondence | correspondence | True | False | 1 | False |
| `shapiro-r/deleted_items/1253.` | correspondence | correspondence | True | False | 1 | False |
| `shapiro-r/deleted_items/695.` | correspondence | correspondence | True | False | 1 | False |
| `shively-h/deleted_items/399.` | correspondence | correspondence | True | False | 1 | False |
| `skilling-j/deleted_items/354.` | correspondence | correspondence | True | False | 1 | False |
| `skilling-j/inbox/1073.` | correspondence | correspondence | True | False | 1 | False |
| `skilling-j/inbox/1151.` | correspondence | correspondence | True | False | 1 | False |
| `staab-t/deleted_items/177.` | correspondence | correspondence | True | False | 1 | False |
| `steffes-j/deleted_items/246.` | correspondence | correspondence | True | False | 1 | False |
| `steffes-j/deleted_items/55.` | correspondence | correspondence | True | False | 1 | False |
| `stepenovitch-j/inbox/138.` | correspondence | correspondence | True | False | 1 | False |
| `stokley-c/chris_stokley/iso/mrkt_info/105.` | correspondence | correspondence | True | False | 1 | False |
| `swerzbin-m/sent_items/33.` | correspondence | correspondence | True | False | 1 | False |
| `symes-k/_sent_mail/493.` | correspondence | correspondence | True | False | 1 | False |
| `symes-k/_sent_mail/711.` | correspondence | correspondence | True | False | 1 | False |
| `symes-k/all_documents/3070.` | correspondence | correspondence | True | False | 1 | False |
| `taylor-m/all_documents/3058.` | correspondence | insurance_claim | False | False | 1 | False |
| `taylor-m/all_documents/3430.` | correspondence | correspondence | True | False | 1 | False |
| `taylor-m/all_documents/7312.` | correspondence | correspondence | True | False | 1 | False |
| `taylor-m/all_documents/7361.` | correspondence | correspondence | True | False | 1 | False |
| `taylor-m/all_documents/7522.` | correspondence | correspondence | True | False | 1 | False |
| `taylor-m/all_documents/78.` | correspondence | correspondence | True | False | 1 | False |
| `taylor-m/archive/2001_11/58.` | correspondence | correspondence | True | False | 1 | False |
| `taylor-m/inbox/michelle_schultz/1.` | correspondence | correspondence | True | False | 1 | False |
| `thomas-p/deleted_items/209.` | correspondence | correspondence | True | False | 1 | False |
| `thomas-p/deleted_items/241.` | correspondence | correspondence | True | False | 1 | False |
| `ward-k/sent_items/109.` | correspondence | correspondence | True | False | 1 | False |
| `watson-k/deleted_items/119.` | correspondence | correspondence | True | False | 1 | False |
| `whalley-g/all_documents/150.` | correspondence | corporate_record | False | False | 1 | False |
| `whalley-l/_sent_mail/229.` | correspondence | correspondence | True | False | 1 | False |
| `white-s/all_documents/158.` | correspondence | correspondence | True | False | 1 | False |
| `whitt-m/sent_items/59.` | correspondence | correspondence | True | False | 1 | False |
| `ybarbo-p/sent_items/41.` | correspondence | correspondence | True | False | 1 | False |

## Reproduce

```bash
uv run python training/eval/eval_modernbert.py --subset heldout-plus \
  --checkpoint /checkpoints/runs/20261014-113614 --sample 0 --json \
  > reports/eval_m9c-20261014-113614-heldout-plus.json
uv run python training/eval/write_eval_report.py heldout-plus --run-tag m9c-20261014-113614 \
  --eval-json reports/eval_m9c-20261014-113614-heldout-plus.json \
  --baseline-json reports/eval_m9c-20261014-113614.json
```

## Artifacts

| path | role |
| --- | --- |
| `reports/eval_m9c-20261014-113614-heldout-plus.json` | canonical heldout-plus eval |
| `reports/TEST-EVAL/TEST-EVAL-REPORT-heldout-plus-m9c-20261014-113614.md` | this report |
