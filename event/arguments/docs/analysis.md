1. analysis through event coref:
    - however, how do we get the correct argument for this first plan?
        - to abandon plans to raise temporary working capital through debt issued from an agency that would n't be counted on the federal budget .
        - another "plan" will take the same arg with this plan.
    - One way is to do carry over via supports.
    - Another way is to use a nombank parser.
    - Semafor did not produce that, we can get it from the "support" (abandon). but that's quite tricky.
    - At the very least, we should get that if the gold candidate is given.
    - Without the gold candidate, we are filling two arguments at the same time
        - fill the first with an close entity
        - fill the second one via coref
1. analysis through the surrounding events:
    - But higher interest rates paid on off-budget debt could add billions to the bailout costs
    - for the costs, we can analyze "bailout" to get the actual entity
        - such as "the new S&L bailout agency"
    - similar analysis should be done on supports
        - The borrowing to raise these funds (raise is the support for funds)
    - Some answers can be directly found in supports 
        - "the RTC may have to slow -LCB- S&L sales -RCB- or dump acquired assets through fire sales"
1. analysis through multiple events:
    - is it possible to read and analyze complex event chains?
        - Du Pont Co. , Hewlett-Packard Co. and Los Alamos National Laboratory 
        said they signed a three-year , $ 11 million agreement to collaborate 
        on superconductor research 
        - Joint-research programs have proliferated as U.S. companies seek to 
        spread the risks and costs of commercializing new superconductors and 
        to meet the challenges posed by foreign consortia
        - signed-subj -> agreement-obj -> collaborate-prep
        - cost-of
    - first, we need to capture all the structures
    - second, we need to have reasoning mechanisms
1. The usage of the Argument Index as a feature
    - The feature should rank the negative argument higher than the positive
    - When we are replacing the argument at the same slot, the argument index 
    should make few contribution:
        - Since the event representation is constructed with fixed slots, so
          the information of each slot should be already fixed.
        - The multi-layer network may cause some interaction of this feature.
        - In the frame mode, the extra feature can be the FE vector.
        - In the frame mode, the event representation is done auto-regressively.