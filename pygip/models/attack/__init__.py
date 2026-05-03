# from .AdvMEA import AdvMEA
# from .CEGA import CEGA
# from .DataFreeMEA import (
#     DFEATypeI,
#     DFEATypeII,
#     DFEATypeIII
# )
# from .mea.MEA import (
#     ModelExtractionAttack0,
#     ModelExtractionAttack1,
#     ModelExtractionAttack2,
#     ModelExtractionAttack3,
#     ModelExtractionAttack4,
#     ModelExtractionAttack5
# )
# from .Realistic import RealisticAttack

# __all__ = [
#     'AdvMEA',
#     'CEGA',
#     'RealisticAttack',
#     'DFEATypeI',
#     'DFEATypeII',
#     'DFEATypeIII',
#     'ModelExtractionAttack0',
#     'ModelExtractionAttack1',
#     'ModelExtractionAttack2',
#     'ModelExtractionAttack3',
#     'ModelExtractionAttack4',
#     'ModelExtractionAttack5',
# ]

# Attack module for link prediction
# Original node classification imports are commented out to avoid DGL dependency

# Link prediction attacks - these don't require DGL
try:
    from .linkpred_attacks import (
        ModelExtractionAttack0,
        ModelExtractionAttack1,
        ModelExtractionAttack2,
        ModelExtractionAttack3,
        ModelExtractionAttack4,
        ModelExtractionAttack5,
        MEALinkPred,
        AdvMEALinkPred,
        CEGALinkPred,
        DFEATypeILinkPred,
        DFEATypeIILinkPred,
        DFEATypeIIILinkPred
    )
    
    __all__ = [
        'ModelExtractionAttack0',
        'ModelExtractionAttack1',
        'ModelExtractionAttack2',
        'ModelExtractionAttack3',
        'ModelExtractionAttack4',
        'ModelExtractionAttack5',
        'MEALinkPred',
        'AdvMEALinkPred',
        'CEGALinkPred',
        'DFEATypeILinkPred',
        'DFEATypeIILinkPred',
        'DFEATypeIIILinkPred'
    ]
except ImportError as e:
    print(f"Warning: Could not import link prediction attacks: {e}")
    __all__ = []
