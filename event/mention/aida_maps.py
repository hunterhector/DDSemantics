kbp_type_split_map = {
    'Transferownership': ['transfer', 'ownership'],
    'Transfermoney': ['transfer', 'money'],
    'Transportperson': ['transport', 'person'],
    'TransportArtifact': ['transport'],
    'Startposition': ['employ'],
    'Endposition': ['resign'],
    'Arrestjail': ['arrest', 'jail'],
    'Chargeindict': ['charge', 'indict'],
    'Trialhearing': ['trial', 'hearing'],
    'ReleaseParole': ['release', 'parole'],
    'Declarebankruptcy': ['bankruptcy'],
    'StartOrg': ['start'],
    'EndOrg': ['end'],
    'MergeOrg': ['merge'],
    'BeBorn': ['born'],
}

ldc_ontology_skips = {
    'artifact',
    'in',
    'person',
    'start',
    'end',
    'life',
    'Existence',
}

onto_token_nom_map = {
    'correspondence': 'correspond',
    'prevarication': 'prevaricate',
    'gathering': 'gather',
    'agreements': 'agreement',
    'degradation': 'degrade',
    'movement': 'move',
    'hiring': 'hire',
    'injury': 'injure',
    'stabbing': 'stab',
}

frame_direct_map = {
    'Arriving': 'ldcOnt:Movement.TransportPerson',
    'Employing': 'ldcOnt:Personnel.StartPosition',
    'Shoot_projectiles': 'ldcOnt:Conflict.Attack.AirstrikeMissileStrike',
    'Communication_response': 'ldcOnt:Contact.Discussion',
    'Chatting': 'ldcOnt:Contact.Discussion',
    'Hostile_encounter': 'ldcOnt:Conflict.Attack',
}

srl_ldc_arg_map = {
    'ldcOnt:ArtifactExistence.DamageDestroy': {
        'ARG2': 'Instrument',
        'ARG0': 'DamagerDestroyer',
        'ARG1': 'Artifact',
    }
}

arg_direct_map = {
    'missile': ('ldcOnt:Conflict.Attack.AirstrikeMissileStrike',
               'ldcOnt:Conflict.Attack.AirstrikeMissileStrike_Instrument'),

}

# Everyone can have a place now. We will double check with the ontology to see
# if this role does not exist.
for etype, e_args in srl_ldc_arg_map.items():
    e_args['ARGM-LOC'] = f'{etype}_Place'
