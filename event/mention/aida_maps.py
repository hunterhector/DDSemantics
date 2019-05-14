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

token_direct_map = {
    'seize': 'ldcOnt:Transaction.Transaction.TransferControl',
    'casualty': 'ldcOnt:Life.Die.DeathCausedByViolentEvents',
    'capture': 'ldcOnt:Transaction.Transaction.TransferControl',
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
    'Taking': 'ldcOnt:Transaction.Transaction.TransferControl',
    'Meet': 'ldcOnt:Contact.Collaborate.Meet',
}

srl_ldc_arg_map = {
    'ldcOnt:ArtifactExistence.DamageDestroy': {
        'ARG2': 'Instrument',
        'ARG0': 'DamagerDestroyer',
        'ARG1': 'Artifact',
        'Victim': 'Artifact',
    },
    'ldcOnt:Transaction.Transaction.TransferControl': {
        'Conqueror': 'Recipient',
        'Theme': 'TerritoryOrFacility',
        'ARG0': 'Recipient',
        'ARG1': 'TerritoryOrFacility',
        'ARG2': 'Giver',
        'Donor': 'Giver',
        'Recipient': 'Recipient',
        'Controlling_entity': 'Recipient',
        'Dependent_entity': 'TerritoryOrFacility',
        'Dependent_situation': 'TerritoryOrFacility',
        'Dependent_variable': 'TerritoryOrFacility',
        'Agent': 'Recipient',
        'Source': 'Giver',
        'Entity': 'TerritoryOrFacility',
    },
    'ldcOnt:Transaction.TransferOwnership': {
        'Goods': 'Artifact',
        'Buyer': 'Recipient',
        'Dependent': 'Recipient',
        'ARG1': 'Artifact',
    },
    'ldcOnt:Conflict.Attack': {
        'Victim': 'Target',
        'Assailant': 'Attacker',
        'Weapon': 'Instrument',
        'Sides': 'Attacker',
        'Side_1': 'Attacker',
        'Side_2': 'Attacker',
        'Agent': 'Attacker',
        # 'Projectile': 'Instrument',
    },
    'ldcOnt:Movement.TransportPerson': {
        'Effect': 'Passenger',
        'Theme': 'Passenger',
        'Sleeper': 'Passenger',
        'Cause': 'Transporter',
        'Driver': 'Transporter',
        'ARG0': 'Transporter',
        'Agent': 'Transporter',
        'Goal': 'Destination',
        'ARG2': 'Destination',
        'ARGM-DIR': 'Destination',
        'Source': 'Origin',
        'ARG4': 'Vehicle',
        'Self_mover': 'Vehicle',
    },
    'ldcOnt:Movement.TransportArtifact': {
        'Effect': 'Artifact',
        'Sleeper': 'Artifact',
        'Theme': 'Artifact',
        'Cause': 'Transporter',
        'ARG0': 'Transporter',
        'Agent': 'Transporter',
        'Driver': 'Transporter',
        'Goal': 'Destination',
        'ARG2': 'Destination',
        'ARGM-DIR': 'Destination',
        'Source': 'Origin',
        'ARG4': 'Vehicle',
        'Self_mover': 'Vehicle',
    },
    'ldcOnt:Conflict.Attack.Invade': {
        'Invader': 'Attacker',
        'Land': 'Place',
        'Source': 'Attacker',
        'Means': 'Instrument',
        'ARG2': 'Instrument',
    },
    'ldcOnt:Conflict.Yield.Surrender': {
        'Recipient': 'Recipient',
        'Theme': 'Place',
        'Surrenderer': 'Surrenderer',
    },
    'ldcOnt:Justice.ArrestJailDetain.ArrestJailDetain': {
        'Conqueror': 'Detainee',
        'Theme': 'Jailer',
    },
    'ldcOnt:Life.Die.DeathCausedByViolentEvents': {
        'Entity': 'Victim',
    },
    'ldcOnt:Movement.TransportArtifact.SendSupplyExport': {
        'Effect': 'Artifact',
        'ARG2': 'Transporter',
        'ARGM-LOC': 'Destination',
        'Place': 'Destination',
    },
    'ldcOnt:Personnel.StartPosition': {
        'Employee': 'Employee',
        'Employer': 'PlaceOfEmployment',
    },
    'ldcOnt:Personnel.StartPosition.Hiring': {
        'Employee': 'Employee',
        'Employer': 'PlaceOfEmployment',
    },
    'ldcOnt:Contact.Collaborate': {
        'Speaker': 'Communicator',
        'Party_1': 'Participant',
        'Party_2': 'Participant',
        'ARG1': 'Participant',
    },
    'ldcOnt:Contact.Negotiate': {
        'Party_1': 'Participant',
        'Party_2': 'Participant',
        'ARG1': 'Participant',
    },
    'ldcOnt:Contact.CommandOrder': {
        'Speaker': 'Communicator',
    },
    'ldcOnt:Contact.Discussion': {
        'Speaker': 'Participant',
        'ARG0': 'Participant',
        'ARG2': 'Participant',
    },
    'ldcOnt:Contact.MediaStatement': {
        'Speaker': 'Communicator',
    },
    'ldcOnt:Transaction.TransferMoney.BorrowLend': {
        'ARG2': 'Recipient',
    },
    'ldcOnt:Conflict.Yield.Retreat': {
        'Self_mover': 'Retreater',
        'Goal': 'Destination',
        'Source': 'Origin',
        'Intended_goal': 'Destination',
    },
    'ldcOnt:Movement.TransportPerson.BringCarryUnload': {
        'Agent': 'Transporter',
    },
}

arg_direct_map = {
    'missile': ('ldcOnt:Conflict.Attack.AirstrikeMissileStrike',
                'ldcOnt:Conflict.Attack.AirstrikeMissileStrike_Instrument'),
}
