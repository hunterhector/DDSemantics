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
    'Life_Die': ['die'],
}

ldc_ontology_skips = {
    'artifact',
    'in',
    'by',
    'person',
    'start',
    'end',
    'life',
    'existence',
    'event',
    'self',
    'people',
}

kbp_direct_map = {
    'Life_die': 'ldcOnt:Life.Die',
    'Contact_Contact': 'ldcOnt:Contact.CommitmentPromiseExpressIntent',
    'Contact_Meet': 'ldcOnt:Contact.CommitmentPromiseExpressIntent.Meet',
    'Contact_Broadcast': 'ldcOnt:Contact.MediaStatement.Broadcast',
    'Contact_Correspondence': 'ldcOnt:Contact.Collaborate.Correspondence',
    'Justice_Sue': 'ldcOnt:Justice.InitiateJudicialProcess',
    'Justice_Convict': 'ldcOnt:Justice.JudicialConsequences.Convict',
    'Justice_Chargeindict': 'ldcOnt:Justice.InitiateJudicialProcess.ChargeIndict',
    'Justice_Trialhearing': 'ldcOnt:Justice.InitiateJudicialProcess.TrialHearing',
    'Justice_Execute': 'ldcOnt:Justice.JudicialConsequences.Execute',
    'Justice_Extradite': 'ldcOnt:Justice.JudicialConsequences.Extradite',
    'Manufacture_Artifact': 'ldcOnt:Manufacture.Artifact.CreateManufacture',
    'Justice_Arrestjail': 'ldcOnt:Justice.ArrestJailDetain.ArrestJailDetain',
}

kbp_backup_map = {
    'Justice_Arrestjail': 'ldcOnt:Justice.ArrestJailDetain',
    'Personnel_Endposition': 'ldcOnt:Personnel.EndPosition',
    'Movement_Transportartifact': 'ldcOnt:Movement.TransportArtifact',
    'Personnel_Startposition': 'ldcOnt:Personnel.StartPosition',
    'Justice_Sentence': 'ldcOnt:Justice.JudicialConsequences',
    'Movement_Transportperson': 'ldcOnt:Movement.TransportPerson',
    'Business_Mergeorg': 'ldcOnt:Government.Formation.MergeGPE',
}

kbp_lemma_map = {
    ('Conflict_Attack', 'shoot'): 'ldcOnt:Conflict.Attack.FirearmAttack',
    ('Conflict_Attack', 'shooting'): 'ldcOnt:Conflict.Attack.FirearmAttack',
    ('Conflict_Attack', 'fire'): 'ldcOnt:Conflict.Attack.FirearmAttack',
    ('Conflict_Attack', 'firing'): 'ldcOnt:Conflict.Attack.FirearmAttack',
    ('Change_of_leadership', 'election'): 'ldcOnt:Personnel.Elect',
    ('Change_of_leadership', 'elect'): 'ldcOnt:Personnel.Elect',
    ('Change_of_leadership', 'overthrow'): 'ldcOnt:Personnel.EndPosition',
    ('Impact', 'crash'): 'ldcOnt:Disaster.AccidentCrash.AccidentCrash',
}

kbp_frame_correction = {
    ('Justice_Arrestjail', 'Conquering'):
        'ldcOnt:Transaction.Transaction.TransferControl',
}

token_direct_map = {
    'seize': 'ldcOnt:Transaction.Transaction.TransferControl',
    'casualty': 'ldcOnt:Life.Die.DeathCausedByViolentEvents',
    'capture': 'ldcOnt:Transaction.Transaction.TransferControl',
    'bloodsh': 'ldcOnt:Life.Die.DeathCausedByViolentEvents',
    'bloodshed': 'ldcOnt:Life.Die.DeathCausedByViolentEvents',
    'interview': 'ldcOnt:Contact.Discussion.Meet',
    'vote': 'ldcOnt:Government.Vote',
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

frame_lemma_map = {
    ('Killing', 'fire'): 'ldcOnt:Conflict.Attack.FirearmAttack',
    ('Killing', 'firing'): 'ldcOnt:Conflict.Attack.FirearmAttack',
    ('Killing', 'shoot'): 'ldcOnt:Conflict.Attack.FirearmAttack',
}

propbank_sense_map = {
    'fire.02': 'ldcOnt:Conflict.Attack.FirearmAttack',
}

frame_direct_map = {
    'Arriving': 'ldcOnt:Movement.TransportPerson',
    'Departing': 'ldcOnt:Movement.TransportPerson',
    'Employing': 'ldcOnt:Personnel.StartPosition',
    'Shoot_projectiles': 'ldcOnt:Conflict.Attack.FirearmAttack',
    'Statement': 'ldcOnt:Contact.CommitmentPromiseExpressIntent.Broadcast',
    'Communication_response': 'ldcOnt:Contact.Discussion',
    'Chatting': 'ldcOnt:Contact.Discussion',
    'Hostile_encounter': 'ldcOnt:Conflict.Attack',
    'Taking': 'ldcOnt:Transaction.Transaction.TransferControl',
    'Meet': 'ldcOnt:Contact.Collaborate.Meet',
    'Killing': 'ldcOnt:Life.Die.DeathCausedByViolentEvent',
    'Questioning': 'ldcOnt:Contact.Discussion',
    'Contacting': "ldcOnt:Contact",
    'Discussion': 'ldcOnt:Contact.Discussion',
    'Cause_harm': 'ldcOnt:Life.Injure.InjuryCausedByViolentEvent',
    'Attack': 'ldcOnt:Conflict.Attack',
    'Death': 'ldcOnt:Life.Die',
    'Dead_or_alive': 'ldcOnt:Life.Die',
    'Manufacturing': 'ldcOnt:Manufacture.Artifact',
    'Expressing_publicly': 'ldcOnt:Contact.CommitmentPromiseExpressIntent.Broadcast',
    'Hiring': 'ldcOnt:Personnel.StartPosition.Hiring',
    'Setting_fire': 'ldcOnt:Conflict.Attack.SetFire',
    'Communication': 'ldcOnt:Contact',
    'Trial': 'ldcOnt:Justice.InitiateJudicialProcess.TrialHearing',
    'Travel': 'ldcOnt:Movement.TransportPerson',
    'Firing': 'ldcOnt:Personnel.EndPosition.FiringLayoff',
    'Robbery': 'ldcOnt:Conflict.Attack.StealRobHijack',
    'Disaster_scenario': 'ldcOnt:Disaster',
    'Fire_emergency_scenario': 'ldcOnt:Disaster.FireExplosion.FireExplosion',
    'Criminal_investigation': 'ldcOnt:Justice.Investigate.InvestigateCrime',
    'Execution': 'ldcOnt:Justice.JudicialConsequences.Execute',
    'Extradition': 'ldcOnt:Justice.JudicialConsequences.Extradite',
    'Lending': 'ldcOnt:Transaction.TransferMoney.BorrowLend',
    'Commerce_buy': 'ldcOnt:Transaction.TransferMoney.Purchase',
    'Commerce_pay': 'ldcOnt:Transaction.TransferMoney.Purchase',
}

entity_specific_srl_map = {
    'ldcOnt:Conflict.Attack.FirearmAttack': {
        ('ARG3', 'PERSON'): 'Attacker',
    },
}

srl_ldc_arg_map = {
    'ldcOnt:ArtifactExistence.DamageDestroy': {
        'ARG2': 'Instrument',
        'ARG0': 'DamagerDestroyer',
        'ARG1': 'Artifact',
        'Victim': 'Artifact',
    },
    "ldcOnt:Conflict.Attack.SetFire": {
        'AR0': 'Attacker',
        "Kindler": "Attacker",
    },
    "ldcOnt:Disaster.AccidentCrash.AccidentCrash": {
        'ARG0': 'CrashObject',
        "Impactee": "CrashObject",
        "Impactor": "Vehicle"
    },
    "ldcOnt:Disaster.FireExplosion.FireExplosion": {
        "Fire": "FireExplosionObject",
    },
    "ldcOnt:Government.Formation.MergeGPE": {
        "ARG0": "Participant",
        "ARG1": "Participant",
    },
    "ldcOnt:Government.Vote": {
        "ARG1": "Ballot",
        "ARG2": "Candidate",
        "ARG0": "Voter",
    },
    "ldcOnt:Justice.InitiateJudicialProcess": {
        "Defendant": "Defendant",
        "Judge": "JudgeCourt",
        "Court": "JudgeCourt",
        "Prosecution": "Prosecutor",
        "Charges": "Crime",
        "Case": "Crime",
    },
    "ldcOnt:Justice.Investigate.InvestigateCrime": {
        "Suspect": "Defendant",
        "Investigator": "Investigator",
        "Incident": "Crime",
    },
    "ldcOnt:Justice.JudicialConsequences.Execute": {
        "Reason": "Crime",
        "Executed": "Defendant",
        "Executioner": "Executioner"
    },
    "ldcOnt:Justice.JudicialConsequences.Extradite": {
        "Crime_jurisdiction": "Destination",
        "Suspect": "Defendant",
        "Authorities": "Extraditer",
        "Current_jurisdiction": "Origin",
        "Explanation": "Crime"
    },
    "ldcOnt:Manufacture.Artifact": {
        "Instrument": "Instrument",
        "Product": "Artifact",
        "Producer": "Manufacturer",
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
        'Killer': 'Attacker',
        'Victim': 'Target',
        'Assailant': 'Attacker',
        'Instrument': 'Instrument',
        'Weapon': 'Instrument',
        'Sides': 'Attacker',
        'Side_1': 'Attacker',
        'Side_2': 'Attacker',
        'Agent': 'Attacker',
        'Person': 'Attacker',  # open fire case
        'prep-on': 'Target',
        'prep-by': 'Attacker',
        'ARGM-ADV': 'Target',
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
        'Mode_of_transportation': 'Vehicle',
        'Means': 'Vehicle',
        'Place': 'Destination',
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
    "ldcOnt:Personnel.Elect": {
        "Selector": "Voter",
        "New_leader": "Candidate"
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
        'Authorities': 'Detainee',
        'Suspect': 'Jailer',
        'Charges': 'Crime',
    },
    "ldcOnt:Life.Die": {
        "Protagonist": "Victim",
    },
    'ldcOnt:Life.Die.DeathCausedByViolentEvents': {
        'Entity': 'Victim',
        "Protagonist": "Victim",
        'Assailant': 'Attacker',
        'Cause': 'Instrument',
        'Weapon': 'Instrument',
        'Agent': 'Attacker',
    },
    "ldcOnt:Life.Injure": {
        "Victim": "Victim",
        "Cause": "Injurer",
        "Agent": "Injurer",
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
        'Lender': 'Giver',
        'Theme': 'Money',
    },
    "ldcOnt:Transaction.TransferMoney.Purchase": {
        "Money": "Money",
        "Buyer": "Recipient",
        "Seller": "Giver",
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
