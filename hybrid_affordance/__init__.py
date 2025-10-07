AFFORDANCE_CLASSES = [
    'background',
    'grasp',
    'cut',
    'scoop',
    'contain',
    'pound',
    'support',
    'wrap-grasp',
]

CLASS_TO_INDEX = {name: idx for idx, name in enumerate(AFFORDANCE_CLASSES)}
INDEX_TO_CLASS = {idx: name for idx, name in enumerate(AFFORDANCE_CLASSES)}

DEFAULT_ENCODER = 'efficientnet-b3'
NUM_CLASSES = len(AFFORDANCE_CLASSES)
