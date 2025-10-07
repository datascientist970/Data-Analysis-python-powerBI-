AFFORDANCE_CLASSES = [
    'background', 'grasp', 'cut', 'scoop', 'contain', 'pound', 'support', 'wrap-grasp'
]

AFFORDANCE_COLORS = {
    0: (0, 0, 0),          # background - black
    1: (0, 255, 0),        # grasp - green
    2: (255, 0, 0),        # cut - red
    3: (255, 165, 0),      # scoop - orange
    4: (0, 191, 255),      # contain - deep sky blue
    5: (128, 0, 128),      # pound - purple
    6: (255, 255, 0),      # support - yellow
    7: (255, 20, 147),     # wrap-grasp - deeppink
}

DEFAULT_MODEL_PATH = "/workspace/best_model.pth"
DEFAULT_RESULTS_DIR = "/workspace/hybrid_results"
