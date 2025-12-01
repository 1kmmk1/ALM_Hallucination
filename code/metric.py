from typing import Dict, List, Tuple, Set

CATEGORY_VARIATIONS = {
    "laughing": ["laughter", "laugh", "laughs", "giggle", "chuckle"],
    "crying": ["cry", "cries", "sob", "sobbing"],
    "breathing": ["breaths", "gasp", "gasping", "sigh", "sighing"],
    "speaking": ["speech", "speak", "speaks"],
    "whispering": ["whisper", "whispers"],
    "screaming": ["scream", "screams", "shout", "yelling"],
    "coughing": ["cough", "coughs"],
    "singing": ["sing", "sings", "hum", "humming"],
    "man": ["male", "adult male"],
    "woman": ["female", "adult female"],
}

def build_norm_map(variations: dict) -> dict:
    norm_map = {}
    for standard, variants in variations.items():
        norm_map[standard] = standard
        for v in variants:
            norm_map[v.lower()] = standard
    return norm_map

def calculate_abef(
    generated: List[List[str]], 
    ground_truth: List[List[str]],
    norm_map: dict = None
) -> Dict[str, float]:
    """
    Calculate ABEF scores (H_obj, H_attr, H_bind).
    
    Args:
        generated: List of [object, attribute] pairs from model
        ground_truth: List of [object, attribute] pairs (GT)
    
    Returns:
        Dictionary with Acc, H_obj, H_attr, H_bind percentages
    """
    if norm_map is None:
        norm_map = build_norm_map(CATEGORY_VARIATIONS)
    
    if not generated:
        return {"acc": 0, "h_obj": 0, "h_attr": 0, "h_bind": 0}
    
    # Normalize
    def normalize(facts):
        return [(norm_map.get(o.lower(), o.lower()), 
                 norm_map.get(a.lower(), a.lower())) 
                for o, a in facts]
    
    gt_norm = set(normalize(ground_truth))
    gt_objs = {o for o, _ in gt_norm}
    gt_attrs = {a for _, a in gt_norm}
    gen_norm = normalize(generated)
    
    n = len(gen_norm)
    correct = sum(1 for f in gen_norm if f in gt_norm)
    h_obj = sum(1 for o, _ in gen_norm if o not in gt_objs)
    h_attr = sum(1 for _, a in gen_norm if a not in gt_attrs)
    h_bind = sum(1 for f in gen_norm if f not in gt_norm)
    
    return {
        "acc": correct / n * 100,
        "h_obj": h_obj / n * 100,
        "h_attr": h_attr / n * 100,
        "h_bind": h_bind / n * 100,
    }
