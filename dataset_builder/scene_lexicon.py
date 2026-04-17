import re
from collections import Counter

SCENE_KEYWORDS = {
    "hospital": {
        "hospital", "ward", "doctor", "nurse", "patient", "medical",
        "clinic", "bed", "stretcher", "syringe", "ambulance"
    },
    "kitchen": {
        "kitchen", "cooking", "cook", "stove", "sink", "refrigerator",
        "fridge", "counter", "pan", "pot", "oven", "cabinet"
    },
    "office": {
        "office", "desk", "computer", "laptop", "monitor", "keyboard",
        "meeting", "workspace", "chair", "printer", "screen"
    },
    "street": {
        "street", "road", "sidewalk", "traffic", "car", "truck", "bus",
        "crosswalk", "vehicle", "pavement", "intersection"
    },
    "school": {
        "school", "classroom", "teacher", "student", "desk", "blackboard",
        "whiteboard", "lesson", "campus", "book", "notebook"
    }
}

PERSON_WORDS = {
    "person", "people", "man", "woman", "boy", "girl", "child", "children",
    "adult", "adults", "guy", "lady", "gentleman", "human", "worker"
}

VAGUE_OBJECTS = {
    "area", "thing", "stuff", "object", "objects", "part", "parts",
    "background", "foreground", "scene", "environment", "setting",
    "structure", "detail", "details"
}

WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower()
    return text


def tokenize(text: str):
    text = normalize_text(text)
    return WORD_RE.findall(text)


def normalize_object_name(name: str) -> str:
    name = normalize_text(name).strip()

    synonym_map = {
        "men": "person",
        "women": "person",
        "man": "person",
        "woman": "person",
        "boy": "person",
        "girl": "person",
        "people": "person",
        "persons": "person",
        "babies": "baby",
        "chairs": "chair",
        "desks": "desk",
        "laptops": "laptop",
        "helmets": "helmet",
        "trucks": "truck",
        "cars": "car",
        "buses": "bus",
        "monitors": "monitor",
        "computers": "computer",
        "roads": "road",
        "streets": "street",
        "sidewalks": "sidewalk",
        "doctors": "doctor",
        "nurses": "nurse",
        "patients": "patient",
        "students": "student",
        "teachers": "teacher",
    }
    return synonym_map.get(name, name)


def dedup_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def scene_match_from_texts(texts):
    """
    输入若干文本，返回:
    {
        "scene": "hospital",
        "score": 4,
        "hits": {"hospital": ["doctor", "patient"], ...}
    }
    """
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenize(t))

    token_counter = Counter(all_tokens)
    scene_scores = {}
    scene_hits = {}

    for scene, keywords in SCENE_KEYWORDS.items():
        hits = [kw for kw in keywords if token_counter.get(kw, 0) > 0]
        if hits:
            scene_scores[scene] = sum(token_counter[h] for h in hits)
            scene_hits[scene] = hits

    if not scene_scores:
        return {"scene": None, "score": 0, "hits": {}}

    best_scene = sorted(scene_scores.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return {
        "scene": best_scene,
        "score": scene_scores[best_scene],
        "hits": scene_hits,
    }


def contains_person_words(texts):
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenize(t))
    return any(tok in PERSON_WORDS for tok in all_tokens)


def filter_object_list(obj_list):
    cleaned = []
    for x in obj_list:
        x = normalize_object_name(x)
        if not x:
            continue
        if x in VAGUE_OBJECTS:
            continue
        cleaned.append(x)
    return dedup_keep_order(cleaned)