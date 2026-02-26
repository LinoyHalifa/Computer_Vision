import numpy as np
import torch
from PIL import Image


# ============================================
# Ablation flags – turn ON/OFF each station
# ============================================
USE_STATION1 = True
USE_STATION2 = True
USE_STATION3 = True
USE_STATION4 = True

# Global device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPT model name (used in GPT logic + explanation)
GPT_MODEL = "gpt-4o-mini"


# ============================================================
#  NORMAL LABELS — מצבים תקינים בלבד
# ============================================================
NORMAL_LABELS = [

    (
        "person_walking_on_sidewalk",
        "a clear sidewalk at night with a person walking safely "
        "cars are parked normally and the walkway is unobstructed."
    ),

    (
        "person_walking_on_sidewalk_with_a_dog",
        "a clear sidewalk at night with a man walking safely with a dog. "
        "cars are parked normally and the walkway is unobstructed."
    ),

    (
        "a_woman_walking_on_sidewalk",
        "a clear sidewalk at night with a woman walking safely with a handbag "
        "cars are parked normally and the walkway is unobstructed."
    ),

    (
      "scooter_park_correctly",
      "electric scooters parked correctly in a designated parking area on the sidewalk at night, with helmets attached and cars parked on the road"
    ),

    (
       "pedestrians_cross_crosswalk",
        "Two pedestrians cross the crosswalk correctly when the traffic light is green."
    ),

    (
       "person_cross_crosswalk",
        "a person cross the crosswalk correctly ."
    ),


    #(
      #"two_women_walking_on_sidewalk",
      #"a wide and well-lit walkway at night with two women walking safely."
    #),

    (  "two_women_walking_on_sidewalk",
       "clean sidewalk at night with two people walking, cars parked on the road, streetlights and trees along the walkway"
    ),

    (
        "normal_sidewalk_with_parked_cars_and_streetlight",
        "a nighttime sidewalk with a clear walkway, a streetlight pole, and cars parked on the road."
    ),

    (
        "normal_empty_sidewalk",
        "an empty, clean urban sidewalk with open walking space and cars parked correctly on the road."
    ),

(
        "normal_scene",
        "an empty, clean urban sidewalk with open walking space and cars drive correctly on the road."
    ),

    (
        "normal_sidewalk_with_sign_and_wall",
        "a clear sidewalk at dusk with a crossing sign, a wall, and cars parked on the road."
    ),

    (
        "normal_active_intersection",
        "an urban sidewalk at night adjacent to a road intersection, with a car driving through the junction while the sidewalk remains clear and unobstructed for pedestrians."
    ),

]


# ============================================================
#  ANOMALY LABELS — רק מצבים חריגים
# ============================================================
ANOMALY_LABELS = [

    (
     "puddle_on_sidewalk",
      "a wet puddle on the sidewalk"

    ),
    (
        "vehicle_on_sidewalk",
        "a car, scooter, or motorcycle standing on the pedestrian sidewalk, "
        "blocking the walking path. no pedestrians are walking normally in this location."
    ),

    (
        "illegal_parking",
        #"a vehicle parked in a forbidden location such as on the sidewalk"
        #"blocking pedestrian movement. no normal pedestrian is standing or walking there."
         "A car is parked on the pedestrian sidewalk at night, blocking part of the walkway. The sidewalk surface is dry and clear, with no puddles or obstacles present."
    ),

    (
        "obstacle_on_sidewalk",
        "a large object, trash bag, cardboard box, construction material, or debris placed on the sidewalk, "
        "blocking pedestrian walking space. there are no pedestrians walking normally around the obstacle."
    ),

    (
        "vegetation_on_sidewalk",
        "bushes, plants, or overgrown vegetation spreading onto the sidewalk and blocking pedestrian passage. "
        "no pedestrians are walking normally in this location."
    ),
(
    "vegetation_on_road",
    "cut branches and green debris lying on the road next to the curb. "
    "this foreign object does not belong on the roadway and can obstruct visibility or movement. "
    "even though the sidewalk is clear, the presence of vegetation on the road is an anomaly."
),

(
    "large_obstacle_on_sidewalk",
    "a large wooden panel and cardboard boxes are lying on the sidewalk, blocking the pedestrian path. "
    "these foreign objects do not belong on the walkway and create a safety hazard by preventing normal movement. "
    "the sidewalk should be clear, but here it is obstructed by debris and discarded materials, making the scene anomalous."
),

(
    "foreign_object_on_road",
    "A foreign object lying on the road, inside the vehicle driving lane"

),

(
    "a_dress_on_road",
    "A dress lying on the road, inside the vehicle driving lane"

),

(

    "pedestrian_crossing_on_red_light",
    "a pedestrian cross the crosswalk when the traffic light is red."
),

    (
    "damaged_sidewalk",
    "a nighttime urban sidewalk with cracked and uneven pavement tiles creating an irregular walking surface."

    ),

    (
      "crack_on_sidewalk",
      "a nighttime urban sidewalk with visible longitudinal cracks along the pavement surface."
    ),



]


# ============================================================
#  Utility functions
# ============================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity assuming a and b are L2-normalized."""
    return float(np.dot(a, b))


def crop_region(img_rgb: np.ndarray, bbox):
    """Crop region from RGB image given [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), img_rgb.shape[1])
    y2 = min(int(y2), img_rgb.shape[0])
    return img_rgb[y1:y2, x1:x2]


def clip_image_embedding(clip_model, preprocess, img_rgb: np.ndarray) -> np.ndarray:
    """Compute CLIP image embedding from numpy RGB image."""
    pil_img = Image.fromarray(img_rgb)
    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze(0).cpu().numpy()
