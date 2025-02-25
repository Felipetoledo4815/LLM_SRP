import sg_primitives as P
from functools import partial
from Property import Property

######################
### New predicates ###
######################

EGO = partial(P.filter_by_attr, "G", "name", "ego")

WITHIN_25M = partial(P.rel_set, EGO, "within_25m", edge_type="incoming")
IS_WITHIN_25M = partial(partial(P.gt, partial(P.size, WITHIN_25M), 0))
BETWEEN_25_40M = partial(P.rel_set, EGO, "between_25m_and_40m", edge_type="incoming")
IS_BETWEEN_25_40M = partial(partial(P.gt, partial(P.size, BETWEEN_25_40M), 0))
BETWEEN_40_60M = partial(P.rel_set, EGO, "between_40m_and_60m", edge_type="incoming")
IS_BETWEEN_40_60M = partial(partial(P.gt, partial(P.size, BETWEEN_40_60M), 0))

IN_FRONT_OF = partial(P.rel_set, EGO, "in_front_of", edge_type="incoming")
IS_IN_FRONT_OF = partial(partial(P.gt, partial(P.size, IN_FRONT_OF), 0))
TO_LEFT_OF = partial(P.rel_set, EGO, "to_left_of", edge_type="incoming")
IS_TO_LEFT_OF = partial(partial(P.gt, partial(P.size, TO_LEFT_OF), 0))
TO_RIGHT_OF = partial(P.rel_set, EGO, "to_right_of", edge_type="incoming")
IS_TO_RIGHT_OF = partial(partial(P.gt, partial(P.size, TO_RIGHT_OF), 0))

IS_DEACCELERATING = partial(P.eq, partial(P.size, partial(
    P.filter_by_attr, "Ego", "acceleration", (lambda a: P.le(a, b=0)))), 1)
IS_ACCELERATING = partial(P.eq, partial(P.size, partial(
    P.filter_by_attr, "Ego", "acceleration", (lambda a: P.gt(a, b=0)))), 1)

FASTER_THAN_15MPH = partial(P.eq, partial(P.size, partial(
    P.filter_by_attr, "Ego", "speed", (lambda a: P.gt(a, b=15)))), 1)
FASTER_THAN_25MPH = partial(P.eq, partial(P.size, partial(
    P.filter_by_attr, "Ego", "speed", (lambda a: P.gt(a, b=25)))), 1)
FASTER_THAN_35MPH = partial(P.eq, partial(P.size, partial(
    P.filter_by_attr, "Ego", "speed", (lambda a: P.gt(a, b=35)))), 1)
FASTER_THAN_45MPH = partial(P.eq, partial(P.size, partial(
    P.filter_by_attr, "Ego", "speed", (lambda a: P.gt(a, b=45)))), 1)

# Vehicle
VEHICLE_WITHIN_25M = partial(P.filter_by_attr, WITHIN_25M, "name", "vehicle")
IS_VEHICLE_WITHIN_25M = partial(partial(P.gt, partial(P.size, VEHICLE_WITHIN_25M), 0))

VEHICLE_BETWEEN_25_40M = partial(P.filter_by_attr, BETWEEN_25_40M, "name", "vehicle")
IS_VEHICLE_BETWEEN_25_40M = partial(partial(P.gt, partial(P.size, VEHICLE_BETWEEN_25_40M), 0))

VEHICLE_BETWEEN_40_60M = partial(P.filter_by_attr, BETWEEN_40_60M, "name", "vehicle")
IS_VEHICLE_BETWEEN_40_60M = partial(partial(P.gt, partial(P.size, VEHICLE_BETWEEN_40_60M), 0))

# Bicycle
BICYCLE_WITHIN_25M = partial(P.filter_by_attr, WITHIN_25M, "name", "bicycle")
IS_BICYCLE_WITHIN_25M = partial(partial(P.gt, partial(P.size, BICYCLE_WITHIN_25M), 0))

BICYCLE_BETWEEN_25_40M = partial(P.filter_by_attr, BETWEEN_25_40M, "name", "bicycle")
IS_BICYCLE_BETWEEN_25_40M = partial(partial(P.gt, partial(P.size, BICYCLE_BETWEEN_25_40M), 0))

BICYCLE_BETWEEN_40_60M = partial(P.filter_by_attr, BETWEEN_40_60M, "name", "bicycle")
IS_BICYCLE_BETWEEN_40_60M = partial(partial(P.gt, partial(P.size, BICYCLE_BETWEEN_40_60M), 0))

# Person
PERSON_WITHIN_25M = partial(P.filter_by_attr, WITHIN_25M, "name", "person")
IS_PERSON_WITHIN_25M = partial(partial(P.gt, partial(P.size, PERSON_WITHIN_25M), 0))

PERSON_BETWEEN_25_40M = partial(P.filter_by_attr, BETWEEN_25_40M, "name", "person")
IS_PERSON_BETWEEN_25_40M = partial(partial(P.gt, partial(P.size, PERSON_BETWEEN_25_40M), 0))

PERSON_BETWEEN_40_60M = partial(P.filter_by_attr, BETWEEN_40_60M, "name", "person")
IS_PERSON_BETWEEN_40_60M = partial(partial(P.gt, partial(P.size, PERSON_BETWEEN_40_60M), 0))


#############
### PHI 0 ###
#############
phi0_base = Property(
    "Phi0_base",
    "G(((is_vehicle_within_25m | is_bicycle_within_25m | is_person_within_25m) & faster_than_15mph) & X ((is_vehicle_within_25m | is_bicycle_within_25m | is_person_within_25m) & faster_than_15mph) -> (X is_deaccelerating))",
    [("is_vehicle_within_25m", IS_VEHICLE_WITHIN_25M),
    ("is_bicycle_within_25m", IS_BICYCLE_WITHIN_25M),
    ("is_person_within_25m", IS_PERSON_WITHIN_25M),
    ("faster_than_15mph", FASTER_THAN_15MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)

#############
### PHI 1 ###
#############
phi1_base = Property(
    "Phi1_base",
    "G(((is_vehicle_within_25m | is_bicycle_within_25m | is_person_within_25m) & faster_than_25mph) & X ((is_vehicle_within_25m | is_bicycle_within_25m | is_person_within_25m) & faster_than_25mph) -> (X is_deaccelerating))",
    [("is_vehicle_within_25m", IS_VEHICLE_WITHIN_25M),
    ("is_bicycle_within_25m", IS_BICYCLE_WITHIN_25M),
    ("is_person_within_25m", IS_PERSON_WITHIN_25M),
    ("faster_than_25mph", FASTER_THAN_25MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)
phi1_vehicle_base = Property(
    "Phi1_vehicle_base",
    "G((is_vehicle_within_25m & faster_than_25mph) & X (is_vehicle_within_25m & faster_than_25mph) -> (X is_deaccelerating))",
    [("is_vehicle_within_25m", IS_VEHICLE_WITHIN_25M),
    ("faster_than_25mph", FASTER_THAN_25MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)
phi1_bicycle_base = Property(
    "Phi1_bicycle_base",
    "G((is_bicycle_within_25m & faster_than_25mph) & X (is_bicycle_within_25m & faster_than_25mph) -> (X is_deaccelerating))",
    [("is_bicycle_within_25m", IS_BICYCLE_WITHIN_25M),
    ("faster_than_25mph", FASTER_THAN_25MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)
phi1_person_base = Property(
    "Phi1_person_base",
    "G((is_person_within_25m & faster_than_25mph) & X (is_person_within_25m & faster_than_25mph) -> (X is_deaccelerating))",
    [("is_person_within_25m", IS_PERSON_WITHIN_25M),
    ("faster_than_25mph", FASTER_THAN_25MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)

#############
### PHI 2 ###
#############
phi2_base = Property(
    "Phi2_base",
    "G(((is_vehicle_between_25_40m | is_bicycle_between_25_40m | is_person_between_25_40m) & faster_than_35mph) & X ((is_vehicle_between_25_40m | is_bicycle_between_25_40m | is_person_between_25_40m) & faster_than_35mph) -> (X is_deaccelerating))",
    [("is_vehicle_between_25_40m", IS_VEHICLE_BETWEEN_25_40M),
    ("is_bicycle_between_25_40m", IS_BICYCLE_BETWEEN_25_40M),
    ("is_person_between_25_40m", IS_PERSON_BETWEEN_25_40M),
    ("faster_than_35mph", FASTER_THAN_35MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)
phi2_vehicle_base = Property(
    "Phi2_vehicle_base",
    "G((is_vehicle_between_25_40m & faster_than_25mph) & X (is_vehicle_between_25_40m & faster_than_25mph) -> (X is_deaccelerating))",
    [("is_vehicle_between_25_40m", IS_VEHICLE_BETWEEN_25_40M),
    ("faster_than_25mph", FASTER_THAN_25MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)
phi2_bicycle_base = Property(
    "Phi2_bicycle_base",
    "G((is_bicycle_between_25_40m & faster_than_25mph) & X (is_bicycle_between_25_40m & faster_than_25mph) -> (X is_deaccelerating))",
    [("is_bicycle_between_25_40m", IS_BICYCLE_BETWEEN_25_40M),
    ("faster_than_25mph", FASTER_THAN_25MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)
phi2_person_base = Property(
    "Phi2_person_base",
    "G((is_person_between_25_40m & faster_than_25mph) & X (is_person_between_25_40m & faster_than_25mph) -> (X is_deaccelerating))",
    [("is_person_between_25_40m", IS_PERSON_BETWEEN_25_40M),
    ("faster_than_25mph", FASTER_THAN_25MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)


#############
### PHI 3 ###
#############
phi3_base = Property(
    "Phi3_base",
    "G(((is_vehicle_between_40_60m | is_bicycle_between_40_60m | is_person_between_40_60m) & faster_than_45mph) & X ((is_vehicle_between_40_60m | is_bicycle_between_40_60m | is_person_between_40_60m) & faster_than_45mph) -> (X is_deaccelerating))",
    [("is_vehicle_between_40_60m", IS_VEHICLE_BETWEEN_40_60M),
    ("is_bicycle_between_40_60m", IS_BICYCLE_BETWEEN_40_60M),
    ("is_person_between_40_60m", IS_PERSON_BETWEEN_40_60M),
    ("faster_than_45mph", FASTER_THAN_45MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)
phi3_vehicle_base = Property(
    "Phi3_vehicle_base",
    "G((is_vehicle_between_40_60m & faster_than_25mph) & X (is_vehicle_between_40_60m & faster_than_25mph) -> (X is_deaccelerating))",
    [("is_vehicle_between_40_60m", IS_VEHICLE_BETWEEN_40_60M),
    ("faster_than_25mph", FASTER_THAN_25MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)
phi3_bicycle_base = Property(
    "Phi3_bicycle_base",
    "G((is_bicycle_between_40_60m & faster_than_25mph) & X (is_bicycle_between_40_60m & faster_than_25mph) -> (X is_deaccelerating))",
    [("is_bicycle_between_40_60m", IS_BICYCLE_BETWEEN_40_60M),
    ("faster_than_25mph", FASTER_THAN_25MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)
phi3_person_base = Property(
    "Phi3_person_base",
    "G((is_person_between_40_60m & faster_than_25mph) & X (is_person_between_40_60m & faster_than_25mph) -> (X is_deaccelerating))",
    [("is_person_between_40_60m", IS_PERSON_BETWEEN_40_60M),
    ("faster_than_25mph", FASTER_THAN_25MPH),
    ("is_deaccelerating", IS_DEACCELERATING)]
)

all_properties = [
    phi0_base,
    phi1_base,
    phi2_base,
    phi3_base
]