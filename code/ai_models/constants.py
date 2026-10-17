from types import MappingProxyType

PERSON_CLASS_ID = 0

VEHICLE_CLASS_NAMES = MappingProxyType(
    {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
)

DRONE_PROXY_CLASS_NAMES = MappingProxyType({4: "airplane", 14: "bird", 33: "kite"})

DRONE_LABELS = (
    "drone",
    "uav",
    "uas",
    "quadcopter",
    "multirotor",
    "fixed_wing",
    "fixed-wing",
)

DRONE_REAL_WIDTH_M = MappingProxyType(
    {
        "micro_drone": 0.12,
        "quadcopter": 0.35,
        "fixed_wing": 1.5,
        "unknown": 0.5,
    }
)

WEIGHT_SUFFIXES = (".pt", ".onnx", ".engine", ".torchscript")
