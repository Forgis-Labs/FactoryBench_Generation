from enum import Enum


class Stage(str, Enum):
    telemetry_literacy = "telemetry_literacy"
    root_cause_analysis = "root_cause_analysis"
    guided_remediation = "guided_remediation"


ALIASES = {
    "stage1": Stage.telemetry_literacy,
    "ts_understanding": Stage.telemetry_literacy,
    "time_series_understanding": Stage.telemetry_literacy,
    "stage2": Stage.root_cause_analysis,
    "fault_diagnosis": Stage.root_cause_analysis,
    "stage3": Stage.guided_remediation,
    "fault_fixing": Stage.guided_remediation,
    "repair_instruction_generation": Stage.guided_remediation,
}


def normalize_stage(s: str) -> Stage:
    s = (s or "").strip()
    # exact value
    for v in Stage:
        if s == v.value:
            return v
    # name-based
    if s in Stage.__members__:
        return Stage[s]  # type: ignore[index]
    # alias
    if s in ALIASES:
        return ALIASES[s]
    raise ValueError(f"Unknown stage: {s}")
