from angle_computation import metrics, cadence, vertical_oscillation
import numpy as np
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

#benchmarks
MODERATE_TIBIAL_INCLINATION = 10
MAX_TIBIAL_INCLINATION = 15

MAX_KNEE_FLEXION = 165
MODERATE_KNEE_FLEXION = 155

MAX_KNEE_FLEXION_MIDSWING = 115 
MIN_KNEE_FLEXION_MIDSWING = 90

MIN_CADENCE = 160 
MODERATE_CADENCE = 170

MODERATE_ASYMMETRY = 5
MAX_ASYMMETRY = 10


def compute_metrics(metrics, vertical_oscillation, cadence):
    """Compute average metrics across left and right sides and in first and second half of video."""

    def safe_mean(arr):
        return np.mean(arr) if len(arr) > 0 else None
    average_metrics = {}

    max1, max2 = np.array_split(vertical_oscillation[0], 2)
    min1, min2 = np.array_split(vertical_oscillation[1], 2)
    average_metrics['vertical_oscillation'] = [np.mean(max1) - np.mean(min1), np.mean(max2) - np.mean(min2)]

    average_metrics["cadence"] = [cadence]

    for metric in metrics: 
        left1, left2 = np.array_split(metrics[metric]['left'], 2)
        right1, right2 = np.array_split(metrics[metric]['right'], 2)

        average_metrics[metric] = [safe_mean(x) for x in [left1, left2, right1, right2]]

    return average_metrics

def build_prompt(metrics):
    findings = {"overstriding" : overstriding_feedback(metrics),
                "asymmetry": asymmetry_feedback(metrics),
                "fatigue": fatigue_feedback(metrics)
    }
    
    prompt = f"""<instructions>
        You are a running biomechanics coach. Given the analysis below, write a 3-4 
        sentence feedback paragraph for a recreational runner. Be encouraging but 
        specific. Focus on the top 2-3 issues. For each issue, include one concrete 
        cue the runner can think about during their next run. Do not invent metrics 
        or problems beyond what is provided. If no findings are flagged, tell the 
        runner their form looks solid.
        </instructions>

        <analysis_data>
        {json.dumps(findings, indent=2)}
        </analysis_data>"""
    
    return prompt

def overstriding_feedback(metrics):
    severity = 'none'
    worsening = False
    low_cadence = False
    detail = ''
    cue = ''

    if metrics["tibial_inclination"] is None or metrics["tibial_inclination"][0] is None or metrics["tibial_inclination"][2] is None:
        return {
            'issue': 'overstriding',
            'severity': 'none',
            'detail': 'Insufficient data to assess overstriding.',
            'cue': '',
        }

    tib_inc_1 = (metrics['tibial_inclination'][0] + metrics['tibial_inclination'][2]) / 2
    # Second half may be missing if the video is very short
    if metrics["tibial_inclination"][1] is not None and metrics["tibial_inclination"][3] is not None:
        tib_inc_2 = (metrics["tibial_inclination"][1] + metrics["tibial_inclination"][3]) / 2
    else:
        tib_inc_2 = tib_inc_1  # fall back to first half, skip worsening check

    if metrics['cadence'] is not None and metrics['cadence'][0] is not None and metrics['cadence'][0] < MIN_CADENCE:
        low_cadence = True


    if tib_inc_1 > MAX_TIBIAL_INCLINATION:
        severity = 'high'
        if tib_inc_2 > tib_inc_1 + 3:
            worsening = True
    elif tib_inc_1 > MODERATE_TIBIAL_INCLINATION:
        severity = 'moderate'
        if tib_inc_2 > tib_inc_1 + 3:
            worsening = True
    elif tib_inc_2 > MODERATE_TIBIAL_INCLINATION: 
        severity = 'moderate'
        worsening = True

    
    if severity == 'none':
        if not low_cadence: 
            detail = f"Tibial inclination and cadence within normal limits."
            cue = f'No changes needed.'
        else:
            detail = f"Tibial inclination within normal limits. Cadence of {metrics['cadence'][0]} is slightly low."
            cue = f'Consider increasing cadence slightly.'
    else: 
        detail = f"Tibial inclination is high, averaging {tib_inc_1:.1f} degrees forward in the first half"
        if worsening: 
            detail += f", worsening in the second half to {tib_inc_2:.1f} degrees in the second half of the video."
        else:
            detail += f", stable in the second half at {tib_inc_2:.1f} degrees."
        cue += f"Focus on landing feet underneath body."

        if low_cadence:
            detail += f" Cadence of {metrics['cadence'][0]} is also low."
            cue += f'Increasing cadence slightly may help.'


    findings = {
        "issue": 'overstriding',
        "severity": severity,
        "detail": detail,
        "cue": cue
    }
    
    return findings

def asymmetry_feedback(metrics):
    asymmetries = []
 
    per_side_metrics = {
        'tibial_inclination': 'tibial inclination',
        'knee_flexion': 'knee flexion at contact',
        'thigh_angle': 'thigh angle',
        'knee_swing': 'knee flexion during swing',
    }
 
    for key, label in per_side_metrics.items():
        if key not in metrics:
            continue
 
        values = metrics[key]
        # [left_first, left_second, right_first, right_second]
        left_avg = np.mean([v for v in [values[0], values[1]] if v is not None])
        right_avg = np.mean([v for v in [values[2], values[3]] if v is not None])
 
        if np.isnan(left_avg) or np.isnan(right_avg):
            continue
 
        diff = abs(left_avg - right_avg)
        if diff >= MODERATE_ASYMMETRY:
            affected_side = 'left' if left_avg > right_avg else 'right'
            asymmetries.append({
                'metric': label,
                'diff': diff,
                'affected_side': affected_side,
                'left_avg': left_avg,
                'right_avg': right_avg,
            })
 
    if not asymmetries:
        return {
            'issue': 'asymmetry',
            'severity': 'none',
            'detail': 'No significant left-right asymmetries detected.',
            'cue': 'No changes needed.',
        }
 
    # Report the largest asymmetry
    worst = max(asymmetries, key=lambda a: a['diff'])
 
    if any(a['diff'] >= MAX_ASYMMETRY for a in asymmetries):
        severity = 'high'
    else:
        severity = 'moderate'
 
    detail_parts = []
    for a in sorted(asymmetries, key=lambda x: x['diff'], reverse=True):
        detail_parts.append(
            f"{a['metric']}: left averages {a['left_avg']:.1f} degrees, "
            f"right averages {a['right_avg']:.1f} degrees "
            f"({a['diff']:.1f} degree difference)")

    detail = "Notable left-right asymmetries detected. " + ". ".join(detail_parts) + "."

 
    other_side = 'right' if worst['affected_side'] == 'left' else 'left'
    cue = (f"Your {worst['affected_side']} side shows a different pattern than your "
           f"{other_side} side. Consider consulting a physical therapist if this persists.")
 
    return {
        'issue': 'asymmetry',
        'severity': severity,
        'detail': detail,
        'cue': cue,
    }

def fatigue_feedback(metrics):
    degradations = []
 
    # Per-side metrics: average left+right for each half, then compare halves.
    # 'direction' indicates which direction is worse (increase or decrease).
    # 'threshold' is the minimum delta in degrees to count as degradation.
    per_side_metrics = {
        'tibial_inclination': {
            'label': 'tibial inclination',
            'direction': 'increase',  # higher = more forward shin = worse
            'threshold': 3,
        },
        'knee_flexion': {
            'label': 'knee flexion at contact',
            'direction': 'increase',  # higher = straighter leg = worse
            'threshold': 5,
        },
        'thigh_angle': {
            'label': 'thigh angle',
            'direction': 'increase',  # more forward reach = worse
            'threshold': 3,
        },
        'knee_swing': {
            'label': 'knee flexion during swing',
            'direction': 'increase',  # higher = less folded = worse
            'threshold': 5,
        },
    }
 
    for key, config in per_side_metrics.items():
        if key not in metrics:
            continue
 
        values = metrics[key]
        # [left_first, left_second, right_first, right_second]
        first_half_vals = [v for v in [values[0], values[2]] if v is not None]
        second_half_vals = [v for v in [values[1], values[3]] if v is not None]
 
        if not first_half_vals or not second_half_vals:
            continue
 
        first_avg = np.mean(first_half_vals)
        second_avg = np.mean(second_half_vals)
 
        if config['direction'] == 'increase':
            delta = second_avg - first_avg
        else:
            delta = first_avg - second_avg
 
        if delta > config['threshold']:
            degradations.append({
                'metric': config['label'],
                'first_half': first_avg,
                'second_half': second_avg,
                'delta': delta,
            })
 
    # Vertical oscillation: [first_half, second_half]
    # Uses relative increase (%) since values are in normalized coordinates, not cm
    if 'vertical_oscillation' in metrics:
        vo = metrics['vertical_oscillation']
        if vo[0] is not None and vo[1] is not None and vo[0] > 0:
            relative_increase = (vo[1] - vo[0]) / vo[0]
            if relative_increase > 0.15:  # 15% increase in bounce
                degradations.append({
                    'metric': 'vertical oscillation',
                    'first_half': vo[0],
                    'second_half': vo[1],
                    'delta': relative_increase,
                })
 
    if not degradations:
        return {
            'issue': 'fatigue',
            'severity': 'none',
            'detail': 'Form remains consistent between first and second half of the video.',
            'cue': 'No changes needed.',
        }
 
    if len(degradations) >= 3:
        severity = 'high'
    else:
        severity = 'moderate'
 
    # Lead with the worst degradation
    worst = max(degradations, key=lambda d: d['delta'])
 
    if worst['metric'] == 'vertical oscillation':
        detail = (f"Form shows signs of fatigue in the second half of the video. "
                  f"Vertical oscillation increased by {worst['delta']:.0%}.")
    else:
        detail = (f"Form shows signs of fatigue in the second half of the video. "
                  f"{worst['metric'].capitalize()} changed from "
                  f"{worst['first_half']:.1f} to {worst['second_half']:.1f} degrees.")
 
    if len(degradations) > 1:
        other_labels = [d['metric'] for d in degradations if d != worst]
        detail += f" {', '.join([l.capitalize() for l in other_labels])} also degraded."
 
    cue = (f"Your form breaks down as the clip progresses, particularly your {worst['metric']}, which may indicate "
           "you are pushing past your current endurance. Consider shorter intervals "
           "or a slightly slower pace to maintain form.")
 
    return {
        'issue': 'fatigue',
        'severity': severity,
        'detail': detail,
        'cue': cue,
    }

prompt = build_prompt(compute_metrics(metrics, vertical_oscillation, cadence))



# Securely set your key as an environment variable or provide it directly
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-3-flash-preview", 
    contents=[prompt],
    config=types.GenerateContentConfig(
        temperature=0.1
    )
)

print(response.text)