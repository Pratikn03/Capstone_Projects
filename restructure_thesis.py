import os
import re
from pathlib import Path

GROUPS = {
    "ch01_introduction.tex": [
        "ch01_introduction.tex", 
        "ch03_problem_formulation.tex"
    ],
    "ch02_related_work.tex": [
        "ch02_related_work.tex"
    ],
    "ch03_background.tex": [
        "ch05_orius_system_context.tex", 
        "ch06_data_telemetry_scope.tex", 
        "ch04_observed_state_safety_illusion.tex", 
        "ch07_battery_dynamics_dispatch.tex"
    ],
    "ch04_theoretical_foundations.tex": [
        "ch09_dc3s_battery_adapter.tex", 
        "ch15_assumptions_notation_proof_discipline.tex", 
        "ch16_battery_theorem_oasg_existence.tex", 
        "ch17_battery_theorem_safety_preservation.tex", 
        "ch18_orius_core_bound_battery.tex", 
        "ch19_no_free_safety_battery.tex", 
        "ch20_temporal_behavioral_extensions.tex"
    ],
    "ch05_architecture_setup.tex": [
        "ch08_forecasting_calibration.tex", 
        "ch10_cpsbench_battery_track.tex", 
        "ch30_orius_bench_battery_track.tex", 
        "ch22_latency_systems_footprint.tex"
    ],
    "ch06_main_results.tex": [
        "ch11_main_battery_results.tex", 
        "ch12_ablations_failure_analysis.tex", 
        "ch13_case_studies_operational_traces.tex", 
        "ch14_battery_lessons_domain_interpretation.tex", 
        "ch21_fault_performance_stress_tests.tex", 
        "ch23_hyperparameter_surface_stability.tex", 
        "ch24_conditional_coverage_subgroups.tex", 
        "ch25_regional_decomposition_real_prices.tex", 
        "ch26_asset_preservation_aging_proxy.tex", 
        "ch28_certificate_half_life_blackout.tex", 
        "ch29_graceful_degradation_safe_landing.tex", 
        "ch31_compositional_safety_battery_fleets.tex", 
        "ch32_adversarial_robustness_active_probing.tex", 
        "ch32_certos_runtime_certificate_lifecycle.tex"
    ],
    "ch07_universal_validation.tex": [
        "ch21_battery_to_universal_orius.tex", 
        "ch27_hardware_in_loop_validation.tex", 
        "ch34_outside_current_evidence.tex"
    ],
    "ch08_conclusion.tex": [
        "ch22_what_this_thesis_proves.tex", 
        "ch33_what_battery_thesis_proves.tex", 
        "ch35_deployment_path_verification_discipline.tex", 
        "ch23_research_roadmap.tex", 
        "ch36_conclusion.tex",
        "ch24_conclusion.tex"
    ]
}

CHAPTER_TITLES = {
    "ch01_introduction.tex": "Introduction and Problem Formulation",
    "ch02_related_work.tex": "Related Work",
    "ch03_background.tex": "System Context and Safety Distinctions",
    "ch04_theoretical_foundations.tex": "ORIUS Framework and Theoretical Foundations",
    "ch05_architecture_setup.tex": "System Deployment and Testbench Architecture",
    "ch06_main_results.tex": "Comprehensive Battery Fault and Performance Results",
    "ch07_universal_validation.tex": "Universal Defense: AV, Industrial, and Healthcare",
    "ch08_conclusion.tex": "Deployment Roadmap and Conclusion"
}

SRC_DIR = Path("chapters")
OUT_DIR = Path("chapters_merged")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def strip_padding(content: str) -> str:
    # Remove simple padding phrases
    pad_phrases = [
        r"This chapter demonstrates[^\.]*\.",
        r"This chapter shows[^\.]*\.",
        r"This chapter investigates[^\.]*\.",
        r"This chapter answers[^\.]*\.",
        r"In this chapter,[^\.]*\.",
    ]
    for phrase in pad_phrases:
        content = re.sub(phrase, "", content, flags=re.IGNORECASE)
    
    # Remove entire "Chapter Summary" sections or Demote them
    # For now, let's just demote "Chapter Summary" to a paragraph or remove it completely to save space.
    content = re.sub(r"\\section\{Chapter Summary\}.*?(?=\\(section|chapter|$))", "", content, flags=re.IGNORECASE | re.DOTALL)
    
    return content

def downgrade_headings(content: str) -> str:
    content = content.replace(r"\subsubsection{", r"\paragraph{")
    content = content.replace(r"\subsection{", r"\subsubsection{")
    content = content.replace(r"\section{", r"\subsection{")
    content = content.replace(r"\chapter{", r"\section{")
    return content

def process_group(out_filename: str, input_files: list[str], group_title: str):
    merged_text = f"\\chapter{{{group_title}}}\n\\label{{ch:{out_filename.split('.')[0]}}}\n\n"
    
    for i, file in enumerate(input_files):
        in_path = SRC_DIR / file
        if not in_path.exists():
            print(f"Warning: {in_path} not found. Skipping.")
            continue
        
        with open(in_path, "r") as f:
            content = f.read()
            
            # Clean padding
            content = strip_padding(content)

            # If it's not the first file, or if we want to treat all imported files as sub-sections
            # since we just added the top-level \chapter for the group:
            
            # Find the original \chapter{} tag to convert it into a \section{}
            # and downgrade everything else.
            content = downgrade_headings(content)
            
            merged_text += f"% --- Merged from {file} ---\n{content}\n\n"
            
    with open(OUT_DIR / out_filename, "w") as f:
        f.write(merged_text)
    print(f"Wrote {OUT_DIR / out_filename}")

for out_file, in_files in GROUPS.items():
    process_group(out_file, in_files, CHAPTER_TITLES[out_file])

# Now update the main file
MAIN_FILE = Path("orius_battery_409page_figures_upgraded_main.tex")
if MAIN_FILE.exists():
    with open(MAIN_FILE, "r") as f:
        main_content = f.read()
    
    # Replace all \include{chapters/...} with our new list
    new_includes = "\n".join([f"\\include{{chapters_merged/{k.split('.')[0]}}}" for k in GROUPS.keys()])
    
    # regex to find the block of chapters
    # We will just find all \include{chapters/ch...} and replace the entire contiguous block
    # or just replace every \include{chapters/...} with empty, and insert the new ones where the first one was.
    import re
    # Replace the \include{chapters/...} block. We'll use a function to only insert new_includes on the first match
    first_match_done = False
    def replacer(m):
        global first_match_done
        if not first_match_done:
            first_match_done = True
            return new_includes + "\n"
        return ""
    
    main_content = re.sub(r"\\include\{chapters/[^\}]+\}\n?", replacer, main_content)
    
    # Write to new main file
    with open("thesis_final_main.tex", "w") as f:
        f.write(main_content)
    print("Wrote thesis_final_main.tex")
