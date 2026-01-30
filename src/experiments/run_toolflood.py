#!/usr/bin/env python3
"""
Experiment script for ToolFlood attack evaluation.

This script:
1. Runs the attack to generate attacker tools
2. Merges attacker tools with benign tools
3. Creates a new vector store from merged tools
4. Evaluates queries on the victim agent
5. Calculates metrics: ASR (Attack Success Rate), TDR
   (Top-k Domination Rate), and Mean Domination
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# Add repo root to path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# pylint: disable=wrong-import-position
from loguru import logger  # noqa: E402

from src.experiments.common import (  # noqa: E402
    evaluate_queries,
    get_completed_combinations,
    load_existing_results,
    merge_tools,
    write_results_to_disk,
)
from src.agent import VictimAgent  # noqa: E402
from src.attacks.toolflood_attack import AttackConfig as ToolFloodAttackConfigClass, ToolFloodAttack  # noqa: E402
from src.metrics import (  # noqa: E402
    calculate_asr,
    calculate_tdr,
    calculate_mean_domination,
)
from src.utils import (  # noqa: E402
    Tool,
    get_base_path,
    init_embedding_model,
    init_llm,
    load_agent_config,
    load_config,
    load_experiment_config,
    load_queries_from_tasks,
    load_tools,
    load_toolflood_config,
    load_vector_store,
    resolve_path,
)
from src.scripts.build_vectorstore import init_vector_store  # noqa: E402


def generate_attacker_tools_for_domain(
    task_name: Optional[str],
    cfg: Dict,
    exp_cfg,
    attack_cfg,
    benign_data_dir: Path,
    attack_embedding_model,
    attack_embedding_model_name: str,
) -> tuple[List[Tool], List[str], List[str], str, Dict[str, int]]:
    """Generate attacker tools for a task.

    Returns tools, train/test queries, task_str, and phase2_coverage.
    """
    task_list = [task_name] if task_name else None
    task_str = task_name if task_name else "all"

    logger.info(f"Generating attacker tools for task: {task_str}")

    # Load queries from tasks folder
    train_queries, test_queries = load_queries_from_tasks(
        benign_data_dir / "tasks",
        task_names=task_list
    )

    # Apply limits if specified
    if exp_cfg.max_train_queries:
        train_queries = train_queries[:exp_cfg.max_train_queries]

    if exp_cfg.max_test_queries:
        test_queries = test_queries[:exp_cfg.max_test_queries]

    logger.info(
        f"Task {task_str}: {len(train_queries)} train, "
        f"{len(test_queries)} test queries"
    )

    # Generate attacker tools
    llm_optimizer = init_llm(
        cfg, model_name=attack_cfg.llm_optimizer_model
    )
    attack_config = ToolFloodAttackConfigClass(
        num_tools_per_query=attack_cfg.num_tools_per_query,
        query_sample_size=attack_cfg.query_sample_size,
        num_tools_per_sample=attack_cfg.num_tools_per_sample,
        max_generation_iterations=attack_cfg.max_generation_iterations,
        max_embedding_distance=attack_cfg.max_embedding_distance,
        total_tool_budget=attack_cfg.total_tool_budget,
        max_concurrent_tasks=attack_cfg.max_concurrent_tasks,
    )
    attack = ToolFloodAttack(
        train_queries,
        attack_embedding_model,
        llm_optimizer,
        attack_config=attack_config,
    )

    attacker_tools, attack_results = attack.attack()

    # Extract phase 2 query coverage for each tool (phase2 is dict with "iterations" list)
    phase2_coverage = {}
    phase2_data = attack_results.get("phase2", {})
    iterations = phase2_data.get("iterations", [])
    for tool, iter_info in zip(attacker_tools, iterations):
        phase2_coverage[tool.name] = iter_info.get("coverage", 0)

    return attacker_tools, train_queries, test_queries, task_str, phase2_coverage


def save_attacker_tools(
    merged_tools: List[Tool],
    attacker_tool_names: set[str],
    output_path: Path,
    phase2_coverage: Optional[Dict[str, int]] = None
) -> None:
    """Save attacker tools to JSON file.
    
    Args:
        merged_tools: List of all merged tools (benign + attacker)
        attacker_tool_names: Set of attacker tool names (after merging/renaming)
        output_path: Path to save the JSON file
        phase2_coverage: Optional dict mapping tool name to phase 2 query coverage count
    """
    # Filter to only attacker tools
    attacker_tools = [
        tool for tool in merged_tools
        if tool.name in attacker_tool_names
    ]
    
    logger.info(f"Saving {len(attacker_tools)} attacker tools...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build tools dict with phase 2 coverage info
    if phase2_coverage:
        tools_dict = {}
        for tool in attacker_tools:
            # Use original name for phase2_coverage lookup if renamed
            original_name = tool.name
            coverage_count = phase2_coverage.get(original_name, 0)
            tools_dict[tool.name] = {
                "description": tool.description,
                "phase2_query_coverage": coverage_count
            }
    else:
        # Fallback to simple format if phase2_coverage not provided
        tools_dict = {tool.name: tool.description for tool in attacker_tools}
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tools_dict, f, indent=2)
    logger.success(f"Attacker tools saved to: {output_path}")


def prepare_vector_store_for_embedding(
    task_str: str,
    attacker_tools: List[Tool],
    experiment_dir: Path,
    victim_embedding_model,
    victim_embedding_model_name: str,
    attack_embedding_model_name: str,
    benign_tools: List[Tool],
    phase2_coverage: Optional[Dict[str, int]] = None,
) -> tuple[Path, List[Tool], set[str]]:
    """Prepare vector store and merged tools for an embedding combination.
    
    This is done once per embedding combination and reused across all models.
    Returns (vectorstore_path, merged_tools, attacker_tool_names).
    """
    # Merge tools - use unique path per embedding model combination
    task_experiment_dir = (
        experiment_dir / task_str if task_str != "all" else experiment_dir
    )
    embedding_subdir = (
        f"attack_emb_{attack_embedding_model_name}_"
        f"victim_emb_{victim_embedding_model_name}"
    )
    task_experiment_dir = task_experiment_dir / embedding_subdir
    merged_tools_path = task_experiment_dir / "merged_tools.json"
    merged_tools, attacker_tool_names = merge_tools(
        benign_tools, attacker_tools, merged_tools_path
    )
    
    # Save attacker tools separately (from merged tools to match num_injected_tools count)
    attacker_tools_path = task_experiment_dir / "attack_tools.json"
    save_attacker_tools(merged_tools, attacker_tool_names, attacker_tools_path, phase2_coverage)

    # Build vector store (only once per embedding combination)
    vectorstore_path = task_experiment_dir / "vectorstore"
    init_vector_store(
        merged_tools,
        victim_embedding_model,
        vectorstore_path,
        force_rebuild=False  # Reuse existing if it exists
    )
    
    return vectorstore_path, merged_tools, attacker_tool_names


def run_experiment_for_model(
    model_name: str,
    task_str: str,
    train_queries: List[str],
    test_queries: List[str],
    cfg: Dict,
    exp_cfg,
    agent_cfg,
    victim_embedding_model,
    victim_embedding_model_name: str,
    attack_embedding_model_name: str,
    vectorstore_path: Path,
    merged_tools: List[Tool],
    attacker_tool_names: set[str],
) -> Dict:
    """Run experiment for a specific model.

    Uses pre-generated attacker tools and pre-built vector store.
    """
    logger.info(
        f"Evaluating model: {model_name} with embedding: "
        f"{victim_embedding_model_name} for task: {task_str}"
    )

    # Load existing vector store (shared across models with same embedding)
    merged_vectorstore = load_vector_store(
        vectorstore_path, victim_embedding_model
    )
    llm = init_llm(cfg, model_name=model_name)
    agent = VictimAgent(
        tools=merged_tools,
        vectorstore=merged_vectorstore,
        embedding_model=victim_embedding_model,
        llm=llm,
        top_k=agent_cfg.top_k,
        verbose=False,
    )

    # Evaluate train queries
    train_eval_queries = (
        random.sample(
            train_queries,
            min(exp_cfg.max_train_evaluation_queries, len(train_queries))
        )
        if exp_cfg.max_train_evaluation_queries
        else train_queries
    )
    train_results = asyncio.run(
        evaluate_queries(agent, train_eval_queries, attacker_tool_names)
    )
    train_asr = calculate_asr(
        train_results["successful_selections"],
        train_results["attacker_selections"],
    )
    train_tdr = calculate_tdr(
        train_results["top_k_tools"],
        attacker_tool_names,
        agent_cfg.top_k,
    )
    train_mean_domination = calculate_mean_domination(
        train_results["top_k_tools"],
        attacker_tool_names,
        agent_cfg.top_k,
    )

    # Evaluate test queries
    test_results = asyncio.run(
        evaluate_queries(agent, test_queries, attacker_tool_names)
    )
    test_asr = calculate_asr(
        test_results["successful_selections"],
        test_results["attacker_selections"],
    )
    test_tdr = calculate_tdr(
        test_results["top_k_tools"],
        attacker_tool_names,
        agent_cfg.top_k,
    )
    test_mean_domination = calculate_mean_domination(
        test_results["top_k_tools"],
        attacker_tool_names,
        agent_cfg.top_k,
    )

    return {
        "domain": task_str,
        "model": model_name,
        "attack_embedding_model": attack_embedding_model_name,
        "victim_embedding_model": victim_embedding_model_name,
        "train_stats": {
            "total_queries": train_results["total_queries"],
            "successful_selections": train_results["successful_selections"],
            "attacker_selections": train_results["attacker_selections"],
            "benign_selections": train_results["benign_selections"],
            "no_selection": train_results["no_selection"],
            "asr": train_asr,
            "tdr": train_tdr,
            "mean_domination": train_mean_domination,
        },
        "test_stats": {
            "total_queries": test_results["total_queries"],
            "successful_selections": test_results["successful_selections"],
            "attacker_selections": test_results["attacker_selections"],
            "benign_selections": test_results["benign_selections"],
            "no_selection": test_results["no_selection"],
            "asr": test_asr,
            "tdr": test_tdr,
            "mean_domination": test_mean_domination,
        },
        "num_injected_tools": len(attacker_tool_names),
        "train_query_results": train_results["query_results"],
        "test_query_results": test_results["query_results"],
    }


def update_results_table(
    all_results: List[Dict],
    experiment_dir: Path,
    exp_cfg,
    agent_cfg,
    benign_tools: List[Tool],
) -> pd.DataFrame:
    """Create or update results table from all results."""
    benchmark_name = Path(exp_cfg.benign_data_directory).name

    rows = []
    for result in all_results:
        rows.append({
            "Split": "Train",
            "Model": result["model"],
            "Attack_Embedding_Model": result.get(
                "attack_embedding_model", "N/A"
            ),
            "Victim_Embedding_Model": result.get(
                "victim_embedding_model", "N/A"
            ),
            "Benchmark": benchmark_name,
            "Scenario": result["domain"],
            "Num_Injected_Tools": result["num_injected_tools"],
            "Benign_Tools": len(benign_tools),
            "Total_Queries": result["train_stats"]["total_queries"],
            "ASR": f"{result['train_stats']['asr']:.4f}",
            "TDR": f"{result['train_stats']['tdr']:.4f}",
            "Mean_Domination": (
                f"{result['train_stats']['mean_domination']:.4f}"
            ),
            "Top_K": agent_cfg.top_k,
        })
        rows.append({
            "Split": "Test",
            "Model": result["model"],
            "Attack_Embedding_Model": result.get(
                "attack_embedding_model", "N/A"
            ),
            "Victim_Embedding_Model": result.get(
                "victim_embedding_model", "N/A"
            ),
            "Benchmark": benchmark_name,
            "Scenario": result["domain"],
            "Num_Injected_Tools": result["num_injected_tools"],
            "Benign_Tools": len(benign_tools),
            "Total_Queries": result["test_stats"]["total_queries"],
            "ASR": f"{result['test_stats']['asr']:.4f}",
            "TDR": f"{result['test_stats']['tdr']:.4f}",
            "Mean_Domination": (
                f"{result['test_stats']['mean_domination']:.4f}"
            ),
            "Top_K": agent_cfg.top_k,
        })

    return pd.DataFrame(rows)


def save_results(
    all_results: List[Dict],
    experiment_dir: Path,
    exp_cfg,
    agent_cfg,
    benign_tools: List[Tool],
    results_json_path: Path,
    results_table_path: Path,
) -> None:
    """Save results to JSON and update CSV table."""
    results_table = update_results_table(
        all_results, experiment_dir, exp_cfg, agent_cfg, benign_tools
    )
    write_results_to_disk(
        all_results, results_json_path, results_table, results_table_path
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run ToolFlood attack experiment and calculate ASR"
    )
    ap.add_argument("--config", required=True, help="Path to config YAML")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    exp_cfg = load_experiment_config(cfg_path)
    attack_cfg = load_toolflood_config(cfg_path)
    agent_cfg = load_agent_config(cfg_path)

    base_path = get_base_path(cfg_path)
    benign_data_dir = resolve_path(base_path, exp_cfg.benign_data_directory)
    experiment_dir = resolve_path(base_path, exp_cfg.output_directory)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Paths for results files
    results_json_path = experiment_dir / "results_full.json"
    results_table_path = experiment_dir / "results_table.csv"

    # Load existing results if not doing hard reset
    if exp_cfg.hard_reset:
        logger.info("Hard reset enabled: starting fresh")
        all_results = []
    else:
        all_results = load_existing_results(results_json_path)

    # Track completed combinations
    completed_combinations = get_completed_combinations(
        all_results,
        ["domain", "model", "attack_embedding_model", "victim_embedding_model"],
    )
    if completed_combinations:
        logger.info(
            f"Found {len(completed_combinations)} completed experiment "
            "combinations. Will skip these."
        )

    # Initialize embedding models
    attack_embedding_models = {}
    victim_embedding_models = {}

    for emb_model_name in exp_cfg.attack_embedding_models:
        attack_embedding_models[emb_model_name] = init_embedding_model(
            cfg, model_name=emb_model_name
        )

    for emb_model_name in exp_cfg.victim_embedding_models:
        victim_embedding_models[emb_model_name] = init_embedding_model(
            cfg, model_name=emb_model_name
        )

    benign_tools = load_tools(benign_data_dir / "tools.json")

    # Determine tasks and models to process
    tasks = (
        exp_cfg.task_names
        if exp_cfg.task_names
        else [None]
    )
    victim_models = exp_cfg.victim_models

    # Run experiments: generate attacker tools per task and attack embedding
    # model, test all victim models and victim embedding model combinations
    total_combinations = (
        len(tasks) * len(exp_cfg.attack_embedding_models) *
        len(victim_models) * len(exp_cfg.victim_embedding_models)
    )
    completed_count = len(completed_combinations)
    logger.info(
        f"Total combinations to process: {total_combinations} "
        f"({completed_count} already completed)"
    )

    for task in tasks:
        # Get task_str early to check if we need to process this task
        task_str = task if task else "all"
        
        for attack_emb_model_name in exp_cfg.attack_embedding_models:
            attack_embedding_model = (
                attack_embedding_models[attack_emb_model_name]
            )

            # Check if there are any remaining combinations for this task/attack_emb pair
            has_remaining_combinations = False
            for victim_emb_model_name in exp_cfg.victim_embedding_models:
                for model_name in victim_models:
                    combination_key = (
                        task_str, model_name,
                        attack_emb_model_name, victim_emb_model_name
                    )
                    if combination_key not in completed_combinations:
                        has_remaining_combinations = True
                        break
                if has_remaining_combinations:
                    break
            
            # Skip generating attacker tools if all combinations are already done
            if not has_remaining_combinations:
                logger.info(
                    f"Skipping task '{task_str}' with attack_emb '{attack_emb_model_name}': "
                    "all combinations already completed"
                )
                continue

            # Generate attacker tools once per task and attack embedding model
            attacker_tools, train_queries, test_queries, task_str, phase2_coverage = (
                generate_attacker_tools_for_domain(
                    task,
                    cfg,
                    exp_cfg,
                    attack_cfg,
                    benign_data_dir,
                    attack_embedding_model,
                    attack_emb_model_name,
                )
            )

            # Test all victim models and victim embedding model combinations
            # Build vector store once per victim embedding model (shared across all victim LLM models)
            for victim_emb_model_name in exp_cfg.victim_embedding_models:
                victim_embedding_model = (
                    victim_embedding_models[victim_emb_model_name]
                )
                
                # Prepare vector store once per embedding combination
                # This will be reused across all victim LLM models with the same embedding
                vectorstore_path, merged_tools, attacker_tool_names = (
                    prepare_vector_store_for_embedding(
                        task_str,
                        attacker_tools,
                        experiment_dir,
                        victim_embedding_model,
                        victim_emb_model_name,
                        attack_emb_model_name,
                        benign_tools,
                        phase2_coverage,
                    )
                )
                
                # Test all victim LLM models with this embedding model
                for model_name in victim_models:
                    # Check if this combination is already completed
                    combination_key = (
                        task_str,
                        model_name,
                        attack_emb_model_name,
                        victim_emb_model_name,
                    )
                    if combination_key in completed_combinations:
                        logger.info(
                            f"Skipping already completed: {task_str} / "
                            f"{model_name} / attack_emb: "
                            f"{attack_emb_model_name} / victim_emb: "
                            f"{victim_emb_model_name}"
                        )
                        continue

                    logger.info(
                        f"Running experiment: {task_str} / {model_name} / "
                        f"attack_emb: {attack_emb_model_name} / "
                        f"victim_emb: {victim_emb_model_name} "
                        f"({completed_count + 1}/{total_combinations})"
                    )

                    try:
                        result = run_experiment_for_model(
                            model_name,
                            task_str,
                            train_queries,
                            test_queries,
                            cfg,
                            exp_cfg,
                            agent_cfg,
                            victim_embedding_model,
                            victim_emb_model_name,
                            attack_emb_model_name,
                            vectorstore_path,
                            merged_tools,
                            attacker_tool_names,
                        )
                        all_results.append(result)
                        completed_count += 1

                        # Save results incrementally after each experiment
                        save_results(
                            all_results,
                            experiment_dir,
                            exp_cfg,
                            agent_cfg,
                            benign_tools,
                            results_json_path,
                            results_table_path,
                        )
                        logger.success(
                            f"Completed: {task_str} / {model_name} / "
                            f"attack_emb: {attack_emb_model_name} / "
                            f"victim_emb: {victim_emb_model_name} "
                            f"({completed_count}/{total_combinations})"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error running experiment for "
                            f"{task_str} / {model_name} / "
                            f"attack_emb: {attack_emb_model_name} / "
                            f"victim_emb: {victim_emb_model_name}: {e}"
                        )
                        logger.error("Results saved up to this point.")
                        raise

    # Final save and summary
    if all_results:
        save_results(
            all_results,
            experiment_dir,
            exp_cfg,
            agent_cfg,
            benign_tools,
            results_json_path,
            results_table_path,
        )

        # Print summary
        results_table = pd.read_csv(results_table_path)
        logger.success("\n" + "="*100)
        logger.success("RESULTS TABLE")
        logger.success("="*100)
        print("\n" + results_table.to_string(index=False))
        logger.success("="*100)
        logger.success(f"\nResults table saved to: {results_table_path}")
    else:
        logger.warning("No results to display.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
