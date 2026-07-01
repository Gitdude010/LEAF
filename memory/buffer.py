import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Union
from leaf.journal import Node
from leaf.backend import query, FunctionSpec
import re
import logging
import threading
from pathlib import Path

logger = logging.getLogger("leaf")


class AgentMemoryManager:
    """
    三层记忆缓冲管理器:
    L1: JSON Buffer (原始节点数据) → buffer_limit 触发压缩
    L2: Stage Summary Queue (阶段性总结 FIFO) → stage_limit 触发精炼
    L3: Refined Strategy (精炼策略库, overwrite) → 读取时按重要性分层
    """

    def __init__(self, cfg, buffer_limit: int = 4, stage_limit: int = 5):
        self.buffer_limit = buffer_limit  # L1 上限
        self.stage_limit = stage_limit    # L2 上限，达到后触发 L3 精炼

        base_dir = Path(cfg.workspace_dir) / "memory"
        base_dir.mkdir(parents=True, exist_ok=True)

        # L1: 原始节点缓冲
        self.improve_json = base_dir / "improve_buffer.json"
        self.debug_json = base_dir / "debug_buffer.json"

        # L2: 阶段性总结 FIFO 队列
        self.improve_stage = base_dir / "improve_stage.json"
        self.debug_stage = base_dir / "debug_stage.json"

        # L3: 精炼策略库 (overwrite)
        self.improve_refined = base_dir / "improve_refined.md"
        self.debug_refined = base_dir / "debug_refined.md"

        self.cfg = cfg
        self.acfg = cfg.agent
        self.buffer_lock = threading.Lock()

        # L1 → L2 压缩用的 Function Schema (不变)
        self.improve_func_spec = FunctionSpec(
            name="record_improve_summary",
            description="Record a staged experience summary for Model/Code Optimization (Improve) tasks.",
            json_schema={
                "type": "object",
                "properties": {
                    "positive_guidance": {
                        "type": "string",
                        "description": "Effective Optimization Strategies: Extract proven methodologies that led to metric improvement. (CRITICAL: Limit to 3-5 sentences.)"
                    },
                    "negative_constraints": {
                        "type": "string",
                        "description": "Ineffective/Negative Patterns: Actions that dropped or froze the metric. (CRITICAL: Limit to 3-5 sentences.)"
                    },
                    "synergy_observations": {
                        "type": "string",
                        "description": "Potential Synergies: Effects produced by combining different tricks. (CRITICAL: Limit to 3-5 sentences.)"
                    },
                    "next_steps": {
                        "type": "string",
                        "description": "Next Exploration Steps: Recommended code directions. (CRITICAL: Limit to 3-5 sentences.)"
                    }
                },
                "required": ["positive_guidance", "negative_constraints", "synergy_observations", "next_steps"]
            }
        )

        self.debug_func_spec = FunctionSpec(
            name="record_debug_summary",
            description="Record a staged experience summary for Code Repair (Debug) tasks.",
            json_schema={
                "type": "object",
                "properties": {
                    "frequent_error_patterns": {
                        "type": "string",
                        "description": "Frequent Error Patterns: Root causes of errors. (CRITICAL: Limit to 3-5 sentences.)"
                    },
                    "standard_fixes": {
                        "type": "string",
                        "description": "Standard Fixes: Solutions that resolved the errors. (CRITICAL: Limit to 3-5 sentences.)"
                    },
                    "defensive_coding_rules": {
                        "type": "string",
                        "description": "Defensive Coding Rules: Standards preventing future bugs. (CRITICAL: Limit to 3-5 sentences.)"
                    }
                },
                "required": ["frequent_error_patterns", "standard_fixes", "defensive_coding_rules"]
            }
        )

        self._init_files()

    def _init_files(self):
        """初始化所有层级的文件"""
        # L1: 原始缓冲
        for file in [self.improve_json, self.debug_json]:
            if not os.path.exists(file):
                with open(file, 'w', encoding='utf-8') as f:
                    json.dump([], f)

        # L2: 阶段总结队列
        for file in [self.improve_stage, self.debug_stage]:
            if not os.path.exists(file):
                with open(file, 'w', encoding='utf-8') as f:
                    json.dump([], f)

        # L3: 精炼策略库
        for file in [self.improve_refined, self.debug_refined]:
            if not os.path.exists(file):
                with open(file, 'w', encoding='utf-8') as f:
                    f.write("")  # 初始为空

    # =========================================================================
    # L1: 节点写入与压缩
    # =========================================================================

    def save_buffer_safely(self, node: Node) -> dict:
        """安全的缓冲池序列化，手动提取字段杜绝循环引用"""
        safe_dict = {
            "id": str(node.id),
            "plan": getattr(node, "plan", ""),
            "diff_patch": getattr(node, "diff_patch", ""),
            "node_type": getattr(node, "node_type", ""),
            "is_buggy": getattr(node, "is_buggy", False),
            "exc_type": getattr(node, "exc_type", ""),
            "analysis": getattr(node, "analysis", ""),
            "report": getattr(node, "report", ""),
            "metric": node.metric.value if (hasattr(node, "metric") and node.metric) else None,
            "parent_improve_metric": getattr(node, "parent_improve_metric", None),
            "parent_id": node.parent.id if getattr(node, "parent", None) else None,
        }

        if safe_dict["node_type"] == "debug":
            safe_dict["term_out"] = "\n".join(getattr(node, "_term_out", [])[-30:]) if hasattr(node, "_term_out") else ""

        return safe_dict

    def _get_hybrid_improve_dict(self, debug_node: Node) -> dict:
        """Debug 成功时追溯源头 improve 意图，融合为成功的 improve 节点"""
        curr = debug_node.parent
        original_improve_node = None

        while curr:
            node_type = getattr(curr, "node_type", "")
            if node_type in ["improve", "draft", "merge"]:
                original_improve_node = curr
                break
            curr = getattr(curr, "parent", None)

        original_plan = getattr(original_improve_node, "plan", "") if original_improve_node else ""
        parent_metric = getattr(original_improve_node, "parent_improve_metric", None) if original_improve_node else None
        is_baseline_establishment = (parent_metric is None)

        return {
            "id": str(debug_node.id),
            "plan": f"[Original Strategy]: {original_plan}\n[Applied Fixes]: {getattr(debug_node, 'plan', '')}",
            "node_type": "improve",
            "diff_patch": getattr(debug_node, "diff_patch", ""),
            "is_buggy": False,
            "exc_type": "",
            "analysis": getattr(debug_node, "analysis", ""),
            "report": getattr(debug_node, "report", ""),
            "metric": debug_node.metric.value if (hasattr(debug_node, "metric") and debug_node.metric) else None,
            "parent_improve_metric": getattr(original_improve_node, "parent_improve_metric", None) if original_improve_node else None,
            "is_baseline_establishment": is_baseline_establishment,
            "parent_id": original_improve_node.id if original_improve_node else (debug_node.parent.id if getattr(debug_node, "parent", None) else None),
        }

    def add_node(self, node: Node):
        """将新节点加入对应的缓冲池"""
        if node.node_type not in ['improve', 'debug']:
            raise ValueError("node_type 必须是 'improve' 或 'debug'")

        tasks = []

        if node.node_type == 'improve':
            if getattr(node, 'is_buggy', False):
                logger.info(f"[Memory] Improve node {node.id} has bugs. Skipping improve memory insertion.")
            else:
                tasks.append(('improve', self.save_buffer_safely(node)))

        elif node.node_type == 'debug':
            tasks.append(('debug', self.save_buffer_safely(node)))
            if getattr(node, 'is_buggy', False) is False:
                logger.info(f"[Memory] Debug node {node.id} succeeded! Recovering improve intent into memory.")
                hybrid_dict = self._get_hybrid_improve_dict(node)
                tasks.append(('improve', hybrid_dict))

        for buffer_type, data_dict in tasks:
            self._process_buffer_append(buffer_type, data_dict)

    def _process_buffer_append(self, buffer_type: str, data_dict: dict):
        """L1 写入，满时触发压缩到 L2"""
        file_path = self.improve_json if buffer_type == 'improve' else self.debug_json
        needs_compression = False
        buffer_data_to_compress = []

        with self.buffer_lock:
            with open(file_path, 'r', encoding='utf-8') as f:
                buffer: List[Dict] = json.load(f)

            buffer.append(data_dict)

            if len(buffer) >= self.buffer_limit:
                needs_compression = True
                buffer_data_to_compress = list(buffer)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(buffer, f, ensure_ascii=False, indent=2)

        if needs_compression:
            threading.Thread(
                target=self._compress_to_l2,
                args=(buffer_type, buffer_data_to_compress),
                daemon=True
            ).start()

    # =========================================================================
    # L1 → L2: 压缩为阶段性总结
    # =========================================================================

    def _compress_to_l2(self, node_type: str, buffer_data: List[Dict]):
        """调用 LLM 将 L1 原始数据压缩为 Stage Summary，写入 L2 队列"""
        logger.info(f"[{node_type.upper()}] L1 buffer full ({self.buffer_limit} items), compressing to L2...")

        if node_type == 'improve':
            prompt = self._build_improve_prompt(buffer_data)
            current_func_spec = self.improve_func_spec
        else:
            prompt = self._build_debug_prompt(buffer_data)
            current_func_spec = self.debug_func_spec

        last_completion_text = None
        for i in range(3):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
                func_spec=current_func_spec,
                cfg=self.cfg,
            )
            if completion_text:
                last_completion_text = completion_text
                break

        if last_completion_text is None:
            logger.warning(f"[{node_type.upper()}] LLM compression failed after 3 attempts, skipping.")
            return

        # 解析为 dict 后写入 L2
        summary_dict = self._parse_completion_to_dict(last_completion_text)
        if not summary_dict:
            logger.warning(f"[{node_type.upper()}] Failed to parse LLM output, skipping L2 write.")
            return

        # 写入 L2 队列，满时触发 L3 精炼
        self._append_l2_and_check_refine(node_type, summary_dict)

    def _parse_completion_to_dict(self, completion_text: Union[str, dict]) -> Optional[Dict]:
        """将 LLM 返回结果解析为 dict"""
        try:
            if isinstance(completion_text, dict):
                return completion_text
            return json.loads(completion_text)
        except (json.JSONDecodeError, TypeError):
            return None

    def _append_l2_and_check_refine(self, node_type: str, summary_dict: Dict):
        """将 summary 追加到 L2 队列，满时触发 L3 精炼"""
        stage_file = self.improve_stage if node_type == 'improve' else self.debug_stage

        with self.buffer_lock:
            with open(stage_file, 'r', encoding='utf-8') as f:
                stage_queue: List[Dict] = json.load(f)

            stage_queue.append(summary_dict)
            needs_refine = len(stage_queue) >= self.stage_limit

            if needs_refine:
                # 精炼需要用到全部 L2 数据，拷贝后清空 L2
                stage_data_to_refine = list(stage_queue)
                with open(stage_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            else:
                with open(stage_file, 'w', encoding='utf-8') as f:
                    json.dump(stage_queue, f, ensure_ascii=False, indent=2)

        if needs_refine:
            threading.Thread(
                target=self._refine_to_l3,
                args=(node_type, stage_data_to_refine),
                daemon=True
            ).start()

    # =========================================================================
    # L2 → L3: 精炼策略库 (overwrite)
    # =========================================================================

    def _refine_to_l3(self, node_type: str, new_summaries: List[Dict]):
        """
        精炼 L2 summaries 到 L3 策略库。
        读取旧 L3 + 新 L2 → LLM 输出全新 L3 → 覆盖写入。
        """
        logger.info(f"[{node_type.upper()}] L2 stage queue full ({self.stage_limit} items), refining to L3...")

        # 读取旧 L3 内容
        refined_file = self.improve_refined if node_type == 'improve' else self.debug_refined
        with self.buffer_lock:
            if os.path.exists(refined_file):
                with open(refined_file, 'r', encoding='utf-8') as f:
                    old_l3_content = f.read().strip()
            else:
                old_l3_content = ""

        # 构建精炼 prompt
        prompt = self._build_refine_prompt(node_type, old_l3_content, new_summaries)

        # 调用 LLM 精炼
        last_completion_text = None
        for i in range(3):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
                cfg=self.cfg,
            )
            if completion_text:
                last_completion_text = completion_text
                break

        if not last_completion_text:
            logger.warning(f"[{node_type.upper()}] L3 refinement failed after 3 attempts, keeping old L3.")
            # 精炼失败：保留旧 L3，L2 已在上面清空（避免重复精炼同一批数据）
            return

        # 覆盖写入 L3
        with self.buffer_lock:
            with open(refined_file, 'w', encoding='utf-8') as f:
                f.write(last_completion_text.strip() + "\n")

        logger.info(f"[{node_type.upper()}] L3 refined strategy library updated (overwrite).")

    def _build_refine_prompt(self, node_type: str, old_l3: str, new_summaries: List[Dict]) -> str:
        """构建 L3 精炼 prompt"""
        summaries_str = json.dumps(new_summaries, ensure_ascii=False, indent=2)

        if node_type == 'improve':
            domain = "Model/Code Optimization (Improve)"
            old_l3_section = f"""## Current Refined Strategy Library:
{old_l3 if old_l3 else "(Empty — this is the first refinement cycle.)"}"""
        elif node_type == 'debug':
            domain = "Code Repair (Debug)"
            old_l3_section = f"""## Current Refined Strategy Library:
{old_l3 if old_l3 else "(Empty — this is the first refinement cycle.)"}"""

        return f"""You are the Memory Refiner for an ML competition agent's {domain} workflow.
Your job is to merge and distill the current refined strategy library with new stage summaries into a single, clean, deduplicated strategy library.

{old_l3_section}

## New Stage Summaries (recent experiments to integrate):
{summaries_str}

## Your Task:
1. **MERGE**: Integrate new findings into the existing library. If a new strategy duplicates an existing one, increment its verification count and upgrade confidence if appropriate.
2. **CONFLICT RESOLUTION**: If a new summary contradicts an existing strategy, evaluate based on metric evidence. Keep the one with stronger quantitative support. If both have merit under different conditions, note the conditions.
3. **FORGET**: Remove strategies that:
   - Have "[Low Confidence]" and no new supporting evidence in the latest summaries.
   - Were suspected pitfalls but new evidence shows they were false alarms.
4. **UPGRADE/DOWNGRADE confidence levels**:
   - Low → Medium: verified 2+ times total across all summaries
   - Medium → High: verified 3+ times with consistent metric improvement
   - High → Medium: latest experiment showed no improvement or contradictory results
5. **DEDUPLICATE**: Merge near-identical strategies. Remove vague statements lacking quantitative evidence.

## Output Format (MUST follow exactly):

# Refined Strategy Library

## ✅ Positive Strategies
### [High Confidence]
- Strategy description with quantitative effect (verified: N)

### [Medium Confidence]
- Strategy description with quantitative effect (verified: N)

### [Low Confidence]
- Strategy description with quantitative effect (verified: N)

## ❌ Negative Constraints
### [Confirmed Pitfalls]
- Pitfall description with consequence

### [Suspected Pitfalls]
- Pitfall description with consequence

## CRITICAL Rules:
- Output the FULL updated library, not just changes.
- Keep each strategy to 1-2 sentences. Be concise.
- Always cite specific metric changes when available (e.g., "AUC 0.7419 → 0.7551").
- Do NOT include vague strategies or generic ML advice. Only include findings grounded in the data above.
- If a section has no entries, write "(None)" under it.
- Do NOT add any "Next Exploration Steps" or suggestions section. Only refine what has been observed."""

    # =========================================================================
    # 读取: 三层组装 (按重要性分层)
    # =========================================================================

    def get_memory_context(self, tag: str) -> str:
        """
        提取三层记忆上下文，按重要性从高到低排列:
        L3 (精炼策略) > L2 (近期总结) > L1 (原始缓冲)
        """
        if tag not in ['improve', 'debug']:
            raise ValueError("提取标签必须是 'improve' 或 'debug'")

        context_lines = []

        # =================================================================
        # L3: 精炼策略库 (最高优先级 — 铁律，必须遵守)
        # =================================================================
        refined_file = self.improve_refined if tag == 'improve' else self.debug_refined
        context_lines.append(f"### 🔥 {tag.upper()} Refined Strategy Library (HIGHEST PRIORITY)")
        context_lines.append("These strategies have been distilled from ALL past experiments. They are verified, deduplicated, and ranked by confidence.")
        context_lines.append("You MUST follow Positive Strategies and MUST avoid Negative Constraints.")

        if os.path.exists(refined_file):
            with open(refined_file, 'r', encoding='utf-8') as f:
                l3_content = f.read().strip()
                if l3_content:
                    context_lines.append(l3_content)
                else:
                    context_lines.append("(No refined strategies yet.)")
        else:
            context_lines.append("(No refined strategies yet.)")

        context_lines.append("")

        # =================================================================
        # L2: 近期阶段总结 (中等优先级 — 有参考价值但未经充分验证)
        # =================================================================
        stage_file = self.improve_stage if tag == 'improve' else self.debug_stage
        context_lines.append(f"### 📊 {tag.upper()} Recent Stage Summaries (MEDIUM PRIORITY)")
        context_lines.append("These are recent compressed summaries. They provide useful context but have NOT been cross-validated yet. Use them as reference, not as rules.")

        with self.buffer_lock:
            if os.path.exists(stage_file):
                with open(stage_file, 'r', encoding='utf-8') as f:
                    try:
                        stage_data = json.load(f)
                        if stage_data:
                            for idx, summary in enumerate(stage_data):
                                context_lines.append(f"\n--- Stage Summary {idx + 1} ---")
                                context_lines.append(json.dumps(summary, ensure_ascii=False, indent=2))
                        else:
                            context_lines.append("(No recent stage summaries.)")
                    except json.JSONDecodeError:
                        context_lines.append("(Error reading stage data.)")
            else:
                context_lines.append("(No recent stage summaries.)")

        context_lines.append("")

        # =================================================================
        # L1: 原始节点缓冲 (最低优先级 — 最新但未压缩的原始数据)
        # =================================================================
        buffer_file = self.improve_json if tag == 'improve' else self.debug_json
        context_lines.append(f"### 📝 {tag.upper()} Recent Raw Attempts (LOWEST PRIORITY)")
        context_lines.append("These are the most recent raw attempts that haven't been compressed yet. They are noisy and unverified — treat them as raw signals only.")

        with self.buffer_lock:
            if os.path.exists(buffer_file):
                with open(buffer_file, 'r', encoding='utf-8') as f:
                    try:
                        buffer_data = json.load(f)
                        if buffer_data:
                            context_lines.append(json.dumps(buffer_data, ensure_ascii=False, indent=2))
                        else:
                            context_lines.append("(No recent raw attempts.)")
                    except json.JSONDecodeError:
                        context_lines.append("(Error reading buffer data.)")
            else:
                context_lines.append("(No recent raw attempts.)")

        return "\n".join(context_lines)

    # =========================================================================
    # L1 压缩用的 Prompt (保持不变)
    # =========================================================================

    def _build_improve_prompt(self, nodes: List[Dict]) -> str:
        nodes_str = json.dumps(nodes, ensure_ascii=False, indent=2)
        return f"""
You are a top-tier Machine Learning Expert and AI Architect. Below are the detailed records of {len(nodes)} nodes generated by an AI Agent during recent "Model/Code Optimization (Improve)" iterations.

Your task is to act as the "Long-term Memory & Search Policy Engine" for the Agent. Analyze these attempts and call the provided function to generate a highly concise, quantitative, and instructive summary.

[Node Data]
{nodes_str}

[Analysis Requirements & Constraints]
1. Quantitative Traceability (CRITICAL):
   - NEVER use vague qualitative words like "moderate", "slightly better", or "large".
   - You MUST cite specific metric changes (e.g., "AUC 0.7419 -> 0.7551"), exact parameter values, and specific Node IDs.

2. Performance Comparison & Hybrid Evaluation:
   - Compare the `metric` of each node against its `parent_improve_metric`.
   - Note: Some plans contain `[Original Strategy]` and `[Applied Fixes]`. Evaluate the *combined* algorithmic intent, not the bugs.
   - BASELINE EXCEPTION: If `is_baseline_establishment` is true (or `parent_improve_metric` is null), treat this as "Initial Baseline Establishment". Document the core architecture.

3. Algorithmic Focus for Pitfalls (Negative Constraints):
   - Only list fundamental ML strategy failures. DO NOT list basic Python syntax errors.

4. Method Induction & Synergy:
   - Extract proven combinations. Observe synergy benefits.

5. Strategic Next Steps:
   - Propose 2-4 concrete, incremental ML experiments based strictly on current successful baselines.
"""

    def _build_debug_prompt(self, nodes: List[Dict]) -> str:
        nodes_str = json.dumps(nodes, ensure_ascii=False, indent=2)
        return f"""
You are a top-tier Senior Software Engineer acting as a strict Code Auditor.
Below are the detailed records of {len(nodes)} historical attempts extracted from an AI Agent's Debug Memory Buffer during a "Code Repair" task.

[Memory Buffer Data]
{nodes_str}

[🚨 CRITICAL AUDIT CONSTRAINTS 🚨]
1. ANTI-HALLUCINATION: You MUST base your analysis STRICTLY on the provided Memory Buffer. Do NOT invent or guess.
2. TRUTH OF is_buggy:
   - If `is_buggy = true`, the `plan` and `code` represent a FAILED attempt. Do NOT summarize these as valid fixes.
   - A bug is ONLY "Resolved" if there is a specific node where `is_buggy = false`.

[Analysis Requirements]
1. Root Cause Tracking: Analyze `term_out` to summarize the most frequent errors.
2. Verified Fixes (If Any): Identify nodes where `is_buggy = false` and summarize the EXACT fixes.
3. Failed Attempts: List what DID NOT work to prevent repeating mistakes.
4. Defensive Programming Guidance: Extract coding standards based ONLY on validated fixes and identified failures.
"""
