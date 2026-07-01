import logging
from pathlib import Path

logger = logging.getLogger("leaf")


class SkillEvolver:
    """
    Skill guidance provider for the LEAF agent.

    Users can provide their own skill file (a Markdown file with domain-specific
    guidance) by setting `skill_file` in config.yaml or via CLI:
        python -m leaf.run skill_file=/path/to/my_skill.md ...

    If no skill file is provided, the agent runs in free exploration mode.
    """

    def __init__(self, cfg, task_desc):
        self.cfg = cfg
        self.task_desc = task_desc
        self.skill_file = getattr(cfg, 'skill_file', None)
        self.sota_skill = "No skill file provided. Free exploration mode."

        # Load user-provided skill file if available
        if self.skill_file:
            skill_path = Path(self.skill_file)
            if skill_path.exists():
                try:
                    with open(skill_path, 'r', encoding='utf-8') as f:
                        self.sota_skill = f.read()
                    logger.info(f"Loaded skill file: {skill_path}")
                except Exception as e:
                    logger.warning(f"Failed to load skill file {skill_path}: {e}")
            else:
                logger.warning(f"Skill file not found: {skill_path}")

    def get_skill_guidance(self, task_desc=None, global_step=None):
        """Return skill guidance text."""
        if self.skill_file:
            logger.info(f"Using skill file: {self.skill_file}")
        else:
            logger.info("No skill file loaded, free exploration mode.")
        return self.sota_skill
