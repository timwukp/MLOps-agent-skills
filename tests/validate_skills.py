#!/usr/bin/env python3
"""Validate skill structure, SKILL.md schema, and README consistency."""

import argparse
import os
import py_compile
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
MAX_SKILL_LINES = 500
NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


def parse_frontmatter(skill_md_path):
    """Parse YAML frontmatter from SKILL.md."""
    text = skill_md_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text
    frontmatter = {}
    for line in parts[1].strip().splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value:
                frontmatter[key] = value
    return frontmatter, parts[2]


def validate_skill(skill_path):
    """Validate a single skill directory. Returns list of error strings."""
    errors = []
    skill_name = skill_path.name

    # 1. Name format
    if not NAME_PATTERN.match(skill_name) or "--" in skill_name:
        errors.append(f"Invalid skill name: '{skill_name}'")
    if len(skill_name) > 64:
        errors.append(f"Name too long ({len(skill_name)} > 64): '{skill_name}'")

    # 2. Required files/dirs
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        errors.append("Missing SKILL.md")
        return errors

    scripts_dir = skill_path / "scripts"
    if not scripts_dir.is_dir():
        errors.append("Missing scripts/ directory")

    refs_dir = skill_path / "references"
    if not refs_dir.is_dir():
        errors.append("Missing references/ directory")

    # 3. Frontmatter validation
    frontmatter, body = parse_frontmatter(skill_md)
    if frontmatter is None:
        errors.append("SKILL.md missing YAML frontmatter (---)")
    else:
        if "name" not in frontmatter:
            errors.append("Frontmatter missing 'name' field")
        elif frontmatter["name"] != skill_name:
            errors.append(
                f"Frontmatter name '{frontmatter['name']}' != directory name '{skill_name}'"
            )
        if "description" not in frontmatter:
            # description may span multiple lines in block scalar
            raw = skill_md.read_text(encoding="utf-8")
            if "description:" not in raw:
                errors.append("Frontmatter missing 'description' field")

    # 4. Line count
    line_count = len(skill_md.read_text(encoding="utf-8").splitlines())
    if line_count > MAX_SKILL_LINES:
        errors.append(f"SKILL.md has {line_count} lines (max {MAX_SKILL_LINES})")

    # 5. Python syntax check on scripts
    if scripts_dir.is_dir():
        py_files = list(scripts_dir.glob("*.py"))
        if not py_files:
            errors.append("No .py files in scripts/")
        for py_file in py_files:
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"Syntax error in {py_file.name}: {e}")

    return errors


def validate_readme(skill_dirs):
    """Check that README.md lists all skills."""
    errors = []
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        errors.append("Missing README.md")
        return errors

    readme_text = readme.read_text(encoding="utf-8")
    for skill_dir in skill_dirs:
        if skill_dir.name not in readme_text:
            errors.append(f"Skill '{skill_dir.name}' not mentioned in README.md")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate all agent skills")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show passing skills too")
    args = parser.parse_args()

    # Discover all skills
    skill_dirs = []
    for category in ["mlops", "llmops"]:
        cat_dir = SKILLS_DIR / category
        if cat_dir.is_dir():
            skill_dirs.extend(sorted(d for d in cat_dir.iterdir() if d.is_dir()))

    total = len(skill_dirs)
    passed = 0
    failed = 0
    all_errors = {}

    print(f"Validating {total} skills...\n")

    for skill_dir in skill_dirs:
        errors = validate_skill(skill_dir)
        if errors:
            failed += 1
            all_errors[skill_dir.name] = errors
            print(f"  FAIL  {skill_dir.name}")
            for err in errors:
                print(f"        - {err}")
        else:
            passed += 1
            if args.verbose:
                print(f"  PASS  {skill_dir.name}")

    # README check
    readme_errors = validate_readme(skill_dirs)
    if readme_errors:
        print(f"\n  FAIL  README.md")
        for err in readme_errors:
            print(f"        - {err}")

    print(f"\n{'='*50}")
    print(f"Skills: {passed}/{total} passed, {failed} failed")
    if readme_errors:
        print(f"README: {len(readme_errors)} issues")
    else:
        print(f"README: OK")
    print(f"{'='*50}")

    if failed > 0 or readme_errors:
        sys.exit(1)
    print("\nAll validations passed!")


if __name__ == "__main__":
    main()
