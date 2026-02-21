#!/bin/bash
# test_skill_load.sh
# Tests if the gemini-cli skill is loaded properly by validating its structure

set -e

SKILL_DIR="$(dirname "$0")/.."
SKILL_FILE="$SKILL_DIR/SKILL.md"

echo "Running gemini-cli skill integration test..."

# 1. Check if SKILL.md exists
if [ ! -f "$SKILL_FILE" ]; then
    echo "❌ ERROR: SKILL.md not found in $SKILL_DIR"
    exit 1
fi
echo "✅ Configuration file SKILL.md exists."

# 2. Extract and validate YAML frontmatter
# This ensures Gemini framework can parse the metadata
FRONTMATTER=$(sed -n '1,/^---$/p' "$SKILL_FILE")

if echo "$FRONTMATTER" | grep -q "name: gemini-cli"; then
    echo "✅ Skill name 'gemini-cli' found in frontmatter."
else
    echo "❌ ERROR: Skill name 'gemini-cli' missing or malformed in frontmatter."
    exit 1
fi

if echo "$FRONTMATTER" | grep -q "description:"; then
    echo "✅ Skill description found in frontmatter."
else
    echo "❌ ERROR: Skill description missing in frontmatter."
    exit 1
fi

# 3. Verify Claude Context reference
if grep -q "CLAUDE.md" "$SKILL_FILE"; then
     echo "✅ CLAUDE.md reference found in instructions."
else
     echo "❌ ERROR: SKILL.md does not reference CLAUDE.md context."
     exit 1
fi

echo ""
echo "🎉 SUCCESS: gemini-cli skill structure is valid and ready for the framework."
exit 0
