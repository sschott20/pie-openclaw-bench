"""Synthetic prompt content generator for benchmark workloads.

Generates realistic prompt modules (core_instructions, tool_schemas, skills,
memory, conversation_history) at configurable token counts. Designed to be
replaced with real content later.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

from harness.models import PromptModule

# Approximate chars-per-token for English prose (conservative for Llama tokenizer)
CHARS_PER_TOKEN = 4


@dataclass
class PromptSizes:
    """Target token counts for each module type."""

    core_tokens: int = 2000
    tool_tokens: int = 8000
    skill_tokens: int = 4000
    memory_tokens: int = 2000
    history_tokens_per_turn: int = 300
    response_tokens: int = 200


# ---------------------------------------------------------------------------
# Template content for each module type. These are padded/truncated to hit
# target token counts while maintaining realistic tokenization.
# ---------------------------------------------------------------------------

CORE_INSTRUCTIONS_TEMPLATE = textwrap.dedent("""\
    You are an advanced AI coding assistant. Your primary purpose is to help \
    software engineers write, debug, refactor, and understand code. You have \
    access to tools that let you read files, write files, execute shell commands, \
    search codebases, and browse the web.

    ## Core Principles
    - Always read existing code before suggesting modifications
    - Prefer editing existing files over creating new ones
    - Keep solutions simple and focused on the task at hand
    - Write safe, secure code that avoids common vulnerabilities
    - Use appropriate error handling at system boundaries

    ## Response Format
    - Be concise and direct
    - Lead with the answer or action, not reasoning
    - Use code blocks with language annotations
    - Reference specific file paths and line numbers

    ## Tool Usage Guidelines
    - Use the most appropriate tool for each task
    - Read files before editing them
    - Search before assuming file locations
    - Run tests after making changes
    - Prefer dedicated tools over shell commands
""")

TOOL_SCHEMA_TEMPLATES = {
    "bash": {
        "name": "bash",
        "description": "Execute shell commands in a sandboxed environment. Use for system operations, builds, and running tests.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in milliseconds"},
                "working_dir": {"type": "string", "description": "Working directory"},
            },
            "required": ["command"],
        },
    },
    "read": {
        "name": "read",
        "description": "Read file contents from the filesystem. Supports text files, images, PDFs, and notebooks.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to file"},
                "offset": {"type": "integer", "description": "Line number to start from"},
                "limit": {"type": "integer", "description": "Number of lines to read"},
            },
            "required": ["file_path"],
        },
    },
    "write": {
        "name": "write",
        "description": "Write content to a file. Creates the file if it does not exist, overwrites if it does.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["file_path", "content"],
        },
    },
    "edit": {
        "name": "edit",
        "description": "Perform exact string replacements in files. The old_string must be unique.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to file"},
                "old_string": {"type": "string", "description": "Text to find and replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    "glob": {
        "name": "glob",
        "description": "Find files matching glob patterns. Returns paths sorted by modification time.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern like **/*.py"},
                "path": {"type": "string", "description": "Directory to search in"},
            },
            "required": ["pattern"],
        },
    },
    "grep": {
        "name": "grep",
        "description": "Search file contents using regular expressions. Supports context lines and filtering.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search"},
                "glob": {"type": "string", "description": "File pattern filter"},
                "context": {"type": "integer", "description": "Context lines around matches"},
            },
            "required": ["pattern"],
        },
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the web for current information. Returns relevant results with snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Max results to return"},
            },
            "required": ["query"],
        },
    },
    "web_fetch": {
        "name": "web_fetch",
        "description": "Fetch and read content from a URL. Returns the text content of the page.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
            },
            "required": ["url"],
        },
    },
}

SKILL_TEMPLATES = {
    "code_review": textwrap.dedent("""\
        # Code Review Skill
        When reviewing code changes, follow this structured approach:
        1. Check for correctness: Does the code do what it claims?
        2. Check for security: Are there injection risks, auth bypasses, or data leaks?
        3. Check for performance: Are there N+1 queries, unnecessary allocations, or blocking calls?
        4. Check for maintainability: Is the code readable, well-structured, and documented?
        5. Check test coverage: Are edge cases tested? Are tests meaningful?
        Provide specific, actionable feedback with line references.
    """),
    "web_search_research": textwrap.dedent("""\
        # Web Search Research Skill
        When researching a topic, use systematic web search:
        1. Start with broad queries to understand the landscape
        2. Narrow down with specific technical queries
        3. Cross-reference multiple sources for accuracy
        4. Prefer official documentation over blog posts
        5. Check publication dates for currency
        Summarize findings with source attribution.
    """),
    "debugging": textwrap.dedent("""\
        # Systematic Debugging Skill
        When debugging issues, follow this methodology:
        1. Reproduce the issue reliably with a minimal test case
        2. Form a hypothesis about the root cause
        3. Gather evidence through logging, profiling, or stepping
        4. Test the hypothesis with a targeted fix
        5. Verify the fix resolves the issue without regressions
        Never apply fixes without understanding the root cause.
    """),
    "test_writing": textwrap.dedent("""\
        # Test Writing Skill
        When writing tests, follow test-driven development:
        1. Write failing tests first that describe expected behavior
        2. Implement the minimum code to make tests pass
        3. Refactor while keeping tests green
        4. Cover edge cases: empty input, boundary values, error conditions
        5. Use descriptive test names that explain what and why
        Tests should be fast, independent, and deterministic.
    """),
    "architecture_planning": textwrap.dedent("""\
        # Architecture Planning Skill
        When designing system architecture:
        1. Identify core requirements and constraints
        2. Propose 2-3 approaches with trade-offs
        3. Consider scalability, maintainability, and operational complexity
        4. Define clear interfaces between components
        5. Plan for failure modes and recovery
        Document decisions and rationale for future reference.
    """),
}

MEMORY_ENTRIES = [
    "User prefers TypeScript over JavaScript for new projects.",
    "Project uses React 19 with server components.",
    "Database is PostgreSQL 16 with pgvector extension.",
    "CI/CD runs on GitHub Actions with required checks.",
    "User uses pytest with pytest-asyncio for Python tests.",
    "Preferred code style: functional where possible, classes for stateful components.",
    "Project deployment target is Kubernetes on AWS EKS.",
    "User has a B200 GPU available on RunPod for ML workloads.",
    "Last successful deployment was March 8, 2026.",
    "Auth system uses JWT with 15-minute access tokens.",
    "Rate limiting is configured at 100 req/min per user.",
    "Monitoring uses Prometheus + Grafana stack.",
]

CONVERSATION_TURNS = [
    ("user", "Can you help me optimize this database query? It's taking 3 seconds."),
    ("assistant", "I'll look at the query. Let me read the relevant code first."),
    ("user", "It's in src/db/queries.py, the get_user_activity function."),
    ("assistant", "I see the issue — there's a missing index on the activity.user_id column, and the query is doing a sequential scan. Let me add the index and also rewrite the query to use a CTE for the aggregation."),
    ("user", "That brought it down to 50ms. Can you also add caching?"),
    ("assistant", "I'll add a Redis cache with a 5-minute TTL for the aggregated results, with cache invalidation on new activity writes."),
    ("user", "Can you also write tests for the caching behavior?"),
    ("assistant", "I'll write tests covering: cache hit, cache miss, cache invalidation on write, and TTL expiry. Using pytest with a mock Redis client."),
    ("user", "The tests are passing but CI is failing on the linting step."),
    ("assistant", "The linter is flagging unused imports in the test file. Let me fix those."),
    ("user", "Thanks, CI is green now. Can you update the docs?"),
    ("assistant", "I'll update the API documentation to reflect the new caching behavior and the query optimization."),
]

TOOL_RESULTS = [
    "File contents of src/main.py (45 lines):\n```python\nimport asyncio\nfrom pathlib import Path\n\nasync def main():\n    config = load_config()\n    server = await create_server(config)\n    await server.start()\n```",
    "Command output:\n$ pytest tests/ -v\n====== 12 passed, 0 failed in 2.34s ======",
    "Search results for 'database connection pool':\n1. SQLAlchemy Connection Pooling - docs.sqlalchemy.org\n2. PostgreSQL Connection Pooling with PgBouncer - pgbouncer.org\n3. Best practices for database connections - blog.example.com",
    "grep results for 'def get_user' in src/:\nsrc/db/queries.py:45: def get_user_activity(user_id: int) -> list[Activity]:\nsrc/api/routes.py:123: def get_user_profile(request: Request) -> Response:",
]


def _pad_to_tokens(text: str, target_tokens: int) -> str:
    """Pad or truncate text to approximately hit target token count."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    if len(text) >= target_chars:
        return text[:target_chars]
    # Pad by repeating content with variation
    padding_needed = target_chars - len(text)
    padding_unit = "\nAdditional context and guidelines continue below.\n" + text[:200]
    repetitions = padding_needed // len(padding_unit) + 1
    padded = text + (padding_unit * repetitions)
    return padded[:target_chars]


def make_core_instructions(target_tokens: int = 500) -> PromptModule:
    """Generate a core instructions module at the target token count."""
    content = _pad_to_tokens(CORE_INSTRUCTIONS_TEMPLATE, target_tokens)
    return PromptModule(name="core_instructions", content=content)


def make_tool_schemas(
    target_tokens: int = 2000,
    tool_names: list[str] | None = None,
) -> PromptModule:
    """Generate a tool schemas module with specified tools."""
    if tool_names is None:
        tool_names = list(TOOL_SCHEMA_TEMPLATES.keys())

    import json

    schemas_text = "## Available Tools\n\n"
    for name in tool_names:
        if name in TOOL_SCHEMA_TEMPLATES:
            schema = TOOL_SCHEMA_TEMPLATES[name]
            schemas_text += f"### {schema['name']}\n"
            schemas_text += f"{schema['description']}\n"
            schemas_text += f"```json\n{json.dumps(schema['parameters'], indent=2)}\n```\n\n"

    content = _pad_to_tokens(schemas_text, target_tokens)
    return PromptModule(name="tool_schemas", content=content)


def make_skill(
    skill_name: str,
    target_tokens: int = 1500,
) -> PromptModule:
    """Generate a skill module."""
    template = SKILL_TEMPLATES.get(skill_name, SKILL_TEMPLATES["code_review"])
    content = _pad_to_tokens(template, target_tokens)
    return PromptModule(name=f"skill_{skill_name}", content=content)


def make_memory(target_tokens: int = 200) -> PromptModule:
    """Generate a memory block module."""
    text = "## Agent Memory\n\n"
    for entry in MEMORY_ENTRIES:
        text += f"- {entry}\n"
    content = _pad_to_tokens(text, target_tokens)
    return PromptModule(name="memory", content=content)


def make_conversation_history(
    num_turns: int = 4,
    tokens_per_turn: int = 150,
) -> PromptModule:
    """Generate a conversation history module."""
    text = "## Conversation History\n\n"
    for i in range(min(num_turns, len(CONVERSATION_TURNS))):
        role, msg = CONVERSATION_TURNS[i % len(CONVERSATION_TURNS)]
        text += f"**{role}**: {msg}\n\n"

    # If more turns needed, cycle through with variation
    for i in range(len(CONVERSATION_TURNS), num_turns):
        role = "user" if i % 2 == 0 else "assistant"
        base_msg = CONVERSATION_TURNS[i % len(CONVERSATION_TURNS)][1]
        text += f"**{role}** (turn {i}): {base_msg}\n\n"

    target_tokens = num_turns * tokens_per_turn
    content = _pad_to_tokens(text, target_tokens)
    return PromptModule(name="conversation_history", content=content)


def make_tool_result(tool_name: str = "bash") -> PromptModule:
    """Generate a tool result module."""
    import random

    result = random.choice(TOOL_RESULTS)
    content = f"## Tool Result ({tool_name})\n\n{result}"
    return PromptModule(name=f"tool_result_{tool_name}", content=content)


def make_user_message(message: str = "Please help me with this task.") -> PromptModule:
    """Generate a user message module."""
    return PromptModule(name="user_message", content=message)


def make_heartbeat_prompt() -> PromptModule:
    """Generate a heartbeat check prompt."""
    return PromptModule(
        name="heartbeat_prompt",
        content="Check for any pending tasks, notifications, or scheduled actions. "
        "Report status and any items requiring attention.",
    )


def make_updated_memory(base_tokens: int = 200, update_index: int = 0) -> PromptModule:
    """Generate a memory module with a small update (for heartbeat scenario)."""
    text = "## Agent Memory\n\n"
    for i, entry in enumerate(MEMORY_ENTRIES):
        if i == update_index % len(MEMORY_ENTRIES):
            text += f"- [UPDATED] {entry} (verified at turn {update_index})\n"
        else:
            text += f"- {entry}\n"
    content = _pad_to_tokens(text, base_tokens)
    return PromptModule(name="memory", content=content)
