# Colab Execution Setup (Codex)

This project supports two Colab execution paths:

1. MCP server (`mcp-server-colab-exec`) - preferred when available in session
2. CLI fallback (`colab-exec`) - always available path

## MCP server registration

Codex global config (`~/.codex/config.toml`) should include:

```toml
[mcp_servers.colab]
command = "/Users/paritoshdwivedi/.local/bin/mcp-server-colab-exec"
```

After editing config, restart Codex and verify the `colab` MCP tools are discoverable.

## CLI fallback runner

Use the helper script:

```bash
scripts/colab_run.sh -f /absolute/path/to/script.py --accelerator T4 --timeout 1200
```

Or inline code:

```bash
scripts/colab_run.sh -c "print('hello from colab')"
```

The script runs a short preflight check before the main command to avoid wasting long runs.
