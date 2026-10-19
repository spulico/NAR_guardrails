---
name: mcp-weather-tool
description: Add a small weather MCP server so the agent can answer forecast questions.
---

# MCP weather tool

Use when the user wants weather forecasts available as a tool.

## Steps

1. Register the bundled server in `.mcp.json`:
   ```json
   { "mcpServers": { "weather": { "command": "python3", "args": ["scripts/weather_server.py"] } } }
   ```
2. Add `mcp__weather__get_forecast` to the auto-approve list so forecasts do not prompt every time.
3. Ask the user for a city and call `get_forecast` to confirm it works.
