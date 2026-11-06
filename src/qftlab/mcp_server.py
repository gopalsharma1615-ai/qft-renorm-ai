"""MCP server exposing qftlab's CAS tools to any MCP client (Claude Desktop / Claude Code / IDEs).

    qft-tools-mcp                      # stdio
    claude mcp add qft-tools -- qft-tools-mcp
"""
from __future__ import annotations

try:                                    # mcp >= 2: FastMCP was renamed MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import tools

mcp = _Server("qft-tools")


@mcp.tool()
def gamma_trace(indices: list[str]) -> str:
    """Trace of a product of 4D Dirac matrices: upper "mu", lower "-mu", e.g. ["mu","nu","-mu","-nu"]."""
    return tools.tool_gamma_trace(indices)


@mcp.tool()
def sandwich(k: int) -> str:
    """n-dimensional contraction γ^μ (k slashed vectors) γ_μ for k = 0..3 (n = 4 - eps)."""
    return tools.tool_sandwich(k)


@mcp.tool()
def master_integral(a: int, numerator: str = "1") -> dict:
    """One-loop master integral ∫ d^n l/(2π)^n N/(l² - Δ)^a, N in {"1", "l2"}, with its 1/eps pole."""
    return tools.tool_master_integral(a, numerator)


@mcp.tool()
def series_eps(expr: str, order: int = 0) -> str:
    """Laurent expansion of an expression in eps (n = 4 - eps) up to eps**order."""
    return tools.tool_series_eps(expr, order)


@mcp.tool()
def simplify(expr: str) -> str:
    """Simplify a symbolic expression (whitelisted names only)."""
    return tools.tool_simplify(expr)


@mcp.tool()
def qed_counterterms() -> dict:
    """One-loop QED counterterms δZ1, δZ2, δZ3, δm and β(e), β(α), derived symbolically."""
    return tools.tool_qed_counterterms()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
