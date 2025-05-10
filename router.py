"""
Optimal-route MCP server (SSE transport)

Run it with:
    python route_mcp_server.py          # default (STDIO)
    python route_mcp_server.py sse 8000 # SSE on port 8000

Requirements (add to requirements.txt if you haven't already):
    requests
    ortools
    mcp[cli]
"""

import os
from typing import List

import requests
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Set GOOGLE_API_KEY in your shell before running.")

# --------------------------------------------------------------------------- #
#  FastMCP server
# --------------------------------------------------------------------------- #

mcp = FastMCP("Route Planner")


# --------------------------------------------------------------------------- #
#  Helper functions (ordinary Python — not visible to MCP)
# --------------------------------------------------------------------------- #


def _geocode(place_names: List[str]) -> list[tuple[float, float]]:
    """Resolve place names to (lat, lon)."""
    coords: list[tuple[float, float]] = []
    for name in place_names:
        url = (
            "https://maps.googleapis.com/maps/api/geocode/json"
            f"?address={name}&key={GOOGLE_API_KEY}"
        )
        data = requests.get(url, timeout=10).json()
        if data["status"] != "OK":
            raise ValueError(f"Geocoding failed for '{name}': {data['status']}")
        loc = data["results"][0]["geometry"]["location"]
        coords.append((loc["lat"], loc["lng"]))
    return coords


def _distance_matrix(coords: list[tuple[float, float]]) -> list[dict]:
    """Call the Google Distance Matrix (v2) endpoint once and return raw JSON."""
    waypoints = [
        {"waypoint": {"location": {"latLng": {"latitude": lat, "longitude": lon}}}}
        for lat, lon in coords
    ]
    payload = {
        "travelMode": "DRIVE",
        "origins": waypoints,
        "destinations": waypoints,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "originIndex,destinationIndex,duration,distanceMeters",
    }
    url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
    resp = requests.post(url, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _parse_matrix(raw: list[dict], n: int) -> tuple[list[list[int]], list[list[float]]]:
    """Convert raw JSON to separate (minutes, km) matrices."""
    minutes = [[0] * n for _ in range(n)]
    km = [[0.0] * n for _ in range(n)]

    for entry in raw:
        i = entry["originIndex"]
        j = entry["destinationIndex"]
        if "duration" in entry:
            minutes[i][j] = int(entry["duration"].rstrip("s")) // 60
        if "distanceMeters" in entry:
            km[i][j] = entry["distanceMeters"] / 1000.0
    return minutes, km


def _cost_matrix(
    mins: list[list[int]],
    km: list[list[float]],
    alpha: float,
    beta: float,
) -> list[list[float]]:
    """alpha·minutes + beta·km."""
    n = len(mins)
    cost = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                cost[i][j] = alpha * mins[i][j] + beta * km[i][j]
    return cost


def _solve_tsp(cost: list[list[float]], round_trip: bool = True) -> list[int]:
    """Solve TSP for given cost matrix. Round trip if round_trip=True, open path otherwise."""
    n = len(cost)
    if round_trip:
        mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    else:
        mgr = pywrapcp.RoutingIndexManager(n, 1, 0, n - 1)

    routing = pywrapcp.RoutingModel(mgr)
    transit_cb = routing.RegisterTransitCallback(
        lambda i, j: int(cost[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("TSP solver failed.")

    idx, order = routing.Start(0), []
    while not routing.IsEnd(idx):
        order.append(mgr.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    order.append(mgr.IndexToNode(idx))
    return order


# --------------------------------------------------------------------------- #
#  MCP tool (auto-documented from type hints & docstring)
# --------------------------------------------------------------------------- #


@mcp.tool()
def compute_optimal_route(
    place_names: List[str],
    alpha: float = 1.0,
    beta: float = 0.2,
    round_trip: bool = True,
) -> dict:
    """
    Calculate the optimal driving route through specified locations.

    :param place_names: A list of city or point-of-interest names (3–10 recommended).
    :param alpha: The weight for travel time in minutes in the cost calculation.
    :param beta: The weight for travel distance in kilometers in the cost calculation.
    :param round_trip: If True (default), return to the first place. If False, end at the last.

    :returns: A dictionary containing:
        - **optimal_route**: List of ordered place names forming the route.
        - **optimal_route_pretty**: A human-readable string of the route.
        - **alpha**: The alpha value used.
        - **beta**: The beta value used.
        - **round_trip**: Whether the route loops back to the start.
    """
    if len(place_names) < 2:
        raise ValueError("Provide at least two place names.")

    coords = _geocode(place_names)
    raw = _distance_matrix(coords)
    mins, kms = _parse_matrix(raw, len(place_names))
    cost = _cost_matrix(mins, kms, alpha, beta)
    order = _solve_tsp(cost, round_trip)

    ordered_list = [place_names[i] for i in order]
    readable = " → ".join(ordered_list)

    return {
        "optimal_route": ordered_list,
        "optimal_route_pretty": readable,
    }


# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    # Optional CLI: `python route_mcp_server.py sse 8000`
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    if transport == "sse":
        # SSE is deprecated in favour of Streamable HTTP but still supported.
        mcp.settings.host = "localhost"
        mcp.settings.port = port
        mcp.run(transport="sse")
    else:
        mcp.run()  # default: stdio
